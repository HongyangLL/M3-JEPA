import logging
import sys
import yaml
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from src.models.mm_jepa import init_mmjepa
from src.datasets.IN1K import DatasetManager
from src.helper import load_checkpoint, init_opt
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.optimizer import EarlyStopping
from src.cls_evaluation import evaluate_cls_model

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(local_rank, args):
    """Main training and evaluation function for distributed multimodal JEPA model."""
    device = torch.device(f'cuda:{local_rank}')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Extract configuration
    use_bfloat16 = args['meta']['use_bfloat16']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    num_epochs = args['optimization']['epochs']
    patience = args['optimization']['patience']
    t2i_wd = float(args['optimization']['t2i_weight_decay'])
    t2i_final_wd = float(args['optimization']['t2i_final_weight_decay'])
    t2i_warmup = args['optimization']['t2i_warmup']
    t2i_start_lr = args['optimization']['t2i_start_lr']
    t2i_lr = args['optimization']['t2i_lr']
    t2i_final_lr = args['optimization']['t2i_final_lr']
    ipe_scale = args['optimization']['ipe_scale']
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    tb_path = args['tb']['log_dir']
    port = args['tb']['port']
    imag_encoder_name = args['models']['imag_encoder_name']
    imag_encoder_name_1 = args['models']['imag_encoder_name_1']
    text_encoder_name = args['models']['text_encoder_name']
    num_experts = args['MoE']['num_experts']
    k_expert = args['MoE']['k']
    drop_rate = args['MoE']['drop']
    predictor_type = args['predictor']
    use_cross_att = args['use_cross_att']
    num_head = args['cross_att']['num_head']
    num_layers = args['cross_att']['num_layers']
    test_model = args['test_model']
    hidden_size = args['MoE']['hidden_size'] if predictor_type == 'moe_mlp' else None

    # Setup logging and checkpoints
    folder = os.path.join(folder, datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{global_rank}.csv')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%d', 'time (ms)'))
    with open(os.path.join(folder, 'params-mmjepa.yaml'), 'w') as f:
        yaml.dump(args, f)

    # Setup TensorBoard
    if global_rank == 0:
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(tb_path)
        logger.info(f"TensorBoard running at {tb_path} on port {port}")

    # Initialize model
    mm_jepa = init_mmjepa(
        imag_encoder_name, imag_encoder_name_1, text_encoder_name, num_experts, k_expert, drop_rate,
        predictor_type, num_head, num_layers, use_cross_att, test_model, hidden_size=hidden_size
    )
    mm_jepa = DistributedDataParallel(mm_jepa.to(device), device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f"Model parameters: {sum(p.numel() for p in mm_jepa.parameters() if p.requires_grad)}")

    # Initialize data loaders
    data_manager = DatasetManager(
        batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers, world_size=world_size, rank=global_rank,
        root_path=root_path, imag_encoder_name=imag_encoder_name, imag_encoder_name_1=imag_encoder_name_1,
        text_encoder_name=text_encoder_name
    )
    train_set, test_set, train_loader, test_loader, train_sampler, test_sampler = data_manager.DDP_loader()
    ipe = len(train_loader)

    # Initialize optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        mm_jepa=mm_jepa, wd=t2i_wd, final_wd=t2i_final_wd, start_lr=t2i_start_lr, ref_lr=t2i_lr,
        final_lr=t2i_final_lr, iterations_per_epoch=ipe, warmup=t2i_warmup, num_epochs=num_epochs,
        ipe_scale=ipe_scale, use_bfloat16=use_bfloat16
    )
    early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, maximize=False)

    # Training loop
    criterion = nn.CrossEntropyLoss(reduction='mean')
    min_loss = float('inf')
    log_freq = 10

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        logger.info(f'Epoch {epoch + 1}')

        for itr, sample in enumerate(train_loader):
            imgs = sample['image_features'].to(device, non_blocking=True)
            imgs_1 = sample['image_features_1'].to(device, non_blocking=True)
            input_ids = sample['text_features']['input_ids'].to(device, non_blocking=True)
            attention_mask = sample['text_features']['attention_mask'].to(device, non_blocking=True)
            labels = sample['labels'].to(device, non_blocking=True)

            def train_step():
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    logits = mm_jepa(input_ids, imgs, imgs_1, labels)
                    cls_loss = criterion(logits, labels)
                    dist.all_reduce(cls_loss, op=dist.ReduceOp.SUM)
                    cls_loss /= world_size
                    loss = cls_loss
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(mm_jepa.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mm_jepa.parameters(), max_norm=1.0)
                    optimizer.step()
                return float(loss), grad_logger(mm_jepa.named_parameters())

            loss, grad_stats, etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']
            scheduler.step()
            wd_scheduler.step(optimizer)

            # Logging
            csv_logger.log(epoch + 1, itr, loss, etime)
            if itr % log_freq == 0 or np.isnan(loss):
                logger.info(
                    f'[{epoch}, {itr}] Loss: {loss_meter.avg:.5f} | WD: {current_wd:.2e} | LR: {current_lr:.2e} | '
                    f'Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2e} MB | Time: {time_meter.avg:.1f}ms'
                )
                if global_rank == 0:
                    writer.add_scalar('Loss/train', itr, loss.item())

            assert not np.isnan(), 'Loss is NaN'

        logger.info(f'Epoch {epoch+1}, Avg Loss: {loss_meter.avg:.4f}')
        if global_rank == 0:
            writer.add_scalar('Loss/avg', loss_meter.avg, epoch)

        # Save best model
        if loss_meter.avg <= min_loss:
            min_loss = loss.item()
            if global_rank == 0:
                torch.save(
                    {'mm_jepa_0': mm_jepa.state_dict(), 'opt': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None, 'epoch': epoch + 1},
                    best_path
                )
                logger.info('Saved best model')

        # Early stopping
        early_stop(loss_meter.avg)
        if early_stop.should_stop():
            logger.info('Early stopping triggered')
            break

    # Evaluation
    logger.info('Evaluating on test set...')
    mm_jepa.load_state_dict(torch.load(best_path, map_location=device)['mm_jepa'])
    mm_jepa = DistributedDataParallel(mm_jepa.to(device), device_ids=[local_rank], find_unused_parameters=True)
    results = evaluate_cls_model(mm_jepa, test_loader, device, mode='test', global_rank=global_rank)
    logger.info(f'Classification results: {results}')

    torch.cuda.empty_cache()
    if global_rank == 0:
        writer.close()
