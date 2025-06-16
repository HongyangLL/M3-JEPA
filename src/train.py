import logging
import sys
import yaml
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from src.models.mm_jepa import init_mmjepa
from src.datasets.coco import DatasetManager, make_test_set
from src.helper import load_checkpoint, init_opt
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.optimizer import EarlyStopping
from src.evaluation import dual_embed, log_recall_rates, log_tensorboard_metrics
from src.utils.TaskProbabilityAdjuster import init_ProbabilityAdjuster

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(local_rank, args):
    """Main function for distributed multimodal JEPA training with proxy tasks."""
    # Set random seed and CUDA settings
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    except KeyError:
        pass

    device = torch.device(f'cuda:{local_rank}')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Extract configuration
    use_bfloat16 = args['meta']['use_bfloat16']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    val_split = args['eva']['val_spilit']
    sim_c = args['eva']['smilariity']
    num_epochs = args['optimization']['epochs']
    patience = args['optimization']['patience']
    t2i_wd = float(args['optimization']['t2i_weight_decay'])
    t2i_final_wd = float(args['optimization']['t2i_final_weight_decay'])
    t2i_warmup = args['optimization']['t2i_warmup']
    t2i_start_lr = args['optimization']['t2i_start_lr']
    t2i_lr = args['optimization']['t2i_lr']
    t2i_final_lr = args['optimization']['t2i_final_lr']
    i2t_wd = float(args['optimization']['i2t_weight_decay'])
    i2t_final_wd = float(args['optimization']['i2t_final_weight_decay'])
    i2t_warmup = args['optimization']['i2t_warmup']
    i2t_start_lr = args['optimization']['i2t_start_lr']
    i2t_lr = args['optimization']['i2t_lr']
    i2t_final_lr = args['optimization']['i2t_final_lr']
    ipe_scale = args['optimization']['ipe_scale']
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    tb_path = args['tb']['log_dir']
    port = args['tb']['port']
    imag_encoder_name = args['models']['imag_encoder_name']
    text_encoder_name = args['models']['text_encoder_name']
    num_experts = args['MoE']['num_experts']
    k_expert = args['MoE']['k']
    drop_rate = args['MoE']['drop']
    predictor_type = args['predictor']
    hidden_size = args['MoE']['hidden_size'] if predictor_type == 'moe_mlp' else args['MLP']['hidden_size']
    proxy_tasks = args['proxy_tasks']
    iter_task = args['iter_task']
    prob_samp = args['prob_samp']
    base_prob = args['task_prob']['base_prob']
    max_prob = args['task_prob']['max_prob']
    min_prob = args['task_prob']['min_prob']
    step = args['task_prob']['step']
    total_step = args['task_prob']['total_step']

    # Setup logging and checkpoints
    folder = os.path.join(folder, datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{global_rank}.csv')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    with open(os.path.join(folder, 'params-mmjepa.yaml'), 'w') as f:
        yaml.dump(args, f)
    csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%d', 'time (ms)'))

    # Setup TensorBoard
    writer = None
    if global_rank == 0:
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(tb_path)
        logger.info(f"TensorBoard at {tb_path}, port {port}")

    # Initialize model
    mm_jepa = init_mmjepa(imag_encoder_name, text_encoder_name, num_experts, hidden_size, k_expert, drop_rate, predictor_type)
    mm_jepa = DistributedDataParallel(mm_jepa.to(device), device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f"Model parameters: {sum(p.numel() for p in mm_jepa.parameters() if p.requires_grad)}")

    # Initialize data loaders
    data_manager = DatasetManager(
        batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers, world_size=world_size, rank=global_rank,
        root_path=root_path, image_folder=image_folder, val_split=val_split, imag_encoder_name=imag_encoder_name,
        text_encoder_name=text_encoder_name
    )
    train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = data_manager.resplit_data()
    test_set, test_loader, test_sampler = make_test_set(
        batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers, world_size=world_size, rank=global_rank,
        root_path=root_path, image_folder=image_folder, imag_encoder_name=imag_encoder_name, text_encoder_name=text_encoder_name
    )
    ipe = len(train_loader)

    # Initialize optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        mm_jepa=mm_jepa, wd=t2i_wd, final_wd=t2i_final_wd, start_lr=t2i_start_lr, ref_lr=t2i_lr,
        final_lr=t2i_final_lr, iterations_per_epoch=ipe, warmup=t2i_warmup, num_epochs=num_epochs,
        ipe_scale=ipe_scale, use_bfloat16=use_bfloat16
    )
    task_opt_params = {
        'image2text': {'start_lr': i2t_start_lr, 'ref_lr': i2t_lr, 'final_lr': i2t_final_lr, 'warmup': i2t_warmup, 'wd': i2t_wd, 'final_wd': i2t_final_wd},
        'text2image': {'start_lr': t2i_start_lr, 'ref_lr': t2i_lr, 'final_lr': t2i_final_lr, 'warmup': t2i_warmup, 'wd': t2i_wd, 'final_wd': t2i_final_wd}
    }
    early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, maximize=True)
    prob_adjuster = init_ProbabilityAdjuster(base_prob, max_prob, min_prob, step)
    task_iterator = itertools.cycle(proxy_tasks) if iter_task else None

    def sync_task_selection(task_prob):
        weights = task_prob
        task_index = torch.tensor([0], dtype=torch.long, device=device)
        if global_rank == 0:
            current_task = random.choices(proxy_tasks, weights=weights, k=1)[0]
            task_index[0] = proxy_tasks.index(current_task)
        dist.broadcast(task_index, src=0)
        return proxy_tasks[task_index.item()]

    # Training loop
    best_recall = 0.0
    log_freq = 10
    ks = [1, 5, 10]
    avg_i2t_recall = avg_t2i_recall = None

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        logger.info(f'Epoch {epoch + 1}')

        if prob_samp:
            i2t_prob, t2i_prob, decayed_step = prob_adjuster.update_probabilities_dec(
                recall_a=avg_i2t_recall, recall_b=avg_t2i_recall, current_epoch=epoch, total_epochs=total_step
            )
            logger.info(f'i2t_prob: {i2t_prob:.3f}, t2i_prob: {t2i_prob:.3f}, step: {decayed_step}')

        for itr, sample in enumerate(train_loader):
            imgs = sample['image_features'].to(device, non_blocking=True)
            input_ids = sample['text_features']['input_ids'].to(device, non_blocking=True)
            attention_mask = sample['text_features']['attention_mask'].to(device, non_blocking=True)

            def train_step():
                optimizer.zero_grad()
                task = sync_task_selection([i2t_prob, t2i_prob]) if prob_samp else next(task_iterator) if iter_task else proxy_tasks[0]
                if itr % log_freq == 0:
                    logger.info(f'Itr {itr}, Task: {task}')

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    if prob_samp:
                        adjust_optimizer_params(optimizer, task, task_opt_params)
                    loss, *_ = mm_jepa(input_ids, attention_mask, imgs, task)

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

                return float(loss), scheduler.step(), wd_scheduler.step()

            loss, new_lr, new_wd, etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # Logging
            csv_logger.log(epoch + 1, itr, loss, etime)
            if itr % log_freq == 0 or np.isnan(loss):
                logger.info(
                    f'[{epoch + 1}, {itr}] Loss: {loss_meter.avg:.3f} | WD: {new_wd:.2e} | LR: {new_lr:.2e} | '
                    f'Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2e} MB | Time: {time_meter.avg:.1f} ms'
                )
                if global_rank == 0:
                    writer.add_scalar('Loss/train', loss, epoch * ipe + itr)
            assert not np.isnan(loss), 'Loss is NaN'

        logger.info(f'Epoch {epoch + 1}, Avg Loss: {loss_meter.avg:.3f}')
        if global_rank == 0:
            writer.add_scalar('Loss/avg', loss_meter.avg, epoch)

        # Validation
        logger.info(f'Epoch {epoch + 1} validation')
        pred_text_emb, true_text_emb, pred_imag_emb, true_imag_emb, i2t_cl_imag_emb, i2t_cl_text_emb, t2i_cl_imag_emb, t2i_cl_text_emb = dual_embed(
            proxy_tasks, val_loader, device, mm_jepa
        )
        should_stop, t2i_image_recalls, i2t_text_recalls, i2t_cl_text_recall, t2i_cl_image_recall, avg_i2t_recall, avg_t2i_recall, avg_recall = log_recall_rates(
            folder, global_rank, world_size, pred_text_emb, true_text_emb, pred_imag_emb, true_imag_emb,
            i2t_cl_imag_emb, i2t_cl_text_emb, t2i_cl_imag_emb, t2i_cl_text_emb, ks, device, epoch, sim_c, early_stop, 'val', proxy_tasks
        )
        if global_rank == 0:
            log_tensorboard_metrics(writer, ks, t2i_image_recalls, i2t_text_recalls, epoch)
            log_tensorboard_metrics(writer, ks, t2i_cl_image_recall, i2t_cl_text_recall, epoch)

        # Save best model
        if avg_recall > best_recall and global_rank == 0:
            best_recall = avg_recall
            torch.save({
                'mm_jepa': mm_jepa.state_dict(), 'opt': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None, 'epoch': epoch + 1,
                'loss': loss_meter.avg, 'batch_size': batch_size, 'world_size': world_size,
                't2i_lr': t2i_lr, 'i2t_lr': i2t_lr, 'best_recall': best_recall
            }, best_path)
            logger.info('Saved best model')

        if should_stop:
            logger.info('Early stopping triggered')
            break

    # Test
    logger.info(f'Test: Loading best model from {best_path}')
    mm_jepa, optimizer, scaler, _ = load_checkpoint(device, best_path, mm_jepa, optimizer, scaler)
    pred_text_emb, true_text_emb, pred_imag_emb, true_imag_emb, i2t_cl_imag_emb, i2t_cl_text_emb, t2i_cl_imag_emb, t2i_cl_text_emb = dual_embed(
        proxy_tasks, test_loader, device, mm_jepa
    )
    should_stop, t2i_image_recalls, i2t_text_recalls, i2t_cl_text_recall, t2i_cl_image_recall, *_ = log_recall_rates(
        folder, global_rank, world_size, pred_text_emb, true_text_emb, pred_imag_emb, true_imag_emb,
        i2t_cl_imag_emb, i2t_cl_text_emb, t2i_cl_imag_emb, t2i_cl_text_emb, ks, device, epoch, sim_c, early_stop, 'test', proxy_tasks
    )
    if global_rank == 0:
        log_tensorboard_metrics(writer, ks, t2i_image_recalls, i2t_text_recalls, epoch)
        log_tensorboard_metrics(writer, ks, t2i_cl_image_recall, i2t_cl_text_recall, epoch)
        writer.close()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
