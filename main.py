import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import logging
import yaml
import os
import time
from src.train import main as app_main

def init_distributed(args):
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{args.master_addr}:{args.master_port}',
        world_size=args.nnodes * args.nproc_per_node,
        rank=args.node_rank * args.nproc_per_node + int(os.environ['LOCAL_RANK'])
    )
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def process_main(args):
    local_rank = init_distributed(args)

    logging.basicConfig(level=logging.INFO if dist.get_rank() == 0 else logging.ERROR)
    logger = logging.getLogger()

    logger.info(f'Loading config: {args.fname}')
    try:
        with open(args.fname, 'r') as y_file:
            params = yaml.safe_load(y_file)
        logger.info('Config loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load config: {e}')
        cleanup()
        raise

    start_time = time.time()
    app_main(local_rank, params)
    elapsed_time = time.time() - start_time

    days, rem = divmod(elapsed_time, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = round(rem / 60)
    logger.info(f'Training time: {int(days)} Days {int(hours)} Hours {minutes} Minutes')
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed training script')
    parser.add_argument('--fname', type=str, default='configs/mlp_moe.yaml', help='Config file path')
    parser.add_argument('--nnodes', type=int, required=True, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, required=True, help='Node rank')
    parser.add_argument('--nproc_per_node', type=int, required=True, help='Processes per node')
    parser.add_argument('--master_addr', type=str, required=True, help='Master node address')
    parser.add_argument('--master_port', type=str, required=True, help='Master node port')
    args = parser.parse_args()
    process_main(args)
