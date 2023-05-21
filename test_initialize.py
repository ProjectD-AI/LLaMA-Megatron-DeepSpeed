# test_initialize.py
import sys
sys.path.append("..")

import megatron.mpu as mpu
import torch
import argparse
import os
import random
import numpy
from deepspeed.accelerator import get_accelerator


def initialize_distributed(backend='nccl'):
    """Initialize torch.distributed."""
    # Get local rank in case it is provided.
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher')
    args = parser.parse_args()
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    print('> initializing torch.distributed with local rank: {}, '
          'rank: {}, world size: {}'.format(local_rank, rank, world_size))

    # Set the device id.
    device = rank % get_accelerator().device_count()
    if local_rank is not None:
        device = local_rank
    get_accelerator().set_device(device)

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method)


def print_separator(message):
    torch.distributed.barrier()
    filler_len = (78 - len(message)) // 2
    filler = '-' * filler_len
    string = '\n' + filler + ' {} '.format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
    torch.distributed.barrier()

def run_test(
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size:int):
    print_separator(f'> Test: TP={tensor_model_parallel_size}, PP={pipeline_model_parallel_size}')
    mpu.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size) # 并行初始化
    world_size = torch.distributed.get_world_size() # world_size, 总GPU数量
    global_rank = torch.distributed.get_rank() # 当前GPU的编号
    tp_world_size = mpu.get_tensor_model_parallel_world_size() # 每个张量并行组中GPU的数量
    pp_world_size = mpu.get_pipeline_model_parallel_world_size() # 每个流水线并行组中GPU的数量
    dp_world_size = mpu.get_data_parallel_world_size() # 每个数据并行组中的GPU数量
    tp_rank = mpu.get_tensor_model_parallel_rank() # 在张量并行组中的编号
    pp_rank = mpu.get_pipeline_model_parallel_rank() # 在流水线并行组中的编号
    dp_rank = mpu.get_data_parallel_rank() # 在数据并行组中的编号
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_group = torch.distributed.distributed_c10d._pg_group_ranks[tp_group] # 当前GPU所在张量并行组的映射字典
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_group = torch.distributed.distributed_c10d._pg_group_ranks[pp_group] # 当前GPU所在流水线并行组的映射字典
    dp_group = mpu.get_data_parallel_group()
    dp_group = torch.distributed.distributed_c10d._pg_group_ranks[dp_group] # 当前GPU所在数据并行组的映射字典
    torch.distributed.barrier()
    info = f"="*20 + \
           f"\n> global_rank={global_rank}\n" + \
           f"> world_size={world_size}\n" + \
           f"> tp_world_size={tp_world_size}\n" + \
           f"> pp_world_size={pp_world_size}\n" + \
           f"> dp_world_size={dp_world_size}\n" + \
           f"> tp_rank={tp_rank}\n" + \
           f"> pp_rank={pp_rank}\n" + \
           f"> dp_rank={dp_rank}\n" + \
           f"> tp_group={tp_group}\n" + \
           f"> pp_group={pp_group}\n" + \
           f"> dp_group={dp_group}\n"
    print(info, flush=True)
    torch.distributed.barrier()

if __name__ == '__main__':
    initialize_distributed() # 初始化分布式环境
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 2
    run_test(tensor_model_parallel_size, pipeline_model_parallel_size)