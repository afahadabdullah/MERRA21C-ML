"""Confirm all allocated GPUs can communicate before loading the training archive."""
import os
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    try:
        value = torch.tensor([rank], dtype=torch.int32, device=f'cuda:{local}')
        dist.all_reduce(value)
        torch.cuda.synchronize(local)
        expected = world*(world-1)//2
        if value.item() != expected:
            raise RuntimeError(f'DDP preflight expected {expected}, got {value.item()}')
        print(f'DDP preflight rank {rank}/{world}: cuda:{local}, all-reduce={value.item()}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
