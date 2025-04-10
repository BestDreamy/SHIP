import sys
import os
import torch
# import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../psrc')))

from utils import MoEConfig, ProcessGroupInfo, RankTestData
from utils import init_dist

from Buffer import Buffer

# logger = logging.getLogger(__name__)

moe = MoEConfig(
    num_experts=8,
    experts_per_token=3,
    hidden_dim=512,
    max_num_tokens=10,
)

def _str_1d_tensor(t: torch.Tensor) -> str:
    sl = [f"{x:7.4f}" for x in t.tolist()]
    if len(sl) > 5:
        sl = sl[:5] + ["..."]
    return "[" + ", ".join(sl) + "]"

def work(
    rank: int,
    world_size: int,
    dp_size: int,
) -> None:
    pgi = init_dist(rank, world_size)
    # rank = pgi.rank
    # local_rank = pgi.local_rank
    # world_size = pgi.world_size
    dp_rank = rank // dp_size
    num_dp = world_size // dp_size
    # assert torch.cuda.current_device() == rank
    # device = pgi.device

    ata = Buffer (
        max_num_tokens=moe.max_num_tokens,
        num_experts=moe.num_experts,
        experts_per_token=moe.experts_per_token,
        rank=rank,
        world_size=world_size,
        dp_size=dp_size,
        hidden_dim=moe.hidden_dim,
        hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
        hidden_dim_scale_bytes=(
            0
            if moe.in_dtype.itemsize != 1
            else (
                (moe.hidden_dim + moe.block_size - 1)
                // moe.block_size
                * torch.float32.itemsize
            )
        ),
    )

    print(f"rank: {rank}")

    # Generate the same test data on all ranks
    rng = torch.Generator(device='cuda')
    rng.manual_seed(123)
    all_rank_data = [
        RankTestData(moe, rng, use_max_tokens=False) for _ in range(num_dp)
    ]
    rank_data = all_rank_data[dp_rank]

    # Collect info by expert
    expert_token_from: list[list[tuple[int, int]]] = [
        [] for _ in range(moe.num_experts)
    ]
    for i_rank, rd in enumerate(all_rank_data):
        for token_idx in range(rd.num_tokens):
            for expert_idx in rd.indices[token_idx]:
                expert_token_from[expert_idx].append((i_rank, token_idx))

    for i, rd in enumerate(all_rank_data):
        print(f"  DP Rank {i}:")
        for token_idx in range(rd.num_tokens):
            indices = rd.indices[token_idx].tolist()
            weights = rd.weights[token_idx].tolist()
            print("    x[%d] -> %s", token_idx, list(zip(indices, weights)))
        for token_idx in range(rd.num_tokens):
            print("    x[%d]=%s", token_idx, _str_1d_tensor(rd.x[token_idx]))
        if rd.x_scale is not None:
            for token_idx in range(rd.num_tokens):
                print(
                    "    x_scale[%d]=%s",
                    token_idx,
                    _str_1d_tensor(rd.x_scale[token_idx]),
                )
    for expert_idx in range(moe.num_experts):
        print(
            "  Expert %d: %d tokens, from: %s",
            expert_idx,
            len(expert_token_from[expert_idx]),
            [f"r{r}t{t}" for r, t in expert_token_from[expert_idx]],
        )

if __name__ == '__main__':
    dp_size = 2
    world_size = 8
    torch.multiprocessing.spawn(work, args=(world_size, dp_size), nprocs=world_size)
