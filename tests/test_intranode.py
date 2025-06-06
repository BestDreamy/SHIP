import sys
import os
import torch
# import logging

from ..psrc.utils import MoEConfig, ProcessGroupInfo, RankTestData
from ..psrc.utils import init_dist

from ..psrc.Buffer import Buffer

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
) -> None:
    pgi = init_dist(rank, world_size)
    num_rank = world_size
    # rank = pgi.rank
    # local_rank = pgi.local_rank
    # world_size = pgi.world_size
    # assert torch.cuda.current_device() == rank
    device = pgi.device

    ata = Buffer (
        max_num_tokens=moe.max_num_tokens,
        num_experts=moe.num_experts,
        experts_per_token=moe.experts_per_token,
        rank=rank,
        world_size=world_size,
        hidden_dim=moe.hidden_dim,
        # hidden_dim_bytes=moe.hidden_dim * moe.in_dtype.itemsize,
        # hidden_dim_scale_bytes=(
        #     0
        #     if moe.in_dtype.itemsize != 1
        #     else (
        #         (moe.hidden_dim + moe.block_size - 1)
        #         // moe.block_size
        #         * torch.float32.itemsize
        #     )
        # ),
    )

    # Generate the same test data on all ranks
    rng = torch.Generator(device='cuda')
    rng.manual_seed(123)
    all_rank_data = [
        RankTestData(moe, rng, use_max_tokens=False) for _ in range(num_rank)
    ]
    rank_data = all_rank_data[rank]

    # Collect info by expert
    expert_token_from: list[list[tuple[int, int]]] = [
        [] for _ in range(moe.num_experts)
    ]
    for i_rank, rd in enumerate(all_rank_data):
        for token_idx in range(rd.num_tokens):
            for expert_idx in rd.indices[token_idx]:
                expert_token_from[expert_idx].append((i_rank, token_idx))

    """
    for k in range(world_size):
        if k == rank:
            print(f"rank: {rank}")
            for i, rd in enumerate(all_rank_data):
                print(f"  DP Rank {i}:")
                for token_idx in range(rd.num_tokens):
                    indices = rd.indices[token_idx].tolist()
                    weights = rd.weights[token_idx].tolist()
                    print(f"    x[{token_idx}] -> {list(zip(indices, weights))}")
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
                chars = [f"r{r}t{t}" for r, t in expert_token_from[expert_idx]]
                print(
                    f"  Expert {expert_idx}: {len(expert_token_from[expert_idx])} tokens, from: {chars}"
                )
    """

    num_local_experts = moe.num_experts // world_size
    expert_num_tokens = torch.empty(
        num_local_experts,
        dtype=torch.int32,
        device=device,
    )
    expert_x = torch.empty(
        (num_local_experts, moe.max_num_tokens * num_dp, moe.hidden_dim),
        dtype=moe.in_dtype,
        device=device,
    )

    ata.dispatch(
        out_expert_num_tokens=expert_num_tokens,
        out_expert_x=expert_x,
        tokens=rank_data.x.to(device),
        indices=rank_data.indices.to(device).to(torch.uint32),
    )


if __name__ == '__main__':
    world_size = 8
    torch.multiprocessing.spawn(work, args=(world_size, ), nprocs=world_size)
