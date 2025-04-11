import torch
import ship

class Buffer:
    def __init__(
        self,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        hidden_dim: int,
        # hidden_dim_bytes: int,
        # hidden_dim_scale_bytes: int,
    ) -> None:

        self.rank = rank
        self.world_size = world_size
        self.max_num_tokens = max_num_tokens
        self.hidden_dim = hidden_dim
        # self._has_scales = hidden_dim_scale_bytes > 0
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    def all_to_all_dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        x: torch.Tensor,
        indices: torch.Tensor,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        ship.intranode_dispatch(
            out_expert_num_tokens,
            out_expert_x,
            x,
            indices,
            do_send=do_send,
            do_recv=do_recv,
            self.rank,
            self.world_size,
            self.max_num_tokens,
            self.hidden_dim,
            self.num_experts,
            self.experts_per_token,
        )