import torch

class Buffer:
    def __init__(
        self,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        dp_size: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
    ) -> None:
        assert world_size % dp_size == 0
        assert world_size // dp_size > 1

        self.world_size = world_size
        self.dp_size = dp_size
        self.max_num_tokens = max_num_tokens
        self._has_scales = hidden_dim_scale_bytes > 0

"""
        self._ptr = _ops.all_to_all_create(
            max_num_tokens,
            num_experts,
            experts_per_token,
            rank,
            world_size,
            dp_size,
            hidden_dim,
            hidden_dim_bytes,
            hidden_dim_scale_bytes,
        )
        assert self._ptr != 0
"""
