import torch

class Buffer:
    def __init__(
        self,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
    ) -> None:

        self.world_size = world_size
        self.max_num_tokens = max_num_tokens
        self._has_scales = hidden_dim_scale_bytes > 0
