import dataclasses
import torch

@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    block_size: int = 128