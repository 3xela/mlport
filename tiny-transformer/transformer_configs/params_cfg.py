from dataclasses import dataclass, field

@dataclass
class TransformerConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    vocab_size: int = 1000
    max_seq_len: int = 128
    use_bias: bool = True
    dropout: float = 0.0
    causal: bool = True
    backend: str = "baseline"
    use_KV_cache : bool = False

    d_head: int = field(init=False)

    def __post_init__(self):
        # compute derived params
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads
