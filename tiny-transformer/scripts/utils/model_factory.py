from transformer_configs import TransformerConfig
from model.transformer import TinyTransformer

def config_factory(d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    vocab_size=1000,
    max_seq_len=512,
    use_bias=True,
    dropout=0.0,
    causal=True,
    backend=None):

    return TransformerConfig(
    d_model,
    n_heads,
    n_layers,
    d_ff,
    vocab_size,
    max_seq_len,
    use_bias,
    dropout,
    causal,
    backend)


def model_factory(config, device, eval_mode = True):
    return TinyTransformer(config).to(device).eval_mode() if eval_mode else TinyTransformer(config).to(device)