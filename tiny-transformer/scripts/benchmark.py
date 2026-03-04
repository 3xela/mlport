import argparse
import torch
from transformer_configs import TransformerConfig
from model.transformer import TinyTransformer
import time, json, os

from scripts.utils.model_perf import (
    measure_baseline_decode,
    measure_kv_cache
)
from scripts.utils.model_factory import config_factory, model_factory


def save_benchmark(name, config, ms_per_forward, tokens_per_sec):
    ts = int(time.time())
    out = {
        "timestamp": ts,
        "name": name,
        "config": config.__dict__,
        "ms_per_forward": ms_per_forward,
        "tokens_per_sec": tokens_per_sec,
    }

    os.makedirs("benchmarks/results", exist_ok=True)
    fname = f"benchmarks/results/{name}_{ts}.json"

    with open(fname, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[saved benchmark → {fname}]")


def parse_args():
    parser = argparse.ArgumentParser(description="TinyTransformer benchmarking")

    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--name", type=str, default="baseline")

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--kv-cache", action="store_true",
                        help="Run KV-cache optimized decode benchmark")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = config_factory(args.config) if args.config else config_factory()
    model = TinyTransformer(config).to(device).eval()

    B = args.batch
    T = args.seq_len
    tokens = torch.randint(0, config.vocab_size, (B, T), device=device)

    if args.kv_cache:
        ms_per_forward, tokens_per_sec = measure_kv_cache(
            args.iters, model, tokens, config
        )
    else:
        ms_per_forward, tokens_per_sec = measure_baseline_decode(
            args.iters, model, tokens, config
        )

    save_benchmark(
        name=args.name,
        config=config,
        ms_per_forward=ms_per_forward,
        tokens_per_sec=tokens_per_sec,
    )
