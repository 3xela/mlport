import json, glob
from tabulate import tabulate

files = sorted(glob.glob("benchmarks/results/*.json"))

rows = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    rows.append([
        d["name"],
        d["config"]["d_model"],
        d["config"]["n_layers"],
        d["config"]["max_seq_len"],
        d["ms_per_forward"],
        int(d["tokens_per_sec"]),
    ])

print(tabulate(
    rows,
    headers=["Name", "d_model", "layers", "T", "ms", "tok/s"],
    tablefmt="github",
))
