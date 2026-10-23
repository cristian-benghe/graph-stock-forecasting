"""Merge result shards produced by parallel run_experiments.py invocations.

Each config (naive/var/lstm, mtgnn, ablation, cmaes) is an independent set of
model fits on the same seeded split, so running them in separate processes gives
the same numbers as one sequential run while using more cores. This recombines
the shards into the single results.json and predictions.npz the analysis scripts
expect, and fails loudly if two shards disagree about anything they share.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np

SHARD_GLOB = "outputs/shard_*.json"
OUT_RESULTS = Path("outputs/results.json")
OUT_PREDS = Path("outputs/predictions.npz")


def main():
    shards = sorted(Path().glob(SHARD_GLOB))
    if not shards:
        raise SystemExit(f"no shards matching {SHARD_GLOB}")

    print(f"Merging {len(shards)} shards:")
    merged_results, merged_extras, base = {}, {}, None
    seeds = None

    for path in shards:
        payload = json.loads(path.read_text())
        names = list(payload["results"])
        print(f"  {path.name:28s} -> {names}")

        if base is None:
            base = {k: payload[k] for k in ("config", "data_shape", "tickers")}
            seeds = payload["seeds"]
        else:
            if payload["data_shape"] != base["data_shape"]:
                raise SystemExit(
                    f"{path.name}: data_shape {payload['data_shape']} != {base['data_shape']}"
                )
            if payload["tickers"] != base["tickers"]:
                raise SystemExit(f"{path.name}: ticker order differs")
            if payload["seeds"] != seeds:
                raise SystemExit(f"{path.name}: seeds {payload['seeds']} != {seeds}")

        for name, res in payload["results"].items():
            if name in merged_results:
                raise SystemExit(f"duplicate model '{name}' in {path.name}")
            merged_results[name] = res
        merged_extras.update(payload.get("extras", {}))

    OUT_RESULTS.write_text(json.dumps({
        **base,
        "seeds": seeds,
        "results": merged_results,
        "extras": merged_extras,
    }, indent=2, default=str))
    print(f"\nWrote {OUT_RESULTS} with {len(merged_results)} models")

    # --- Merge predictions, verifying every shard scored the same targets ---
    pred_files = sorted(Path().glob("outputs/shard_*_predictions.npz"))
    combined, ref_true = {}, None
    for pf in pred_files:
        blob = np.load(pf)
        for key in blob.files:
            if key in combined:
                raise SystemExit(f"duplicate prediction key '{key}' in {pf.name}")
            combined[key] = blob[key]
            if key.endswith("|true"):
                if ref_true is None:
                    ref_true = blob[key]
                elif not np.allclose(blob[key], ref_true):
                    raise SystemExit(
                        f"{pf.name}: '{key}' has different targets than other shards; "
                        "the splits are not aligned"
                    )

    if combined:
        np.savez_compressed(OUT_PREDS, **combined)
        n_models = len(combined) // 2
        print(f"Wrote {OUT_PREDS} with {n_models} prediction sets "
              f"(all scored on identical targets, shape {ref_true.shape})")


if __name__ == "__main__":
    main()
