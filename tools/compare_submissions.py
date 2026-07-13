"""
Compare two tracking submission CSV files frame-by-frame.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_csv")
    parser.add_argument("new_csv")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old = pd.read_csv(args.old_csv)
    new = pd.read_csv(args.new_csv)

    print("old shape:", old.shape)
    print("new shape:", new.shape)
    print("old columns:", list(old.columns))
    print("new columns:", list(new.columns))

    required = ["id", "x", "y", "w", "h"]
    if list(old.columns) != required or list(new.columns) != required:
        raise ValueError(f"Both CSVs must have columns {required}")

    old = old.sort_values("id").reset_index(drop=True)
    new = new.sort_values("id").reset_index(drop=True)

    old_ids = set(old["id"])
    new_ids = set(new["id"])
    missing = old_ids - new_ids
    extra = new_ids - old_ids
    print("same row count:", len(old) == len(new))
    print("same id set:", not missing and not extra)

    if missing or extra:
        print("missing in new:", len(missing), sorted(missing)[:20])
        print("extra in new:", len(extra), sorted(extra)[:20])
        return 1

    old = old.set_index("id").loc[sorted(old_ids)]
    new = new.set_index("id").loc[sorted(old_ids)]
    diff = np.abs(old[["x", "y", "w", "h"]].to_numpy() - new[["x", "y", "w", "h"]].to_numpy())
    per_row = diff.max(axis=1)

    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("median abs diff:", float(np.median(diff)))
    print(f"frames changed > {args.tolerance}:", int((per_row > args.tolerance).sum()))
    print("frames changed > 0.1:", int((per_row > 0.1).sum()))
    print("frames changed > 1.0:", int((per_row > 1.0).sum()))

    changed = np.where(per_row > args.tolerance)[0][:20]
    if len(changed):
        ids = old.index.to_numpy()
        print("changed samples:")
        for idx in changed:
            row_id = ids[idx]
            print(row_id)
            print("  old:", old.loc[row_id, ["x", "y", "w", "h"]].to_dict())
            print("  new:", new.loc[row_id, ["x", "y", "w", "h"]].to_dict())
            print("  diff:", diff[idx].tolist())

    return 0 if (per_row <= args.tolerance).all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
