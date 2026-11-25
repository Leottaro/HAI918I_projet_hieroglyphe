from __future__ import annotations

import argparse
import itertools
import traceback
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final

from config import CONFIG, OUTPUT_DIR
from train import main as train_all_models

SEARCH_SPACE: Final[Mapping[str, tuple[Any, ...]]] = MappingProxyType({
    "batch_size": (2, 4, 8, 16, 64),
    "learning_rate": (1e-5, 3e-5, 1.e-4, 3e-4, 1.e-3, 3e-3),
    "max_augmentation_percent": (0.25, 0.5, 0.75, 1.)
})
RATIO_KEYS: Final[tuple[str, ...]] = ("train_ratio", "val_ratio", "test_ratio")
DONE_MARKER: Final[str] = ".sweep_done"
BASELINE: Final[dict[str, Any]] = dict(CONFIG)


def apply_config(values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        CONFIG[key] = value
        object.__setattr__(CONFIG, key, value)


def validate_search_space() -> None:
    unknown = sorted(set(SEARCH_SPACE) - set(BASELINE))
    if unknown:
        raise ValueError(f"Unknown keys in SEARCH_SPACE: {unknown}. Valid keys: {sorted(BASELINE)}")
    empty = [key for key, values in SEARCH_SPACE.items() if not values]
    if empty:
        raise ValueError(f"SEARCH_SPACE has no values for: {empty}")


def iter_overrides() -> Iterator[dict[str, Any]]:
    keys = list(SEARCH_SPACE)
    for values in itertools.product(*SEARCH_SPACE.values()):
        overrides = dict(zip(keys, values))
        merged = {**BASELINE, **overrides}
        if abs(sum(merged[key] for key in RATIO_KEYS) - 1.0) > 1e-9:
            print(f"Skipping {overrides}: train/val/test ratios don't sum to 1")
            continue
        yield overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list the combinations without training")
    parser.add_argument("--rerun", action="store_true", help="retrain combinations already marked as done")
    args = parser.parse_args()

    validate_search_space()
    runs = list(iter_overrides())
    print(f"{len(runs)} configuration(s) in the sweep")

    failures: list[dict[str, Any]] = []
    try:
        for index, overrides in enumerate(runs, start=1):
            apply_config(BASELINE)  # always start from the defaults
            apply_config(overrides)

            run_dir = OUTPUT_DIR / CONFIG.getDirName()
            marker = run_dir / DONE_MARKER
            label = f"[{index}/{len(runs)}] {overrides} -> {run_dir.name[:12]}"

            if args.dry_run:
                print(label)
                continue
            if marker.exists() and not args.rerun:
                print(f"{label}  (already done, skipping)")
                continue

            print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")
            try:
                train_all_models(False)
            except Exception:
                traceback.print_exc()
                failures.append(overrides)
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            marker.touch()
    finally:
        apply_config(BASELINE)

    if failures:
        print(f"\n{len(failures)} configuration(s) failed:")
        for overrides in failures:
            print(f"  {overrides}")


if __name__ == "__main__":
    main()