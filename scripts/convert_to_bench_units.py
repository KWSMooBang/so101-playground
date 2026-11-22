#!/usr/bin/env python3
"""Convert a dataset recorded on our follower (scripts/record.sh) to the SO-101 Bench's joint units.

    envs/lerobot/.venv/bin/python scripts/convert_to_bench_units.py outputs/datasets/pilot
    # -> outputs/datasets/pilot_bench

Copies the dataset and converts `observation.state` and `action` with `so101_playground.bench_units` (our
`.pos` -> sim degrees through our sim alignment -> bench `.pos`). `action` is the leader's `.pos`, which the
follower turned into goal ticks with its own calibration, so it converts the same way as the state.
Their per-episode and global stats are recomputed, `robot_type` becomes the bench's `so101_follower`, and
meta/bench_units.json records the alignment, the calibration and the share of values that saturated at the
bench's range. Videos and everything else are copied unchanged.

The follower's calibration must be the one the dataset was recorded with (bench_units checks it against the
alignment's snapshot, not against the dataset, which does not store it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_feature_stats, get_feature_stats
from lerobot.datasets.io_utils import load_info, load_stats, write_info, write_stats
from so101_playground.bench_units import DEFAULT_ALIGNMENT, JOINTS, BenchUnits, lerobot_calibration

FEATURES = ("observation.state", "action")
BENCH_ROBOT_TYPE = "so101_follower"
MARKER = "meta/bench_units.json"


def bench_range_deg() -> np.ndarray:
    """(6, 2) sim degrees that bench `.pos` can represent: its +-100 (gripper 0..100) ends, within USD limits."""
    lo = [-100.0] * 5 + [0.0]
    ends = [[lerobot_calibration.lerobot_position_to_sim_degrees(j, v) for j, v in zip(JOINTS, end)]
            for end in (lo, [100.0] * 6)]
    return np.array(ends).T


def convert_data(root: Path, units: BenchUnits) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict]:
    """Rewrite the data parquet files in place; returns the converted values per episode and saturation counts."""
    per_episode = {key: {} for key in FEATURES}
    limits = bench_range_deg()
    saturated = {key: np.zeros(len(JOINTS), dtype=np.int64) for key in FEATURES}
    total = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_numpy()
        total += len(episodes)
        for key in FEATURES:
            field = table.schema.field(key)
            values = table.column(key).combine_chunks().flatten().to_numpy().reshape(-1, len(JOINTS))
            sim = units.to_sim_degrees(values)
            saturated[key] += ((sim <= limits[:, 0]) | (sim >= limits[:, 1])).sum(axis=0)
            converted = units.to_bench(values).astype(np.float32)
            column = pa.FixedSizeListArray.from_arrays(pa.array(converted.ravel(), pa.float32()), len(JOINTS))
            table = table.set_column(table.schema.get_field_index(key), field, column.cast(field.type))
            for episode in np.unique(episodes):
                per_episode[key].setdefault(int(episode), []).append(converted[episodes == episode])
        pq.write_table(table, path)
    shares = {key: dict(zip(JOINTS, (saturated[key] / max(total, 1)).round(4).tolist())) for key in FEATURES}
    return {key: {e: np.concatenate(v) for e, v in eps.items()} for key, eps in per_episode.items()}, shares


def rewrite_episode_stats(root: Path, values: dict[str, dict[int, np.ndarray]]) -> dict[str, list[dict]]:
    """Replace stats/<feature>/* in meta/episodes; returns the new per-episode stats in episode order."""
    collected = {key: [] for key in FEATURES}
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        for key in FEATURES:
            stats = [get_feature_stats(values[key][e], axis=0, keepdims=False) for e in episodes]
            collected[key].extend(stats)
            for name in stats[0]:
                column = f"stats/{key}/{name}"
                field = table.schema.field(column)
                data = pa.array([s[name].tolist() for s in stats], type=field.type)
                table = table.set_column(table.schema.get_field_index(column), field, data)
        pq.write_table(table, path)
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Dataset root recorded on our follower.")
    parser.add_argument("dst", type=Path, nargs="?", help="Output root (default: SRC_bench next to SRC).")
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--calibration", type=Path, help="Follower calibration (default: LeRobot's file for the alignment's robot id).")
    parser.add_argument("--overwrite", action="store_true", help="Replace DST if it exists.")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = (args.dst or src.with_name(src.name + "_bench")).resolve()
    if (src / MARKER).exists():
        raise SystemExit(f"{src} is already in bench units ({MARKER}).")
    info = load_info(src)
    expected = [f"{joint}.pos" for joint in JOINTS]
    for key in FEATURES:
        if info.features[key]["names"] != expected:
            raise SystemExit(f"{key} names {info.features[key]['names']} != {expected}")
    units = BenchUnits(args.alignment, args.calibration)
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists (use --overwrite).")
        shutil.rmtree(dst)

    work = dst.with_name(dst.name + ".partial")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(src, work)
    values, saturated = convert_data(work, units)
    episode_stats = rewrite_episode_stats(work, values)

    stats = load_stats(work)
    for key in FEATURES:
        stats[key] = aggregate_feature_stats(episode_stats[key])
    write_stats(stats, work)
    info = load_info(work)
    source_robot_type, info.robot_type = info.robot_type, BENCH_ROBOT_TYPE
    write_info(info, work)
    record = {
        "source": str(src),
        "source_robot_type": source_robot_type,
        "converted": list(FEATURES),
        "saturated_share": saturated,
        **units.describe(),
    }
    (work / MARKER).write_text(json.dumps(record, indent=2) + "\n")
    work.rename(dst)

    print(f"wrote {dst}: {info.total_episodes} episodes, {info.total_frames} frames")
    for key in FEATURES:
        hits = {joint: share for joint, share in saturated[key].items() if share}
        print(f"  {key}: saturated at the bench range {hits or 'nowhere'}")


if __name__ == "__main__":
    main()
