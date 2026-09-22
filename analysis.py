#!/usr/bin/env python3
"""
analysis.py

Compute empirical conditional entropy for product partitions of
one-round histories in a repeated two-action game.

Expected JSON record format:
{
    "history": [["L", "R"], ["R", "R"], ["L", "L"], ["R", "L"]],
    "action_next": "L",
    "mapping_id": 0
}

The script:
    1. Reads a history-action dataset.
    2. Generates all 15 partitions of the four one-round outcomes:
           LL, LR, RL, RR.
    3. Computes CE_{P^k} for k = 0,...,R for each partition P.
    4. Computes CE_{P^k} - CE_{(P0)^k}, where P0 is the full partition.
    5. Orders non-full partitions by their mean CE difference.

Usage:
    python analysis.py data/path/to/file.json

Optional:
    python analysis.py data/path/to/file.json --horizon 4
    python analysis.py data/path/to/file.json --output results.json
    python analysis.py data/path/to/file.json --latex
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


# ============================================================
# Basic helpers
# ============================================================

JOINT_OUTCOMES = (
    ("L", "L"),
    ("L", "R"),
    ("R", "L"),
    ("R", "R"),
)


def history_key(history):
    """Convert a history to an immutable tuple representation."""
    return tuple(tuple(round_outcome) for round_outcome in history)


def outcome_to_string(outcome):
    """Convert ('L', 'R') to 'LR'."""
    return "".join(outcome)


def partition_to_string(partition):
    """
    Convert a partition to a readable string.

    Example:
        ((('L','L'),), (('L','R'), ('R','L'), ('R','R')))
    becomes:
        {{LL},{LR,RL,RR}}
    """
    blocks = []

    for block in partition:
        block_string = ",".join(
            outcome_to_string(outcome)
            for outcome in block
        )
        blocks.append("{" + block_string + "}")

    return "{" + ",".join(blocks) + "}"


def partition_to_latex(partition):
    """
    Convert a partition to LaTeX notation.

    Example:
        {{LL},{LR,RL,RR}}
    becomes:
        \left\{\{\L\L\},\{\L\R,\R\L,\R\R\}\right\}
    """
    blocks = []

    for block in partition:
        histories = ",\\,".join(
            "".join(f"\\{action}" for action in outcome)
            for outcome in block
        )
        blocks.append(r"\{" + histories + r"\}")

    return r"\left\{" + ",\\;".join(blocks) + r"\right\}"


# ============================================================
# Generate all set partitions
# ============================================================

def generate_set_partitions(items):
    """
    Generate all set partitions of `items` using restricted-growth
    strings. For four elements, this returns 15 partitions.

    The output is deterministic.
    """
    n = len(items)

    def recurse(position, restricted_growth):
        if position == n:
            number_of_blocks = max(restricted_growth) + 1
            blocks = [[] for _ in range(number_of_blocks)]

            for item_index, block_index in enumerate(restricted_growth):
                blocks[block_index].append(items[item_index])

            yield tuple(
                tuple(block)
                for block in blocks
            )
            return

        maximum_block = max(restricted_growth)

        for block_index in range(maximum_block + 2):
            yield from recurse(
                position + 1,
                restricted_growth + [block_index]
            )

    # The first item must be placed in the first block.
    yield from recurse(1, [0])


def ordered_partitions():
    """
    Return all 15 partitions in a convenient order:

        - coarsest partition first;
        - partitions with fewer blocks before those with more blocks;
        - full partition last.

    Note: partitions are not totally ordered by refinement. This is
    only a stable display order.
    """
    partitions = list(generate_set_partitions(JOINT_OUTCOMES))

    partitions.sort(
        key=lambda partition: (
            len(partition),
            partition_to_string(partition),
        )
    )

    return partitions


# ============================================================
# Dataset loading
# ============================================================

def load_records(data_file, horizon=None):
    """
    Read a JSON history-action dataset.

    Required fields in each record:
        - history
        - action_next
    """
    with open(data_file, "r") as f:
        records = json.load(f)

    # Allow a top-level wrapper if needed.
    if isinstance(records, dict):
        if "data" in records:
            records = records["data"]
        elif "records" in records:
            records = records["records"]
        else:
            raise ValueError(
                "JSON file is a dictionary, but does not contain "
                "'data' or 'records'."
            )

    if not isinstance(records, list):
        raise ValueError("Expected a list of records.")

    cleaned_records = []

    for record_index, record in enumerate(records):
        if "history" not in record:
            raise KeyError(
                f"Record {record_index} does not contain 'history'."
            )

        if "action_next" not in record:
            raise KeyError(
                f"Record {record_index} does not contain 'action_next'."
            )

        history = record["history"]
        action = str(record["action_next"]).strip().upper()

        if action not in {"L", "R"}:
            raise ValueError(
                f"Record {record_index} has invalid action_next={action!r}. "
                "Expected 'L' or 'R'."
            )

        normalized_history = []

        for round_index, outcome in enumerate(history):
            if len(outcome) != 2:
                raise ValueError(
                    f"Record {record_index}, round {round_index}: "
                    "each history entry must contain two actions."
                )

            focal_action = str(outcome[0]).strip().upper()
            other_action = str(outcome[1]).strip().upper()

            if focal_action not in {"L", "R"} or other_action not in {"L", "R"}:
                raise ValueError(
                    f"Record {record_index}, round {round_index}: "
                    f"invalid outcome {outcome!r}. Expected L/R actions."
                )

            normalized_history.append(
                (focal_action, other_action)
            )

        if horizon is not None and len(normalized_history) != horizon:
            raise ValueError(
                f"Record {record_index} has history length "
                f"{len(normalized_history)}, but horizon={horizon}."
            )

        cleaned_records.append({
            "history": tuple(normalized_history),
            "action_next": action,
        })

    if not cleaned_records:
        raise ValueError("Dataset contains no records.")

    inferred_horizon = len(cleaned_records[0]["history"])

    return cleaned_records, inferred_horizon


# ============================================================
# Partition-product conditional entropy
# ============================================================

def make_outcome_to_block_map(partition):
    """
    Return a dictionary mapping each one-round outcome to its block ID.
    """
    outcome_to_block = {}

    for block_id, block in enumerate(partition):
        for outcome in block:
            outcome_to_block[outcome] = block_id

    return outcome_to_block


def product_partition_key(history, outcome_to_block, k):
    """
    Return the product-partition block ID for the last k rounds.

    For k=0, every history belongs to the same unique block.
    """
    if k == 0:
        return ()

    recent_history = history[-k:]

    return tuple(
        outcome_to_block[outcome]
        for outcome in recent_history
    )


def conditional_entropy(records, partition, k):
    """
    Compute empirical conditional entropy CE_{P^k}.

    For every product-partition block b, estimate the probability of
    L and R using empirical frequencies within b. Then calculate:

        CE = -(1/N) sum_i log Pr(a_i | b_i).

    Natural logarithms are used, so entropy is measured in nats.
    """
    outcome_to_block = make_outcome_to_block_map(partition)

    block_action_counts = defaultdict(
        lambda: {"L": 0, "R": 0}
    )

    assignments = []

    for record in records:
        block = product_partition_key(
            record["history"],
            outcome_to_block,
            k
        )

        action = record["action_next"]

        block_action_counts[block][action] += 1
        assignments.append((block, action))

    total_negative_log_likelihood = 0.0

    for block, action in assignments:
        action_count = block_action_counts[block][action]
        block_total = (
            block_action_counts[block]["L"]
            + block_action_counts[block]["R"]
        )

        probability = action_count / block_total

        total_negative_log_likelihood -= math.log(probability)

    return total_negative_log_likelihood / len(records)


def compute_partition_statistics(records, partitions, horizon):
    """
    Compute CE values for every partition and every k=0,...,horizon.
    """
    all_ce_values = {}

    for partition_index, partition in enumerate(partitions):
        ce_values = []

        for k in range(horizon + 1):
            ce = conditional_entropy(
                records,
                partition,
                k
            )
            ce_values.append(ce)

        all_ce_values[partition_index] = ce_values

    return all_ce_values


# ============================================================
# Output helpers
# ============================================================

def format_number(value):
    """Format numbers consistently and suppress negative zero."""
    if abs(value) < 5e-12:
        value = 0.0

    return f"{value:.6f}"


def print_text_results(partitions, ce_by_partition, full_index, horizon):
    """
    Print full-partition CE values and the sorted non-full table.
    """
    full_ce = ce_by_partition[full_index]

    print("\n" + "=" * 100)
    print("FULL PRODUCT PARTITION")
    print("=" * 100)

    print(
        f"Index: 0\n"
        f"Partition: {partition_to_string(partitions[full_index])}"
    )

    for k, value in enumerate(full_ce):
        print(f"CE_full(k={k}) = {format_number(value)}")

    results = []

    for partition_index, partition in enumerate(partitions):
        if partition_index == full_index:
            continue

        ce_values = ce_by_partition[partition_index]

        differences = [
            ce_values[k] - full_ce[k]
            for k in range(horizon + 1)
        ]

        mean_difference = sum(differences) / len(differences)

        results.append({
            "partition_index": partition_index,
            "partition": partition,
            "ce_values": ce_values,
            "differences": differences,
            "mean_difference": mean_difference,
        })

    results.sort(
        key=lambda result: result["mean_difference"]
    )

    print("\n" + "=" * 100)
    print("NON-FULL PARTITIONS ORDERED BY MEAN CE DIFFERENCE")
    print("=" * 100)

    header = (
        f"{'Rank':<6}"
        f"{'Original index':<16}"
        f"{'Partition':<42}"
        f"{'d(k=0)':>11}"
        f"{'d(k=1)':>11}"
        f"{'d(k=2)':>11}"
        f"{'d(k=3)':>11}"
        f"{'d(k=4)':>11}"
        f"{'Mean diff.':>13}"
    )

    print(header)
    print("-" * len(header))

    for rank, result in enumerate(results, start=1):
        diffs = result["differences"]

        print(
            f"{rank:<6}"
            f"{result['partition_index']:<16}"
            f"{partition_to_string(result['partition']):<42}"
            f"{format_number(diffs[0]):>11}"
            f"{format_number(diffs[1]):>11}"
            f"{format_number(diffs[2]):>11}"
            f"{format_number(diffs[3]):>11}"
            f"{format_number(diffs[4]):>11}"
            f"{format_number(result['mean_difference']):>13}"
        )

    return results


def print_latex_results(partitions, ce_by_partition, full_index, horizon):
    """
    Print a LaTeX tabular representation to stdout.
    """
    full_ce = ce_by_partition[full_index]

    results = []

    for partition_index, partition in enumerate(partitions):
        if partition_index == full_index:
            continue

        ce_values = ce_by_partition[partition_index]
        differences = [
            ce_values[k] - full_ce[k]
            for k in range(horizon + 1)
        ]

        results.append({
            "partition_index": partition_index,
            "partition": partition,
            "differences": differences,
            "mean_difference": sum(differences) / len(differences),
        })

    results.sort(key=lambda result: result["mean_difference"])

    print("\n% ---------- Full partition ----------")
    print(r"\begin{tabular}{|c|c|ccccc|}")
    print(r"\hline")
    print(
        r"\textbf{Index} & \textbf{Partition} "
        r"& $k=0$ & $k=1$ & $k=2$ & $k=3$ & $k=4$ \\"
    )
    print(r"\hline")

    full_values = " & ".join(
        format_number(value)
        for value in full_ce
    )

    print(
        "0 & "
        + "$" + partition_to_latex(partitions[full_index]) + "$"
        + " & "
        + full_values
        + r" \\"
    )

    print(r"\hline")
    print(r"\end{tabular}")

    print("\n% ---------- Non-full partitions ----------")
    print(r"\begin{tabular}{|c|c|ccccc|c|}")
    print(r"\hline")
    print(
        r"\textbf{Rank} & \textbf{Partition} "
        r"& $k=0$ & $k=1$ & $k=2$ & $k=3$ & $k=4$ "
        r"& \textbf{Mean} \\"
    )
    print(r"\hline")

    for rank, result in enumerate(results, start=1):
        diff_values = " & ".join(
            format_number(value)
            for value in result["differences"]
        )

        print(
            f"{rank} & "
            + "$" + partition_to_latex(result["partition"]) + "$"
            + " & "
            + diff_values
            + " & "
            + format_number(result["mean_difference"])
            + r" \\"
        )

    print(r"\hline")
    print(r"\end{tabular}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute conditional entropy for all memory-one "
            "partitions and their product partitions."
        )
    )

    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to a JSON history-action dataset."
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "History horizon. If omitted, infer it from the first "
            "record in the dataset."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path for a JSON file containing all results."
        )
    )

    parser.add_argument(
        "--latex",
        action="store_true",
        help="Also print LaTeX tabular output."
    )

    args = parser.parse_args()

    if not args.data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data_file}"
        )

    records, inferred_horizon = load_records(
        args.data_file,
        horizon=args.horizon
    )

    horizon = (
        args.horizon
        if args.horizon is not None
        else inferred_horizon
    )

    partitions = ordered_partitions()

    if len(partitions) != 15:
        raise RuntimeError(
            f"Expected 15 partitions, found {len(partitions)}."
        )

    # The full partition has four singleton blocks.
    full_index = next(
        index
        for index, partition in enumerate(partitions)
        if len(partition) == 4
    )

    print("\n" + "=" * 100)
    print("PARTITION-BASED CONDITIONAL-ENTROPY ANALYSIS")
    print("=" * 100)
    print(f"Data file: {args.data_file}")
    print(f"Number of records: {len(records)}")
    print(f"Horizon: {horizon}")
    print(f"Number of memory-one partitions: {len(partitions)}")
    print(f"Full-partition index: {full_index}")

    ce_by_partition = compute_partition_statistics(
        records,
        partitions,
        horizon
    )

    sorted_results = print_text_results(
        partitions,
        ce_by_partition,
        full_index,
        horizon
    )

    if args.latex:
        print_latex_results(
            partitions,
            ce_by_partition,
            full_index,
            horizon
        )

    if args.output is not None:
        output = {
            "data_file": str(args.data_file),
            "number_of_records": len(records),
            "horizon": horizon,
            "full_partition_index": full_index,
            "full_partition": partition_to_string(
                partitions[full_index]
            ),
            "full_partition_ce": ce_by_partition[full_index],
            "ranked_nonfull_partitions": [
                {
                    "rank": rank,
                    "original_partition_index": result["partition_index"],
                    "partition": partition_to_string(result["partition"]),
                    "ce_difference_from_full": result["differences"],
                    "mean_ce_difference_from_full": result["mean_difference"],
                }
                for rank, result in enumerate(sorted_results, start=1)
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()