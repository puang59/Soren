"""Split graphs into train, validation and test sets by fixing commit.

BigVul contains near-duplicate functions: one commit often fixes several copies of the same
bug. A random split would put some copies in training and others in test. Assigning whole
commits to a split removes that leak.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from soren.data.filtering import body_hash
from soren.data.schema import GraphRecord

SPLITS: tuple[str, ...] = ("train", "val", "test")


class LeakageError(AssertionError):
    """Raised when a commit or a function body appears in more than one split."""


@dataclass
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split ratios must be non-negative")
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.train, self.val, self.test)


def _group_key(record: GraphRecord) -> str:
    # A function without a commit id forms its own group.
    return record.commit_id or f"sample:{record.sample_id}"


def source_hash(record: GraphRecord) -> str:
    return body_hash("\n".join(record.source_lines))


def split_by_commit(
    records: Sequence[GraphRecord], config: SplitConfig | None = None
) -> dict[str, list[GraphRecord]]:
    """Assign whole commits to train, val and test, stratified by CWE.

    Within each CWE the commits are shuffled and dealt out, largest first, each going to the
    split furthest below its target share of graphs. A commit that spans several CWEs is
    stratified under its most common one. The result is deterministic for a given seed and
    input order.
    """
    config = config or SplitConfig()
    groups: dict[str, list[GraphRecord]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)

    strata: dict[str, list[str]] = defaultdict(list)
    for key in sorted(groups):
        cwe = Counter(r.cwe for r in groups[key]).most_common(1)[0][0]
        strata[cwe].append(key)

    rng = np.random.default_rng(config.seed)
    assigned: dict[str, list[GraphRecord]] = {name: [] for name in SPLITS}
    for cwe in sorted(strata):
        keys = strata[cwe]
        # Shuffle, then place the largest commits first: a commit with a hundred functions
        # dealt late would overshoot whichever split it lands in.
        order = sorted(rng.permutation(len(keys)), key=lambda i: -len(groups[keys[i]]))
        total = sum(len(groups[key]) for key in keys)
        counts = dict.fromkeys(SPLITS, 0)
        for position in order:
            members = groups[keys[position]]
            # Give the commit to the split furthest below its target share.
            deficits = {
                name: ratio * total - counts[name]
                for name, ratio in zip(SPLITS, config.ratios, strict=True)
                if ratio > 0
            }
            target = max(deficits, key=lambda name: (deficits[name], -SPLITS.index(name)))
            assigned[target].extend(members)
            counts[target] += len(members)

    order_of = {record.sample_id: i for i, record in enumerate(records)}
    for name in SPLITS:
        assigned[name].sort(key=lambda record: order_of[record.sample_id])
    check_no_leakage(assigned)
    return assigned


def check_no_leakage(splits: dict[str, list[GraphRecord]]) -> None:
    """Raise :class:`LeakageError` if a commit or a function body is shared between splits."""
    commit_home: dict[str, str] = {}
    body_home: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            commit = _group_key(record)
            if commit_home.setdefault(commit, name) != name:
                raise LeakageError(
                    f"commit {commit} appears in both {commit_home[commit]} and {name}"
                )
            digest = source_hash(record)
            if body_home.setdefault(digest, name) != name:
                raise LeakageError(
                    f"{record.sample_id}: function body also appears in {body_home[digest]}"
                )


def split_summary(splits: dict[str, list[GraphRecord]]) -> dict[str, dict[str, object]]:
    total = sum(len(records) for records in splits.values())
    return {
        name: {
            "graphs": len(records),
            "share": round(len(records) / max(total, 1), 4),
            "commits": len({_group_key(r) for r in records}),
            "by_cwe": dict(sorted(Counter(r.cwe for r in records).items())),
        }
        for name, records in splits.items()
    }
