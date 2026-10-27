#!/usr/bin/env python3
"""Independent standard-library replay of the prospective JMLR suite."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any, Sequence


EXPECTED_PROTOCOL_SHA256 = "0b043d6a41dd8002bb70abcee94b66bf30ca9a2addd0ff886c8b17d5d69e03c8"


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def stable_error(
    claims: Sequence[F], responses: Sequence[Sequence[F]], lipschitz: F
) -> tuple[F, tuple[int, int]]:
    best, witness = F(0), (0, 0)
    for left in range(len(claims)):
        for right in range(left + 1, len(claims)):
            response_gap = max(
                abs(a - b)
                for a, b in zip(responses[left], responses[right], strict=True)
            )
            defect = abs(claims[left] - claims[right]) - lipschitz * response_gap
            if defect > best:
                best, witness = defect, (left, right)
    return best / 2, witness


def replay(main_path: Path, sidecar_path: Path, root: Path) -> dict[str, Any]:
    artifact = json.loads(main_path.read_text(encoding="utf-8"))
    require(artifact["schema"] == "tcabb.jmlr_prospective_suite.v1", "unknown main schema")
    require(artifact["status"] == "PASS", "main status is not PASS")
    protocol = artifact["protocol"]
    require(protocol["status"] == "PROSPECTIVE_FOR_NEW_CHECKPOINTS", "protocol status differs")
    require(protocol["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "protocol digest differs")
    require(
        sha256(root / protocol["protocol_file"]) == EXPECTED_PROTOCOL_SHA256,
        "frozen protocol file differs",
    )
    require(protocol["settings"] == 150, "registered setting count differs")
    require(protocol["net_points_per_anchor"] == 99, "registered net size differs")
    require(protocol["candidate_interfaces"] == 30, "candidate count differs")
    require(sha256(sidecar_path) == artifact["exact_world_sidecar"]["sha256"],
            "world-table sidecar hash differs")
    with gzip.open(sidecar_path, "rt", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    require(sidecar["schema"] == "tcabb.jmlr_prospective_world_tables.v1",
            "unknown sidecar schema")
    require(sidecar["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256,
            "sidecar protocol digest differs")
    tables = {table["setting_id"]: table["worlds"] for table in sidecar["tables"]}
    require(len(tables) == len(sidecar["tables"]), "duplicate sidecar setting id")

    outcome_counts = {name: 0 for name in protocol["outcomes"]}
    setting_ids = set()
    exact_tables = 0
    for setting in artifact["settings"]:
        setting_id = setting["setting_id"]
        require(setting_id not in setting_ids, "duplicate main setting id")
        setting_ids.add(setting_id)
        outcome = setting["outcome"]
        require(outcome in outcome_counts, "unregistered outcome")
        outcome_counts[outcome] += 1
        hashes = setting["frozen_hashes"]
        checkpoint = root / "results" / "prospective" / (
            f"checkpoint_{setting['architecture']}_seed_{setting['seed']}.pt"
        )
        inputs = root / "results" / "prospective" / (
            f"inputs_{setting['architecture']}_seed_{setting['seed']}.npz"
        )
        require(sha256(checkpoint) == hashes["checkpoint_sha256"], "checkpoint hash differs")
        require(sha256(inputs) == hashes["inputs_sha256"], "input hash differs")
        if outcome == "TRAINING_OR_ANCHOR_FAILURE":
            require(setting_id not in tables, "failure setting unexpectedly has a world table")
            continue

        require(setting_id in tables, "exact setting lacks a world table")
        worlds = tables[setting_id]
        exact_tables += 1
        require(len(worlds) == setting["worlds"] == 396, "world count differs")
        require(canonical_digest(worlds) == setting["exact_world_table_sha256"],
                "world table digest differs")
        claims = tuple(F(world["claim"]) for world in worlds)
        hard = tuple(
            tuple(F(value) for value in world["hard_responses"]) for world in worlds
        )
        fractional = tuple(F(world["selected_fractional_response"]) for world in worlds)
        selected = setting["selection"]["selected"]
        replacement = int(selected["replacement"])
        hybrid = []
        for hard_row, value in zip(hard, fractional, strict=True):
            row = list(hard_row)
            row[replacement] = value
            hybrid.append(tuple(row))
        construction = tuple(
            index for index, world in enumerate(worlds) if world["role"] == "construction"
        )
        heldout = tuple(
            index for index, world in enumerate(worlds) if world["role"] == "heldout"
        )
        require(len(construction) == len(heldout) == 198, "anchor role split differs")
        subset = lambda values, indices: tuple(values[index] for index in indices)
        hard_construction, _ = stable_error(
            subset(claims, construction), subset(hard, construction), F(14)
        )
        hard_all, hard_witness = stable_error(claims, hard, F(14))
        hybrid_construction, _ = stable_error(
            subset(claims, construction), subset(hybrid, construction), F(14)
        )
        hybrid_heldout, _ = stable_error(
            subset(claims, heldout), subset(hybrid, heldout), F(14)
        )
        hybrid_all, _ = stable_error(claims, hybrid, F(14))
        require(hard_construction == F(setting["hard_interface"]["construction_error"]["fraction"]),
                "hard construction error differs")
        require(hard_all == F(setting["hard_interface"]["all_anchor_error"]["fraction"]),
                "hard all-anchor error differs")
        require(hybrid_construction == F(selected["construction_error"]),
                "hybrid construction error differs")
        require(hybrid_heldout == F(setting["confirmation"]["heldout_only_error"]["fraction"]),
                "hybrid held-out error differs")
        require(hybrid_all == F(setting["confirmation"]["all_anchor_error"]["fraction"]),
                "hybrid all-anchor error differs")

        candidates = setting["selection"]["all_candidate_construction_errors"]
        require(len(candidates) == setting["selection"]["candidate_count"] == 30,
                "candidate table size differs")
        expected_order = sorted(candidates, key=lambda row: (
            F(row["construction_error"]),
            F(row["construction_required_lipschitz"])
            if row["construction_required_lipschitz"] != "INF" else F(10**100),
            row["id"],
        ))
        require(candidates == expected_order, "candidate table is not in selection order")
        require(candidates[0] == selected, "selected candidate is not the registered optimum")
        require(sum(F(row["construction_error"]) == 0 for row in candidates)
                == setting["selection"]["zero_error_candidates"],
                "zero-error candidate count differs")
        confirmed = hybrid_construction == 0 and hard_all > 0 and hybrid_all == 0
        expected_outcome = (
            "NO_HARD_OBSTRUCTION_ON_DECLARED_NET" if hard_all == 0
            else "NO_ZERO_ERROR_FRACTIONAL_CANDIDATE" if hybrid_construction > 0
            else "CONFIRMED_STRICT_WIN_ON_DECLARED_NET" if confirmed
            else "SELECTED_INTERFACE_DID_NOT_CONFIRM"
        )
        require(outcome == expected_outcome, "outcome classification differs")

        left, right = hard_witness
        weight = F(selected["first_weight"])
        first, second = int(selected["first"]), int(selected["second"])
        jensen = (
            fractional[left] - fractional[right]
            - weight * (hard[left][first] - hard[right][first])
            - (1 - weight) * (hard[left][second] - hard[right][second])
        )
        require(jensen == F(setting["controls"]["selected_signed_jensen_defect_on_hard_witness"]["fraction"]),
                "Jensen control differs")
        affine = tuple(weight * row[first] + (1 - weight) * row[second] for row in hard)
        affine_hybrid = []
        for hard_row, value in zip(hard, affine, strict=True):
            row = list(hard_row)
            row[replacement] = value
            affine_hybrid.append(tuple(row))
        affine_error, _ = stable_error(claims, affine_hybrid, F(14))
        require(affine_error == F(setting["controls"]["affine_interpolation_hybrid_error"]["fraction"]),
                "affine control error differs")
        require(affine_error >= hard_all, "affine interpolation beat the hard library")

    require(len(setting_ids) == protocol["settings"], "main setting count differs")
    require(exact_tables == len(tables) == artifact["exact_world_sidecar"]["tables"],
            "exact table count differs")
    require(outcome_counts == artifact["aggregate"]["outcomes"], "aggregate outcomes differ")
    analysis_path = root / "results" / "jmlr_prospective_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    require(analysis["schema"] == "tcabb.jmlr_prospective_analysis.v1",
            "unknown prospective analysis schema")
    require(analysis["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256,
            "analysis protocol digest differs")
    settings = artifact["settings"]
    hard_obstructed = [row for row in settings if row["outcome"] not in {
        "NO_HARD_OBSTRUCTION_ON_DECLARED_NET", "TRAINING_OR_ANCHOR_FAILURE"
    }]
    eligible = [row for row in settings if row["outcome"] in {
        "CONFIRMED_STRICT_WIN_ON_DECLARED_NET", "SELECTED_INTERFACE_DID_NOT_CONFIRM"
    }]
    wins = [row for row in settings if row["outcome"] == "CONFIRMED_STRICT_WIN_ON_DECLARED_NET"]
    funnel = analysis["failure_funnel"]
    require((funnel["registered"], funnel["hard_obstructed"],
             funnel["construction_eligible"], funnel["confirmed"])
            == (150, len(hard_obstructed), len(eligible), len(wins)),
            "failure funnel differs")
    dominance = Counter()
    for row in settings:
        hard = F(row["hard_interface"]["all_anchor_error"]["fraction"])
        hybrid = F(row["confirmation"]["all_anchor_error"]["fraction"])
        dominance["hybrid_better" if hybrid < hard else
                  "equal" if hybrid == hard else "hybrid_worse"] += 1
    require(dict(dominance) == analysis["all_anchor_error_dominance"],
            "error-dominance analysis differs")
    for lipschitz, record in analysis["fixed_selected_interface_sensitivity"].items():
        strict = 0
        for row in settings:
            sensitivity = next(
                item for item in row["controls"]["lipschitz_sensitivity"]
                if item["lipschitz"] == lipschitz
            )
            strict += F(sensitivity["hard_error"]) > 0 and F(
                sensitivity["selected_hybrid_error"]
            ) == 0
        require(strict == record["fixed_L14_selected_interface_strict_wins"],
                "L-sensitivity analysis differs")
    return {
        "status": "PASS",
        "settings": len(setting_ids),
        "exact_tables": exact_tables,
        "outcomes": outcome_counts,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "analysis_replayed": True,
        "production_module_imported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", type=Path,
        default=Path("results/jmlr_prospective_suite.json"),
    )
    parser.add_argument(
        "--sidecar", type=Path,
        default=Path("results/jmlr_prospective_world_tables.json.gz"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        result = replay(args.artifact, args.sidecar, args.root)
    except (OSError, KeyError, ValueError, ReplayError) as error:
        print(f"JMLR PROSPECTIVE REPLAY — FAIL: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
