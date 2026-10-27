#!/usr/bin/env python3
"""Independent standard-library replay of the JMLR defensibility suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
from typing import Any, Sequence


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


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
            gap = max(
                abs(a - b) for a, b in zip(responses[left], responses[right], strict=True)
            )
            defect = abs(claims[left] - claims[right]) - lipschitz * gap
            if defect > best:
                best, witness = defect, (left, right)
    return best / 2, witness


def replay(path: Path, root: Path | None = None) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    require(artifact["schema"] == "tcabb.jmlr_defensibility_suite.v1", "unknown schema")
    require(artifact["status"] == "PASS", "suite status is not PASS")
    require(
        canonical_digest(artifact["protocol"]) == artifact["protocol_sha256"],
        "protocol digest differs",
    )
    protocol = artifact["protocol"]
    require(protocol["settings"] == 25 and protocol["replication_settings"] == 24,
            "protocol setting counts differ")
    require(protocol["candidate_library"]["candidates"] == 30,
            "candidate-library size differs")
    require(protocol["net_points_per_anchor"] == 99, "mechanism-net size differs")
    require(F(protocol["primary_lipschitz"]) == 14, "primary Lipschitz constant differs")

    outcome_counts: dict[str, int] = {}
    replication_counts: dict[str, int] = {}
    for setting in artifact["settings"]:
        worlds = setting["exact_world_table"]
        require(len(worlds) == setting["worlds"] == 396, "world count differs")
        claims = tuple(F(world["claim"]) for world in worlds)
        hard = tuple(tuple(F(value) for value in world["hard_responses"]) for world in worlds)
        fractional = tuple(F(world["selected_fractional_response"]) for world in worlds)
        selected = setting["selection"]["selected"]
        replacement = int(selected["replacement"])
        hybrid = []
        for hard_row, value in zip(hard, fractional, strict=True):
            row = list(hard_row)
            row[replacement] = value
            hybrid.append(tuple(row))
        construction = tuple(index for index, world in enumerate(worlds)
                             if world["role"] == "construction")
        heldout = tuple(index for index, world in enumerate(worlds)
                        if world["role"] == "heldout")
        require(len(construction) == len(heldout) == 198, "anchor split differs")
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
                "selected construction error differs")
        require(hybrid_heldout == F(setting["confirmation"]["heldout_only_error"]["fraction"]),
                "held-out error differs")
        require(hybrid_all == F(setting["confirmation"]["all_anchor_error"]["fraction"]),
                "hybrid all-anchor error differs")
        candidate_rows = setting["selection"]["all_candidate_construction_errors"]
        require(len(candidate_rows) == setting["selection"]["candidate_count"] == 30,
                "candidate result count differs")
        require(candidate_rows[0]["id"] == selected["id"], "selected candidate is not first")
        require(sum(F(row["construction_error"]) == 0 for row in candidate_rows)
                == setting["selection"]["zero_error_candidates"],
                "zero-error candidate count differs")
        confirmed = hybrid_construction == 0 and hard_all > 0 and hybrid_all == 0
        expected_outcome = (
            "NO_HARD_OBSTRUCTION_ON_DECLARED_NET" if hard_all == 0
            else "NO_ZERO_ERROR_FRACTIONAL_CANDIDATE" if hybrid_construction > 0
            else "CONFIRMED_STRICT_WIN_ON_DECLARED_NET" if confirmed
            else "SELECTED_INTERFACE_DID_NOT_CONFIRM"
        )
        require(setting["outcome"] == expected_outcome, "outcome classification differs")

        first, second = hard_witness
        weight = F(selected["first_weight"])
        q_first, q_second = int(selected["first"]), int(selected["second"])
        jensen = (
            fractional[first] - fractional[second]
            - weight * (hard[first][q_first] - hard[second][q_first])
            - (1 - weight) * (hard[first][q_second] - hard[second][q_second])
        )
        require(jensen == F(setting["controls"]["selected_signed_jensen_defect_on_hard_witness"]["fraction"]),
                "Jensen defect differs")
        affine = tuple(weight * row[q_first] + (1 - weight) * row[q_second] for row in hard)
        affine_hybrid = []
        for hard_row, value in zip(hard, affine, strict=True):
            row = list(hard_row)
            row[replacement] = value
            affine_hybrid.append(tuple(row))
        affine_error, _ = stable_error(claims, affine_hybrid, F(14))
        require(affine_error == F(setting["controls"]["affine_interpolation_hybrid_error"]["fraction"]),
                "affine-control error differs")
        require(affine_error >= hard_all, "affine interpolation beat the complete hard library")

        outcome_counts[setting["outcome"]] = outcome_counts.get(setting["outcome"], 0) + 1
        if not setting["is_original_headline_setting"]:
            replication_counts[setting["outcome"]] = replication_counts.get(setting["outcome"], 0) + 1
        if root is not None:
            checkpoint = root / "results/natural_family" / f"checkpoint_seed_{setting['seed']}.pt"
            inputs = root / "results/natural_family" / f"inputs_seed_{setting['seed']}.npz"
            require(hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    == setting["frozen_hashes"]["checkpoint_sha256"], "checkpoint hash differs")
            require(hashlib.sha256(inputs.read_bytes()).hexdigest()
                    == setting["frozen_hashes"]["inputs_sha256"], "input hash differs")

    require(outcome_counts == artifact["aggregate"]["all_outcomes"],
            "all-setting aggregate differs")
    require(replication_counts == {
        key: value for key, value in artifact["aggregate"]["replication_outcomes"].items()
        if value
    }, "replication aggregate differs")
    semantic = artifact["semantic_ground_truth"]
    require(semantic["status"] == "PASS", "semantic control does not pass")
    proof = semantic["proof"]
    require(F(proof["hard_minimax_error"]) == F(1, 2), "semantic hard error differs")
    require(F(proof["fractional_minimax_error"]) == 0, "semantic fractional error differs")
    require(F(proof["decoder_lipschitz"]) == F(20, 19), "semantic decoder differs")
    require(semantic["negative_control"]["signed_jensen_defect"] == "0/1",
            "semantic affine control differs")
    return {
        "status": "PASS",
        "settings": len(artifact["settings"]),
        "replication_settings": artifact["aggregate"]["replication_settings"],
        "outcomes": outcome_counts,
        "protocol_sha256": artifact["protocol_sha256"],
        "production_module_imported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path,
                        default=Path("results/jmlr_defensibility_suite.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        result = replay(args.artifact, args.root)
    except (OSError, KeyError, ValueError, ReplayError) as error:
        print(f"JMLR DEFENSIBILITY REPLAY — FAIL: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
