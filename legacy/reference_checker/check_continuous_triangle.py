#!/usr/bin/env python3
"""Independent exact replay for the continuous-triangle dual sidecar.

This checker imports no production ``tcabb`` module and invokes no LP solver.
It reconstructs polygon half-spaces and every residual LP row from the stored
rational piecewise-affine cells.  It then verifies the sparse nonnegative dual
multipliers, exact dual identity, nonpositive bound, transcript digest, and the
complete exact range-prune/residual partition.

The sidecar is a standalone proof of the stable inequality for its encoded PWA
instance.  Its link to the frozen CNN additionally relies on the checkpoint and
input hashes plus the extractor audits in the canonical benchmark artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


F = Fraction
Point = tuple[Fraction, Fraction]
Affine = tuple[Fraction, Fraction, Fraction]


class SidecarReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SidecarReplayError(message)


def _fraction(value: Sequence[str]) -> Fraction:
    _require(len(value) == 2, "a rational must contain numerator and denominator")
    return F(int(value[0]), int(value[1]))


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _affine(value: Sequence[Sequence[str]]) -> Affine:
    _require(len(value) == 3, "an affine map must have three coefficients")
    return tuple(_fraction(coefficient) for coefficient in value)  # type: ignore[return-value]


def _value(function: Affine, point: Point) -> Fraction:
    return function[0] + function[1] * point[0] + function[2] * point[1]


def _range(function: Affine, polygon: Sequence[Point]) -> tuple[Fraction, Fraction]:
    values = tuple(_value(function, point) for point in polygon)
    return min(values), max(values)


def _cross(left: Point, middle: Point, right: Point) -> Fraction:
    return ((middle[0] - left[0]) * (right[1] - middle[1])
            - (middle[1] - left[1]) * (right[0] - middle[0]))


def _bbox(polygon: Sequence[Point]) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    return (
        min(point[0] for point in polygon),
        max(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[1] for point in polygon),
    )


def _interiors_disjoint(first: Sequence[Point], second: Sequence[Point]) -> bool:
    """Exact separating-axis test; equality permits shared boundaries."""

    for polygon in (first, second):
        for left, right in zip(polygon, polygon[1:] + polygon[:1]):
            axis = (right[1] - left[1], left[0] - right[0])
            first_projection = tuple(axis[0] * x + axis[1] * y for x, y in first)
            second_projection = tuple(axis[0] * x + axis[1] * y for x, y in second)
            if (max(first_projection) <= min(second_projection)
                    or max(second_projection) <= min(first_projection)):
                return True
    return False


def _inside_closed_convex(point: Point, polygon: Sequence[Point]) -> bool:
    return all(
        (right[0] - left[0]) * (point[1] - left[1])
        - (right[1] - left[1]) * (point[0] - left[0]) >= 0
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    )


def _proper_segment_intersection(
    left: Point, right: Point, other_left: Point, other_right: Point
) -> Point | None:
    dx, dy = right[0] - left[0], right[1] - left[1]
    ex, ey = other_right[0] - other_left[0], other_right[1] - other_left[1]
    determinant = dx * ey - dy * ex
    if determinant == 0:
        return None
    ox, oy = other_left[0] - left[0], other_left[1] - left[1]
    amount = (ox * ey - oy * ex) / determinant
    other_amount = (ox * dy - oy * dx) / determinant
    if 0 <= amount <= 1 and 0 <= other_amount <= 1:
        return left[0] + amount * dx, left[1] + amount * dy
    return None


def _intersection_points(first: Sequence[Point], second: Sequence[Point]) -> set[Point]:
    """Return generators of the convex closed intersection, if any."""

    points = {point for point in first if _inside_closed_convex(point, second)}
    points.update(point for point in second if _inside_closed_convex(point, first))
    for left, right in zip(first, first[1:] + first[:1]):
        for other_left, other_right in zip(second, second[1:] + second[:1]):
            point = _proper_segment_intersection(left, right, other_left, other_right)
            if point is not None:
                points.add(point)
    return points


def _audit_anchor_partition(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Prove exact convex partition geometry and boundary-map consistency."""

    _require(sum(cell["area2"] for cell in cells) == 1,
             "cell areas do not sum to the mechanism triangle")
    pair_checks = 0
    intersecting_pairs = 0
    continuity_point_checks = 0
    for first_index, first in enumerate(cells):
        first_polygon = first["polygon"]
        first_bbox = first["bbox"]
        for second in cells[first_index + 1:]:
            pair_checks += 1
            second_polygon = second["polygon"]
            second_bbox = second["bbox"]
            _require(_interiors_disjoint(first_polygon, second_polygon),
                     "two encoded PWA cells have overlapping interiors")
            if (first_bbox[1] < second_bbox[0] or second_bbox[1] < first_bbox[0]
                    or first_bbox[3] < second_bbox[2]
                    or second_bbox[3] < first_bbox[2]):
                continue
            points = _intersection_points(first_polygon, second_polygon)
            if not points:
                continue
            intersecting_pairs += 1
            functions = (
                (first["claim"], second["claim"]),
                *zip(first["hard_responses"], second["hard_responses"]),
                (first["fractional_response"], second["fractional_response"]),
            )
            for point in points:
                for first_function, second_function in functions:
                    _require(_value(first_function, point) == _value(second_function, point),
                             "adjacent PWA formulas disagree on a shared boundary")
                    continuity_point_checks += 1
    return {
        "polygon_pair_checks": pair_checks,
        "boundary_intersecting_pairs": intersecting_pairs,
        "boundary_continuity_point_checks": continuity_point_checks,
    }


def _decode_cell(raw: dict[str, Any]) -> dict[str, Any]:
    polygon = tuple(
        (_fraction(point[0]), _fraction(point[1])) for point in raw["polygon"]
    )
    _require(len(polygon) >= 3, "a PWA cell polygon has fewer than three vertices")
    _require(len(set(polygon)) == len(polygon),
             "a PWA cell polygon repeats a vertex")
    _require(all(left != right for left, right in zip(polygon, polygon[1:] + polygon[:1])),
             "a PWA cell polygon has a zero-length edge")
    _require(
        all(x >= 0 and y >= 0 and x + y <= 1 for x, y in polygon),
        "a PWA cell vertex leaves the declared mechanism triangle",
    )
    signed_twice_area = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    )
    _require(signed_twice_area > 0, "a PWA cell is degenerate or clockwise")
    turns = tuple(
        _cross(polygon[index - 1], polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    _require(all(turn >= 0 for turn in turns) and any(turn > 0 for turn in turns),
             "a PWA cell is not convex and counter-clockwise")
    claim = _affine(raw["claim"])
    responses = tuple(_affine(response) for response in raw["responses"])
    _require(len(responses) == 5, "the active interface must have five responses")
    hard_responses = tuple(_affine(response) for response in raw["hard_responses"])
    _require(len(hard_responses) == 5, "the hard interface must have five responses")
    fractional_response = _affine(raw["fractional_response"])
    _require(
        responses
        == (
            hard_responses[0],
            hard_responses[1],
            fractional_response,
            hard_responses[3],
            hard_responses[4],
        ),
        "the active hybrid response map does not match its hard/fractional components",
    )
    edges = []
    for left, right in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        edges.append((dy, -dx, dy * left[0] - dx * left[1]))
    return {
        "polygon": polygon,
        "area2": signed_twice_area,
        "bbox": _bbox(polygon),
        "claim": claim,
        "claim_range": _range(claim, polygon),
        "responses": responses,
        "response_ranges": tuple(_range(response, polygon) for response in responses),
        "hard_responses": hard_responses,
        "fractional_response": fractional_response,
        "edges": tuple(edges),
    }


def _constraint(
    first: dict[str, Any],
    second: dict[str, Any],
    sign: Fraction,
    lipschitz: Fraction,
    index: int,
) -> tuple[tuple[Fraction, ...], Fraction]:
    first_edges = first["edges"]
    second_edges = second["edges"]
    if index < len(first_edges):
        a, b, rhs = first_edges[index]
        return (a, b, F(0), F(0), F(0)), rhs
    index -= len(first_edges)
    if index < len(second_edges):
        a, b, rhs = second_edges[index]
        return (F(0), F(0), a, b, F(0)), rhs
    index -= len(second_edges)
    _require(0 <= index < 10, "dual support refers to a nonexistent constraint")
    query, parity = divmod(index, 2)
    response_sign = F(1) if parity == 0 else F(-1)
    pc, qc, rc = first["claim"]
    pd, qd, rd = second["claim"]
    pr, qr, rr = first["responses"][query]
    ps, qs, rs = second["responses"][query]
    row = (
        -sign * qc + lipschitz * response_sign * qr,
        -sign * rc + lipschitz * response_sign * rr,
        sign * qd - lipschitz * response_sign * qs,
        sign * rd - lipschitz * response_sign * rs,
        F(1),
    )
    bound = sign * (pc - pd) - lipschitz * response_sign * (pr - ps)
    return row, bound


def _verify_dual(
    first: dict[str, Any],
    second: dict[str, Any],
    sign: Fraction,
    lipschitz: Fraction,
    support: Sequence[int],
    weights: Sequence[Fraction],
    stored_bound: Fraction,
) -> None:
    _require(len(support) == len(weights) == 5, "a sparse dual must have five rows")
    _require(len(set(support)) == 5, "a sparse dual repeats a support row")
    _require(all(weight >= 0 for weight in weights), "a dual weight is negative")
    rows_and_bounds = [
        _constraint(first, second, sign, lipschitz, index) for index in support
    ]
    identity = tuple(
        sum(weight * row[column] for weight, (row, _) in zip(weights, rows_and_bounds))
        for column in range(5)
    )
    _require(identity == (F(0), F(0), F(0), F(0), F(1)), "exact dual identity fails")
    bound = sum(
        weight * rhs for weight, (_, rhs) in zip(weights, rows_and_bounds)
    )
    _require(bound == stored_bound, "stored and reconstructed dual bounds differ")
    _require(bound <= 0, "an exact dual upper bound is positive")


def _load(sidecar: Path) -> dict[str, Any]:
    with gzip.open(sidecar, "rt", encoding="ascii") as handle:
        return json.load(handle)


def _record_fraction(record: dict[str, Any]) -> Fraction:
    value = F(int(record["numerator"]), int(record["denominator"]))
    _require(value.denominator > 0, "a released exact record has invalid denominator")
    return value


def _world_values(
    cells: Sequence[dict[str, Any]], point: Point
) -> tuple[Fraction, tuple[Fraction, ...], Fraction, int]:
    containing = [cell for cell in cells if _inside_closed_convex(point, cell["polygon"])]
    _require(bool(containing), "an exact witness point is outside the encoded partition")
    claims = {_value(cell["claim"], point) for cell in containing}
    hard = {
        tuple(_value(response, point) for response in cell["hard_responses"])
        for cell in containing
    }
    fractional = {
        _value(cell["fractional_response"], point) for cell in containing
    }
    _require(len(claims) == len(hard) == len(fractional) == 1,
             "encoded affine maps disagree at an exact witness point")
    return claims.pop(), hard.pop(), fractional.pop(), len(containing)


def _audit_strict_separation(
    artifact: dict[str, Any],
    lipschitz: Fraction,
    anchors: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> dict[str, str | int]:
    """Independently recompute both exact sides of the released separation."""

    stable = artifact["stable_fractional_interface"]
    stable_twice_upper = _record_fraction(
        stable["proved_twice_minimax_error_upper_bound"]
    )
    _require(stable["result"] == "CERTIFIED" and stable_twice_upper == 0,
             "the fractional-interface exact upper bound is not zero")
    _require(_record_fraction(stable["lipschitz_constant"]) == lipschitz,
             "artifact and sidecar Lipschitz constants differ")

    hard = artifact["hard_interface_obstruction"]
    _require(hard["result"] == "FALSIFIED_EXACTLY",
             "the hard-interface obstruction is not exact")
    claim_delta = _record_fraction(hard["absolute_claim_delta"])
    hard_deltas = tuple(
        _record_fraction(record) for record in hard["signed_hard_response_deltas"]
    )
    _require(len(hard_deltas) == 5, "the hard witness must contain five response deltas")
    hard_maximum = max(abs(delta) for delta in hard_deltas)
    _require(hard_maximum
             == _record_fraction(hard["maximum_absolute_hard_response_delta"]),
             "the reported hard response maximum is inconsistent")
    twice_lower = claim_delta - lipschitz * hard_maximum
    _require(twice_lower == _record_fraction(hard["twice_minimax_error_lower_bound"]),
             "the reported twice-error lower bound is inconsistent")
    minimax_lower = twice_lower / 2
    _require(minimax_lower == _record_fraction(hard["minimax_error_lower_bound"])
             and minimax_lower > 0,
             "the hard minimax lower bound is inconsistent or nonpositive")

    query = artifact["interfaces"]["fractional_query"]
    _require(query == ["0", "0", "1/4", "0", "3/4"],
             "the released fractional query is not 1/4 e3 + 3/4 e5")
    fractional_delta = _record_fraction(hard["signed_fractional_response_delta"])
    _require(abs(fractional_delta)
             == _record_fraction(hard["absolute_fractional_response_delta"]),
             "the fractional response absolute value is inconsistent")
    jensen = fractional_delta - F(1, 4) * hard_deltas[2] - F(3, 4) * hard_deltas[4]
    _require(jensen == _record_fraction(hard["signed_jensen_defect"]),
             "the signed Jensen defect is inconsistent")
    _require(abs(jensen) == _record_fraction(hard["absolute_jensen_defect"])
             and jensen != 0,
             "the absolute Jensen defect is inconsistent or zero")
    witness_memberships = 0
    if anchors is not None:
        _require([row["frozen_index"] for row in artifact["anchors"]]
                 == [8, 9, 10, 11], "artifact anchor order differs")
        _require(hard["anchor_indices"] == [8, 9], "hard witness anchors differ")
        first_parameter, second_parameter = (
            _record_fraction(record) for record in hard["edge_parameters"]
        )
        _require(0 <= first_parameter <= 1 and 0 <= second_parameter <= 1,
                 "hard witness leaves the V0--V1 mechanism edge")
        first_point = (F(1) - first_parameter, first_parameter)
        second_point = (F(1) - second_parameter, second_parameter)
        first_values = _world_values(anchors[0], first_point)
        second_values = _world_values(anchors[1], second_point)
        encoded_claim_delta = first_values[0] - second_values[0]
        encoded_hard_deltas = tuple(
            left - right for left, right in zip(first_values[1], second_values[1])
        )
        encoded_fractional_delta = first_values[2] - second_values[2]
        _require(abs(encoded_claim_delta) == claim_delta,
                 "encoded PWA claim does not reproduce the hard witness")
        _require(encoded_hard_deltas == hard_deltas,
                 "encoded PWA hard responses do not reproduce the witness")
        _require(encoded_fractional_delta == fractional_delta,
                 "encoded PWA fractional response does not reproduce the witness")
        witness_memberships = first_values[3] + second_values[3]
    return {
        "fractional_twice_error_upper": _fraction_text(stable_twice_upper),
        "hard_minimax_error_lower": _fraction_text(minimax_lower),
        "signed_jensen_defect": _fraction_text(jensen),
        "witness_cell_memberships": witness_memberships,
    }


def replay_sidecar(
    sidecar: Path,
    *,
    artifact: Path | None = None,
    project_root: Path | None = None,
    verify_completeness: bool = True,
) -> dict[str, Any]:
    sidecar = Path(sidecar)
    payload = _load(sidecar)
    _require(
        payload["schema"] == "tcabb.continuous_triangle_duals.v2",
        "unknown dual-sidecar schema",
    )
    lipschitz = _fraction(payload["lipschitz"])
    _require(lipschitz == 14, "the released stable constant differs")
    anchors = [
        [_decode_cell(cell) for cell in anchor] for anchor in payload["anchors"]
    ]
    _require([len(anchor) for anchor in anchors] == [226, 282, 802, 379],
             "released PWA cell counts differ")
    geometry_records = [_audit_anchor_partition(anchor) for anchor in anchors]

    verification = payload["verification"]
    certificates = payload["certificates"]
    _require(
        len(certificates) == verification["certificate_count"] == 32_296,
        "dual certificate count differs",
    )
    separation_record = None
    if artifact is not None:
        released = json.loads(Path(artifact).read_text(encoding="utf-8"))
        record = released["dual_certificate_sidecar"]
        _require(record["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                 "canonical artifact does not bind this sidecar")
        _require(record["certificate_count"] == len(certificates),
                 "canonical and sidecar certificate counts differ")
        _require(released["frozen_artifacts"]["checkpoint_sha256"]
                 == payload["checkpoint_sha256"], "checkpoint bindings differ")
        _require(released["frozen_artifacts"]["inputs_sha256"]
                 == payload["inputs_sha256"], "input bindings differ")
        _require(released["stable_fractional_interface"]["dual_transcript_sha256"]
                 == verification["dual_transcript_sha256"],
                 "canonical and sidecar transcript digests differ")
        separation_record = _audit_strict_separation(released, lipschitz, anchors)
    if project_root is not None:
        root = Path(project_root)
        checkpoint = root / "results/natural_family/checkpoint_seed_20260714.pt"
        inputs = root / "results/natural_family/inputs_seed_20260714.npz"
        _require(hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                 == payload["checkpoint_sha256"], "frozen checkpoint hash differs")
        _require(hashlib.sha256(inputs.read_bytes()).hexdigest()
                 == payload["inputs_sha256"], "frozen input hash differs")

    if not verify_completeness:
        return {
            "status": "STRUCTURE_ONLY",
            "anchors": len(anchors),
            "cells": sum(map(len, anchors)),
            "certificates": len(certificates),
            "geometry": geometry_records,
            "strict_separation": separation_record,
        }

    counts = {
        "cell_sign_cases": 0,
        "exact_claim_range_prunes": 0,
        "exact_response_range_prunes": 0,
        "exact_dual_certificates": 0,
        "lp_support_proposals": 0,
        "unresolved": 0,
    }
    transcript = hashlib.sha256()
    certificate_index = 0
    for first_anchor in range(len(anchors)):
        for second_anchor in range(first_anchor, len(anchors)):
            for first_index, first in enumerate(anchors[first_anchor]):
                for second_index, second in enumerate(anchors[second_anchor]):
                    claim_low = first["claim_range"][0] - second["claim_range"][1]
                    claim_high = first["claim_range"][1] - second["claim_range"][0]
                    for sign in (F(1), F(-1)):
                        counts["cell_sign_cases"] += 1
                        claim_maximum = claim_high if sign == 1 else -claim_low
                        if claim_maximum <= 0:
                            counts["exact_claim_range_prunes"] += 1
                            continue
                        separated = False
                        for first_range, second_range in zip(
                            first["response_ranges"], second["response_ranges"]
                        ):
                            low = first_range[0] - second_range[1]
                            high = first_range[1] - second_range[0]
                            minimum = min(abs(low), abs(high)) if low > 0 or high < 0 else F(0)
                            if claim_maximum - lipschitz * minimum <= 0:
                                separated = True
                                break
                        if separated:
                            counts["exact_response_range_prunes"] += 1
                            continue

                        _require(certificate_index < len(certificates),
                                 "a residual LP has no stored dual certificate")
                        raw = certificates[certificate_index]
                        _require(len(raw) == 8, "a dual record has the wrong arity")
                        expected_case = (
                            first_anchor,
                            second_anchor,
                            first_index,
                            second_index,
                            int(sign),
                        )
                        actual_case = tuple(int(value) for value in raw[:5])
                        _require(actual_case == expected_case,
                                 "stored dual order omits or substitutes a residual case")
                        support = tuple(int(value) for value in raw[5])
                        weights = tuple(_fraction(value) for value in raw[6])
                        bound = _fraction(raw[7])
                        _verify_dual(
                            first, second, sign, lipschitz, support, weights, bound
                        )
                        transcript.update(
                            (
                                f"{first_anchor},{second_anchor},{first_index},"
                                f"{second_index},{int(sign)}:"
                            ).encode("ascii")
                        )
                        transcript.update((",".join(map(str, support)) + ":").encode("ascii"))
                        transcript.update(
                            (",".join(map(_fraction_text, weights)) + ":").encode("ascii")
                        )
                        transcript.update(_fraction_text(bound).encode("ascii"))
                        transcript.update(b"\n")
                        certificate_index += 1
                        counts["lp_support_proposals"] += 1
                        counts["exact_dual_certificates"] += 1
    _require(certificate_index == len(certificates), "sidecar has extra dual records")
    _require(counts == verification["counts"], "independent case partition differs")
    digest = transcript.hexdigest()
    _require(digest == verification["dual_transcript_sha256"],
             "independent dual transcript digest differs")
    return {
        "status": "PASS",
        "anchors": len(anchors),
        "cells": sum(map(len, anchors)),
        "counts": counts,
        "dual_transcript_sha256": digest,
        "lp_solver_used": False,
        "production_module_imported": False,
        "geometry": geometry_records,
        "strict_separation": separation_record,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("results/continuous_triangle_duals.json.gz"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("results/continuous_triangle_benchmark.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="check bindings and schema without replaying all exact duals",
    )
    args = parser.parse_args(argv)
    try:
        result = replay_sidecar(
            args.sidecar,
            artifact=args.artifact,
            project_root=args.root,
            verify_completeness=not args.structure_only,
        )
    except (OSError, KeyError, ValueError, SidecarReplayError) as error:
        print(f"CONTINUOUS TRIANGLE REPLAY — FAIL: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
