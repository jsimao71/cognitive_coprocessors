"""Public heterogeneous-compute benchmark selection and stratification."""

from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import (
    canonical_json,
    environment_manifest,
    file_sha256,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ccpu.common.public_benchmarks import (
    PublicSource,
    load_config,
    read_verified_parquet,
    stable_row_key,
    stratified_select,
    write_selection_artifacts,
)
from ccpu.common.schema import DetectionCandidate
from ccpu.paper1.arithmetic import ArithmeticNormalizer, BoundedCalculator

from .diagnostic import _t1, _t2

Normalizer = Callable[[PublicSource, int, dict[str, Any], int], dict[str, Any]]


def _content_sha(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()


def _record(
    source: PublicSource,
    source_row: int,
    row: dict[str, Any],
    seed: int,
    *,
    example_id: str,
    target_label: str,
    difficulty: int,
    difficulty_stratum: str,
) -> dict[str, Any]:
    return {
        "benchmark": source.benchmark,
        "engine": source.engine,
        "example_id": example_id,
        "source_row": source_row,
        "target_label": target_label,
        "difficulty": difficulty,
        "difficulty_stratum": difficulty_stratum,
        "content_sha256": _content_sha(row),
        "selection_key": stable_row_key(seed, source.benchmark, source_row, row),
    }


def _gsm8k(source: PublicSource, index: int, row: dict[str, Any], seed: int) -> dict[str, Any]:
    steps = max(1, str(row["answer"]).count("<<"))
    answer_match = re.search(r"####\s*([^\r\n]+)", str(row["answer"]))
    if not answer_match:
        raise ValueError(f"GSM8K row {index} has no final answer marker")
    stratum = "2_steps" if steps <= 2 else "3_4_steps" if steps <= 4 else "5plus_steps"
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"gsm8k:test:{index}",
        target_label=answer_match.group(1).strip(),
        difficulty=steps,
        difficulty_stratum=stratum,
    )


def _unit_conversion(
    source: PublicSource, index: int, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    prompt = str(row["inputs"]).lower()
    power_count = prompt.count("^")
    rate_count = prompt.count(" per ") + prompt.count("/")
    difficulty = 1 + rate_count + power_count
    stratum = "powered" if power_count else "compound_rate" if rate_count else "linear"
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"bigbench:unit_conversion:{row['idx']}",
        target_label=str(row["targets"][0]),
        difficulty=difficulty,
        difficulty_stratum=stratum,
    )


_DATE_TOKEN = re.compile(
    r"\b(?:\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_TEMPORAL_CUE = re.compile(
    r"\b(?:before|after|later|earlier|tomorrow|yesterday|ago|from now|next|previous)\b",
    re.IGNORECASE,
)


def _date_understanding(
    source: PublicSource, index: int, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    prompt = str(row["inputs"])
    difficulty = max(1, len(_DATE_TOKEN.findall(prompt)) + len(_TEMPORAL_CUE.findall(prompt)))
    stratum = "single_step" if difficulty <= 2 else "two_step" if difficulty <= 4 else "multi_step"
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"bigbench:date_understanding:{row['idx']}",
        target_label=str(row["targets"][0]),
        difficulty=difficulty,
        difficulty_stratum=stratum,
    )


def _proofwriter(
    source: PublicSource, index: int, row: dict[str, Any], seed: int
) -> dict[str, Any]:
    depth = int(row["QDep"])
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"proofwriter:{row['id']}",
        target_label=str(row["answer"]).upper(),
        difficulty=depth,
        difficulty_stratum=f"depth_{depth}",
    )


def _clutrr(source: PublicSource, index: int, row: dict[str, Any], seed: int) -> dict[str, Any]:
    edges = ast.literal_eval(str(row["edge_types"]))
    if not isinstance(edges, list):
        raise TypeError(f"CLUTRR row {index} has invalid edge list")
    depth = len(edges)
    return _record(
        source,
        index,
        row,
        seed,
        example_id=f"clutrr:{row['id']}",
        target_label=str(row["target_text"]),
        difficulty=depth,
        difficulty_stratum=f"depth_{depth}",
    )


_NORMALIZERS: dict[str, Normalizer] = {
    "gsm8k": _gsm8k,
    "bigbench_unit_conversion": _unit_conversion,
    "bigbench_date_understanding": _date_understanding,
    "proofwriter_balanced": _proofwriter,
    "clutrr": _clutrr,
}


def freeze_public_suite(
    config_path: str | Path, cache_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    seed, sources, config = load_config(config_path)
    selected: list[dict[str, Any]] = []
    for source in sources:
        try:
            normalizer = _NORMALIZERS[source.benchmark]
        except KeyError as error:
            raise ValueError(f"no normalizer for {source.benchmark}") from error
        source_rows = read_verified_parquet(source, cache_root)
        records = [
            normalizer(source, index, row, seed) for index, row in enumerate(source_rows)
        ]
        selected.extend(stratified_select(records, source.max_rows, seed))
    return write_selection_artifacts(output_dir, config, sources, selected)


def _prompt_and_row_target(source: PublicSource, row: dict[str, Any]) -> tuple[str, str]:
    if source.benchmark == "gsm8k":
        match = re.search(r"####\s*([^\r\n]+)", str(row["answer"]))
        if not match:
            raise ValueError("GSM8K row has no final answer marker")
        return str(row["question"]), match.group(1).strip()
    if source.benchmark in {"bigbench_unit_conversion", "bigbench_date_understanding"}:
        return str(row["inputs"]), str(row["targets"][0])
    if source.benchmark == "proofwriter_balanced":
        return f"{row['theory']} Question: {row['question']}", str(row["answer"]).upper()
    if source.benchmark == "clutrr":
        return f"{row['story']} Query: {row['query']}", str(row["target_text"])
    raise ValueError(f"unsupported public benchmark: {source.benchmark}")


def _materialize_selected(
    config_path: str | Path, cache_root: str | Path, selection_path: str | Path
) -> tuple[list[dict[str, Any]], list[PublicSource]]:
    seed, sources, _ = load_config(config_path)
    selected = read_jsonl(selection_path)
    selected_by_source = {
        (str(row["benchmark"]), int(row["source_row"])): row for row in selected
    }
    if len(selected_by_source) != len(selected):
        raise ValueError("public selection contains duplicate source rows")

    materialized = []
    for source in sources:
        rows = read_verified_parquet(source, cache_root)
        normalizer = _NORMALIZERS[source.benchmark]
        for index, raw_row in enumerate(rows):
            selected_row = selected_by_source.get((source.benchmark, index))
            if selected_row is None:
                continue
            normalized = normalizer(source, index, raw_row, seed)
            for key in ("content_sha256", "example_id", "selection_key", "target_label"):
                if normalized[key] != selected_row[key]:
                    raise ValueError(
                        f"selected {source.benchmark} row {index} changed at field {key}"
                    )
            prompt, target = _prompt_and_row_target(source, raw_row)
            materialized.append({**selected_row, "prompt": prompt, "target": target, "raw": raw_row})
    if len(materialized) != len(selected):
        raise ValueError("not every selected public row was materialized")
    return materialized, sources


_GSM_TRACE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def _fraction(raw: str) -> Fraction:
    value = raw.strip().replace(",", "").replace("$", "")
    return Fraction(value)


def _gsm_execution(row: dict[str, Any]) -> dict[str, Any]:
    traces = _GSM_TRACE.findall(str(row["raw"]["answer"]))
    if not traces:
        return {
            "formalization_oracle": False,
            "backend_compatible": False,
            "execution_exact": None,
            "operation_count": 0,
            "failure": "no annotated arithmetic trace",
        }
    normalizer = ArithmeticNormalizer()
    calculator = BoundedCalculator()
    for index, (expression, expected) in enumerate(traces):
        cleaned = expression.replace(",", "").replace("$", "").strip()
        candidate = DetectionCandidate(
            candidate_id=f"{row['example_id']}:{index}",
            family="compute",
            raw_text=cleaned,
            start_offset=0,
            end_offset=len(cleaned),
            detector="public_gold_trace",
        )
        try:
            request = normalizer.normalize(candidate)
            result = calculator.execute(request)
            exact = result.ok and _fraction(result.display) == _fraction(expected)
        except (ValueError, ZeroDivisionError) as error:
            return {
                "formalization_oracle": True,
                "backend_compatible": False,
                "execution_exact": None,
                "operation_count": len(traces),
                "failure": str(error),
            }
        if not exact:
            return {
                "formalization_oracle": True,
                "backend_compatible": result.ok,
                "execution_exact": False,
                "operation_count": len(traces),
                "failure": result.error_message or "gold trace result mismatch",
            }
    return {
        "formalization_oracle": True,
        "backend_compatible": True,
        "execution_exact": True,
        "operation_count": len(traces),
        "failure": None,
    }


def _compatibility(row: dict[str, Any]) -> dict[str, Any]:
    benchmark = str(row["benchmark"])
    if benchmark == "gsm8k":
        return _gsm_execution(row)
    if benchmark == "clutrr":
        return {
            "formalization_oracle": True,
            "backend_compatible": False,
            "execution_exact": None,
            "operation_count": int(row["difficulty"]),
            "failure": "CLUTRR relation path is not ISA/frame closure IR",
        }
    reasons = {
        "bigbench_unit_conversion": "multiple-choice answer is not dimensional conversion IR",
        "bigbench_date_understanding": "multiple-choice answer is not typed ISO date IR",
        "proofwriter_balanced": "controlled English is not bounded Horn IR with open-world negation",
    }
    return {
        "formalization_oracle": False,
        "backend_compatible": False,
        "execution_exact": None,
        "operation_count": 0,
        "failure": reasons[benchmark],
    }


def _safe_rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze_public_coverage(
    config_path: str | Path,
    cache_root: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    materialized, sources = _materialize_selected(config_path, cache_root, selection_path)
    predictions = []
    for row in materialized:
        compatibility = _compatibility(row)
        gold_engine = str(row["engine"]).upper().replace("_TIME", "")
        predictions.append(
            {
                "schema_version": "ccpu.paper2.public_coverage_prediction.v1",
                "benchmark": row["benchmark"],
                "example_id": row["example_id"],
                "source_row": row["source_row"],
                "difficulty": row["difficulty"],
                "difficulty_stratum": row["difficulty_stratum"],
                "gold_engine": gold_engine,
                "t1_engine": _t1(str(row["prompt"])),
                "t2_engine": _t2(str(row["prompt"])),
                **compatibility,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    curves: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["benchmark"])].append(row)
        curves[(str(row["benchmark"]), str(row["difficulty_stratum"]))].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        executable = [row for row in rows if row["execution_exact"] is not None]
        return {
            "count": len(rows),
            "t1_engine_recall": _safe_rate(
                [row["t1_engine"] == row["gold_engine"] for row in rows]
            ),
            "t2_engine_recall": _safe_rate(
                [row["t2_engine"] == row["gold_engine"] for row in rows]
            ),
            "formalization_oracle_coverage": _safe_rate(
                [bool(row["formalization_oracle"]) for row in rows]
            ),
            "backend_contract_coverage": _safe_rate(
                [bool(row["backend_compatible"]) for row in rows]
            ),
            "execution_exact_rate_on_attempted": _safe_rate(
                [bool(row["execution_exact"]) for row in executable]
            ),
            "execution_attempt_count": len(executable),
        }

    summary = {
        "schema_version": "ccpu.paper2.public_coverage_analysis.v1",
        "selection_sha256": file_sha256(selection_path),
        "record_count": len(predictions),
        "per_benchmark": {
            benchmark: summarize(rows) for benchmark, rows in sorted(grouped.items())
        },
        "difficulty_curves": [
            {"benchmark": benchmark, "difficulty_stratum": stratum, **summarize(rows)}
            for (benchmark, stratum), rows in sorted(curves.items())
        ],
        "overall": summarize(predictions),
        "interpretation": {
            "status": "backend_gap",
            "paper3_gate": "no_go",
            "claim": "intent routing must not be reported as public task execution",
        },
        "sources": [source.provenance() for source in sources],
        "environment": environment_manifest(Path(__file__).resolve().parents[3]),
    }
    output = Path(output_dir)
    predictions_path = write_jsonl(output / "predictions.jsonl", predictions)
    summary["predictions_sha256"] = file_sha256(predictions_path)
    write_json(output / "summary.json", summary)
    _plot_public_coverage(summary, output / "public_coverage.png")
    return summary


def _plot_public_coverage(summary: dict[str, Any], output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("public coverage plots require the analysis extra") from error
    labels = list(summary["per_benchmark"])
    short = {
        "bigbench_date_understanding": "Date",
        "bigbench_unit_conversion": "Units",
        "clutrr": "CLUTRR",
        "gsm8k": "GSM8K",
        "proofwriter_balanced": "ProofWriter",
    }
    x = list(range(len(labels)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.4, 3.8))
    for offset, (key, title) in enumerate(
        (
            ("t1_engine_recall", "T1 route"),
            ("t2_engine_recall", "T2 route"),
            ("backend_contract_coverage", "Exact backend"),
        )
    ):
        values = [float(summary["per_benchmark"][label][key] or 0.0) for label in labels]
        axis.bar([value + (offset - 1) * width for value in x], values, width, label=title)
    axis.set_xticks(x, [short[label] for label in labels])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Coverage / recall")
    axis.set_title("Public benchmark routing is not backend coverage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
