"""Resumable local Codex CLI transport for semantic annotation batches."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ccpu.common.artifacts import file_sha256, read_json, write_json, write_jsonl

_TOKEN_COUNT = re.compile(r"tokens used\s*\r?\n\s*([0-9,]+)", re.IGNORECASE)


def _run_batch(
    request_path: Path,
    *,
    output_dir: Path,
    prompt_path: Path,
    schema_path: Path,
    repo_root: Path,
    executable: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    annotation_path = output_dir / "annotations" / request_path.name
    stdout_path = output_dir / "logs" / f"{request_path.stem}.stdout.log"
    stderr_path = output_dir / "logs" / f"{request_path.stem}.stderr.log"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if annotation_path.exists():
        payload = read_json(annotation_path)
        if isinstance(payload.get("annotations"), list):
            return {
                "batch": request_path.name,
                "status": "resumed",
                "annotation_count": len(payload["annotations"]),
                "tokens": None,
                "duration_seconds": 0.0,
            }

    command = [
        executable,
        "exec",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-s",
        "read-only",
        "-C",
        str(repo_root),
        "--output-schema",
        str(schema_path),
        "-o",
        str(annotation_path),
        "--ephemeral",
        f"Follow {prompt_path}. Request batch: {request_path}",
    ]
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    duration = time.perf_counter() - started
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    token_match = _TOKEN_COUNT.search(process.stderr)
    tokens = int(token_match.group(1).replace(",", "")) if token_match else None
    if process.returncode != 0:
        return {
            "batch": request_path.name,
            "status": "failed",
            "returncode": process.returncode,
            "tokens": tokens,
            "duration_seconds": round(duration, 3),
        }
    try:
        payload = read_json(annotation_path)
        annotations = payload["annotations"]
        if not isinstance(annotations, list):
            raise TypeError("annotations must be a list")
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
        return {
            "batch": request_path.name,
            "status": "invalid_output",
            "error": str(error),
            "tokens": tokens,
            "duration_seconds": round(duration, 3),
        }
    return {
        "batch": request_path.name,
        "status": "completed",
        "annotation_count": len(annotations),
        "tokens": tokens,
        "duration_seconds": round(duration, 3),
    }


def run_local_codex_batches(
    requests_dir: str | Path,
    output_dir: str | Path,
    *,
    prompt_path: str | Path,
    schema_path: str | Path,
    repo_root: str | Path,
    executable: str = "codex",
    model: str = "gpt-5.4",
    reasoning_effort: str = "medium",
    concurrency: int = 4,
) -> dict[str, Any]:
    """Run outstanding request batches and merge valid structured outputs."""

    resolved_executable = shutil.which(executable) or executable
    request_paths = sorted(Path(requests_dir).glob("batch_*.json"))
    if not request_paths:
        raise ValueError(f"no batch_*.json files found in {requests_dir}")
    output = Path(output_dir)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_batch,
                request,
                output_dir=output,
                prompt_path=Path(prompt_path).resolve(),
                schema_path=Path(schema_path).resolve(),
                repo_root=Path(repo_root).resolve(),
                executable=resolved_executable,
                model=model,
                reasoning_effort=reasoning_effort,
            ): request
            for request in request_paths
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["batch"])

    annotations = []
    output_hashes = {}
    for result in results:
        path = output / "annotations" / result["batch"]
        if result["status"] not in {"completed", "resumed"}:
            continue
        output_hashes[result["batch"]] = file_sha256(path)
        annotations.extend(read_json(path)["annotations"])
    aggregate = write_jsonl(output / "annotations.jsonl", annotations)
    manifest = {
        "schema_version": "ccpu.dsl_dataset.local_codex_run.v1",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "sandbox": "read-only",
        "concurrency": concurrency,
        "batch_count": len(request_paths),
        "completed_count": sum(row["status"] in {"completed", "resumed"} for row in results),
        "failed_count": sum(row["status"] not in {"completed", "resumed"} for row in results),
        "annotation_count": len(annotations),
        "reported_tokens": sum(row["tokens"] or 0 for row in results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_sha256": file_sha256(prompt_path),
        "schema_sha256": file_sha256(schema_path),
        "output_sha256": file_sha256(aggregate),
        "batch_output_sha256": output_hashes,
        "batches": results,
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest
