"""Click CLI for mining and supervising NL-to-ASL datasets."""

from __future__ import annotations

from pathlib import Path

import click

from ccpu.common.artifacts import read_json

from .annotated import bootstrap_annotated
from .audit import audit_chops
from .expansion import finalize_asl_expansion
from .local_codex import run_local_codex_batches
from .mine import mine_datasets
from .select import select_diverse_seed, select_seed
from .semantic import (
    prepare_local_annotation_batches,
    prepare_repair_batches,
    validate_semantic_annotations,
)
from .teacher import generate_teacher_mappings, prepare_teacher_requests


def _sources(values: tuple[str, ...]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise click.BadParameter("sources must use DATASET=PATH")
        name, path = value.split("=", 1)
        if name in result:
            raise click.BadParameter(f"duplicate source for {name}")
        result[name] = Path(path)
    return result


def _source_splits(values: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise click.BadParameter("source splits must use DATASET=SPLIT")
        name, split = value.split("=", 1)
        if not name or not split:
            raise click.BadParameter("source splits must use non-empty DATASET=SPLIT")
        result[name] = split
    return result


@click.group()
def main() -> None:
    """Build execution-verified ASL-Arith compiler datasets."""


@main.command("mine")
@click.option("--source", "source_values", multiple=True, required=True, help="DATASET=PATH")
@click.option(
    "--source-split",
    "source_split_values",
    multiple=True,
    help="Per-source DATASET=SPLIT override.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--split", default="train", show_default=True)
def mine_command(
    source_values: tuple[str, ...],
    source_split_values: tuple[str, ...],
    output_dir: Path,
    split: str,
) -> None:
    manifest = mine_datasets(
        _sources(source_values),
        output_dir,
        split=split,
        source_splits=_source_splits(source_split_values),
    )
    click.echo(f"mined {len(manifest['datasets'])} datasets -> {output_dir}")


@main.command("audit-chops")
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--sample-per-dataset", default=50, type=click.IntRange(min=1))
@click.option("--output", type=click.Path(path_type=Path), required=True)
def audit_command(input_dir: Path, sample_per_dataset: int, output: Path) -> None:
    report = audit_chops(input_dir, output, sample_per_dataset=sample_per_dataset)
    click.echo(
        f"audited {len(report['datasets'])} datasets; "
        f"scale_teacher_generation={report['scale_teacher_generation']}"
    )


@main.command("select")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--max-examples", type=click.IntRange(min=1), required=True)
@click.option("--seed", default=23003, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def select_command(input_path: Path, max_examples: int, seed: int, output: Path) -> None:
    manifest = select_seed(input_path, output, max_examples=max_examples, seed=seed)
    click.echo(f"selected {manifest['record_count']} execution candidates -> {output}")


@main.command("select-diverse")
@click.option(
    "--input",
    "input_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option(
    "--exclude", "exclude_paths", type=click.Path(exists=True, path_type=Path), multiple=True
)
@click.option("--dataset-target", "dataset_targets", multiple=True, required=True)
@click.option("--seed", default=53011, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def select_diverse_command(
    input_paths: tuple[Path, ...],
    exclude_paths: tuple[Path, ...],
    dataset_targets: tuple[str, ...],
    seed: int,
    output: Path,
) -> None:
    targets = {}
    for value in dataset_targets:
        dataset, separator, count = value.partition("=")
        if not separator or not dataset or int(count) < 1:
            raise click.BadParameter("dataset targets must use dataset=positive_count")
        targets[dataset] = int(count)
    manifest = select_diverse_seed(
        list(input_paths),
        output,
        dataset_targets=targets,
        exclude_paths=list(exclude_paths),
        seed=seed,
    )
    click.echo(f"selected {manifest['record_count']} relation-diverse candidates -> {output}")


@main.command("run-local")
@click.option("--requests-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--prompt", "prompt_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--schema", "schema_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--repo-root", type=click.Path(exists=True, path_type=Path), default=Path.cwd())
@click.option("--executable", default="codex", show_default=True)
@click.option("--model", default="gpt-5.4", show_default=True)
@click.option("--reasoning-effort", default="medium", show_default=True)
@click.option("--concurrency", default=4, type=click.IntRange(min=1, max=16), show_default=True)
def run_local_command(
    requests_dir: Path,
    output_dir: Path,
    prompt_path: Path,
    schema_path: Path,
    repo_root: Path,
    executable: str,
    model: str,
    reasoning_effort: str,
    concurrency: int,
) -> None:
    manifest = run_local_codex_batches(
        requests_dir,
        output_dir,
        prompt_path=prompt_path,
        schema_path=schema_path,
        repo_root=repo_root,
        executable=executable,
        model=model,
        reasoning_effort=reasoning_effort,
        concurrency=concurrency,
    )
    click.echo(
        f"completed {manifest['completed_count']}/{manifest['batch_count']} batches; "
        f"annotations={manifest['annotation_count']} -> {output_dir}"
    )


@main.command("finalize-expansion")
@click.option(
    "--accepted", "candidate_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--existing", "existing_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--frozen-ledger", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--target", default=350, type=click.IntRange(min=1), show_default=True)
@click.option("--seed", default=53011, show_default=True)
def finalize_expansion_command(
    candidate_path: Path,
    existing_path: Path,
    frozen_ledger: Path,
    output_dir: Path,
    target: int,
    seed: int,
) -> None:
    manifest = finalize_asl_expansion(
        candidate_path,
        existing_path,
        frozen_ledger,
        output_dir,
        target=target,
        seed=seed,
    )
    click.echo(
        f"froze {manifest['selected_count']} new programs; "
        f"combined={manifest['combined_count']} -> {output_dir}"
    )


@main.command("bootstrap-annotated")
@click.option(
    "--input",
    "input_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def bootstrap_annotated_command(input_paths: tuple[Path, ...], output_dir: Path) -> None:
    summary = bootstrap_annotated(list(input_paths), output_dir)
    click.echo(
        f"accepted {summary['accepted_count']} annotation-derived ASL programs; "
        f"rejected {summary['rejected_count']} -> {output_dir}"
    )


@main.command("validate-semantic")
@click.option(
    "--seed",
    "seed_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option(
    "--annotations",
    "annotation_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def validate_semantic_command(
    seed_paths: tuple[Path, ...], annotation_paths: tuple[Path, ...], output_dir: Path
) -> None:
    summary = validate_semantic_annotations(list(seed_paths), list(annotation_paths), output_dir)
    click.echo(
        f"accepted {summary['accepted_count']} semantic ASL programs; "
        f"rejected {summary['rejected_count']} -> {output_dir}"
    )


@main.command("prepare-local")
@click.option(
    "--seed",
    "seed_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--batch-size", default=5, type=click.IntRange(min=1, max=25), show_default=True)
def prepare_local_command(seed_paths: tuple[Path, ...], output_dir: Path, batch_size: int) -> None:
    manifest = prepare_local_annotation_batches(list(seed_paths), output_dir, batch_size=batch_size)
    click.echo(
        f"prepared {manifest['example_count']} answer-free examples in "
        f"{manifest['batch_count']} local batches -> {output_dir}"
    )


@main.command("prepare-repair")
@click.option(
    "--seed",
    "seed_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option(
    "--rejected", "rejected_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--batch-size", default=5, type=click.IntRange(min=1, max=25), show_default=True)
@click.option("--repair-round", default=1, type=click.IntRange(min=1, max=5), show_default=True)
def prepare_repair_command(
    seed_paths: tuple[Path, ...],
    rejected_path: Path,
    output_dir: Path,
    batch_size: int,
    repair_round: int,
) -> None:
    manifest = prepare_repair_batches(
        list(seed_paths),
        rejected_path,
        output_dir,
        batch_size=batch_size,
        repair_round=repair_round,
    )
    click.echo(
        f"prepared {manifest['example_count']} rationale-assisted repairs in "
        f"{manifest['batch_count']} batches -> {output_dir}"
    )


@main.command("prepare-teacher")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--skill", "skill_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--max-examples", type=click.IntRange(min=1))
@click.option("--output", type=click.Path(path_type=Path), required=True)
def prepare_teacher_command(
    input_path: Path, skill_path: Path, max_examples: int | None, output: Path
) -> None:
    manifest = prepare_teacher_requests(input_path, skill_path, output, max_examples=max_examples)
    click.echo(f"prepared {manifest['request_count']} local/remote teacher requests -> {output}")


@main.command("generate")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--skill", "skill_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--max-examples", type=click.IntRange(min=1))
@click.option("--retries", default=2, type=click.IntRange(min=0, max=5), show_default=True)
@click.option("--dry-run", is_flag=True, help="Write request envelopes without remote calls.")
def generate_command(
    input_path: Path,
    config_path: Path,
    skill_path: Path,
    output_dir: Path,
    max_examples: int | None,
    retries: int,
    dry_run: bool,
) -> None:
    if dry_run:
        manifest = prepare_teacher_requests(
            input_path,
            skill_path,
            output_dir / "requests.jsonl",
            max_examples=max_examples,
        )
        click.echo(f"dry run: prepared {manifest['request_count']} requests; remote_calls=0")
        return
    config = read_json(config_path)
    summary = generate_teacher_mappings(
        input_path,
        skill_path,
        config,
        output_dir,
        max_examples=max_examples,
        retries=retries,
    )
    click.echo(
        f"accepted {summary['accepted_count']}/{summary['raw_count']} teacher mappings -> {output_dir}"
    )


if __name__ == "__main__":
    main()
