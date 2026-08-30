"""Click CLI for mining and supervising NL-to-ASL datasets."""

from __future__ import annotations

from pathlib import Path

import click

from ccpu.common.artifacts import read_json

from .annotated import bootstrap_annotated
from .audit import audit_chops
from .mine import mine_datasets
from .select import select_seed
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
