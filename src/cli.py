"""Command line entry point: ``python -m src.cli <command>``."""

import json
import sys

import click

from src.config import settings

# Windows consoles default to cp1252, which cannot print some LLM characters.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@click.group()
def cli() -> None:
    """Northstar support RAG assistant."""


@cli.command("chunks")
def chunks_cmd() -> None:
    """Show the chunk-count summary and a sample chunk."""
    from src.ingestion import load_chunks, summarize

    click.echo(summarize(load_chunks()))


@cli.command("build-index")
def build_index_cmd() -> None:
    """Rebuild the Qdrant index from data/knowledge_base/."""
    from src.retrieval import build_index

    manifest = build_index()
    click.echo(f"Index rebuilt at {settings.vector_storage_path}")
    click.echo(json.dumps(manifest, indent=2))


@cli.command("calibrate")
def calibrate_cmd() -> None:
    """Choose k, threshold and min_coverage on calibration.json and freeze them."""
    from src.calibration import SETTINGS_FILE, calibrate

    frozen = calibrate()
    click.echo(f"Frozen to {SETTINGS_FILE}")
    for key in ("k", "threshold", "min_coverage", "calibration_accuracy",
                "calibration_recall_at_k", "calibration_margin"):
        click.echo(f"  {key:<24} {frozen[key]}")
    click.echo("\n  id     expected  decision  reason          top_score  coverage  top_document")
    for row in frozen["calibration_rows"]:
        expected = "abstain" if row["should_abstain"] else "answer"
        click.echo(
            f"  {row['id']:<6} {expected:<9} {row['decision']:<9} {row['reason']:<15} "
            f"{row['top_score']:<10} {row['coverage']:<9} {row['top_document']}"
        )


@cli.command("search")
@click.argument("query")
@click.option("-k", type=int, default=None, help="Number of chunks (default: frozen k).")
def search_cmd(query: str, k: int | None) -> None:
    """Show the top-k chunks, scores and the answer/abstain decision for QUERY."""
    from src.abstention import decide
    from src.calibration import load_frozen_settings
    from src.retrieval import SentenceTransformerEmbedder, VectorIndex

    frozen = load_frozen_settings()
    k = k or frozen["k"]
    index = VectorIndex(SentenceTransformerEmbedder())
    try:
        hits = index.search(query, k)
    finally:
        index.close()
    for hit in hits:
        click.echo(f"{hit.rank}. [{hit.document_id}] score={hit.score:.3f}  ({hit.chunk_id})")
        click.echo(f"   {hit.text}\n")
    decision = decide(query, hits, frozen["threshold"], frozen["min_coverage"])
    verdict = "ANSWER" if decision.answer else "ABSTAIN"
    click.echo(
        f"Decision: {verdict} ({decision.reason}; top_score={decision.top_score:.3f} "
        f"vs {frozen['threshold']}, coverage={decision.coverage:.2f} vs "
        f"{frozen['min_coverage']}; settings from {frozen['source']})"
    )


@cli.command("ask")
@click.argument("question")
@click.option("--mode", type=click.Choice(["extractive", "llm"]), default="extractive",
              show_default=True, help="extractive needs no API key; llm uses LLM_PROVIDER.")
@click.option("--show-context", is_flag=True, help="Print the retrieved chunks too.")
def ask_cmd(question: str, mode: str, show_context: bool) -> None:
    """Answer QUESTION from the knowledge base with citations, or abstain."""
    from src.pipeline import RagPipeline

    pipeline = RagPipeline()
    try:
        result = pipeline.ask(question, mode)
    finally:
        pipeline.close()

    click.echo(result.answer)
    click.echo(
        f"\n[{'answered' if result.answered else 'abstained'}: {result.reason} | mode={result.mode}"
        f" model={result.model} | top_score={result.decision.top_score:.3f}"
        f" coverage={result.decision.coverage:.2f}]"
    )
    for event in result.safety_events:
        click.echo(f"[safety event] removed {event.chunk_id} ({event.rule}): \"{event.excerpt}\"")
    if result.rejected_answer:
        click.echo(f"[rejected generator output] {result.rejected_answer}")
    if show_context:
        for hit in result.retrieved:
            kept = "used" if hit in result.context else "REMOVED"
            click.echo(f"\n  {hit.rank}. {hit.chunk_id} score={hit.score:.3f} ({kept})\n     {hit.text}")


@cli.command("evaluate")
@click.option("--mode", type=click.Choice(["extractive", "llm"]), default="extractive",
              show_default=True)
def evaluate_cmd(mode: str) -> None:
    """Run every questions.json case through the pipeline and score it."""
    from src.evaluation import evaluate

    report = evaluate(mode)
    click.echo(f"Mode: {report['mode']} ({report['generation_model']}) | settings: {report['settings']}")
    for key, value in report["summary"].items():
        click.echo(f"  {key:<32} {value}")
    click.echo("\n  id     expect   got      ok  R@k  RR    rel   faith  reason")
    for c in report["cases"]:
        fmt = lambda v: "-" if v is None else (v if isinstance(v, str) else f"{v:.2f}")
        click.echo(
            f"  {c['id']:<6} {'abstain' if c['should_abstain'] else 'answer':<8} "
            f"{'answer' if c['answered'] else 'abstain':<8} {'Y' if c['decision_correct'] else 'N':<3} "
            f"{fmt(c['recall_at_k']):<4} {fmt(c['reciprocal_rank']):<5} {fmt(c['answer_relevance']):<5} "
            f"{fmt(c['faithfulness']):<6} {c['reason']}"
        )
    click.echo(f"\nSaved {report['output_file']}")


if __name__ == "__main__":
    cli()
