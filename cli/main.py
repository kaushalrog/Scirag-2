"""
cli/main.py
------------
SciRAG-UQ command-line interface.

Usage
-----
  scirag ingest --query "RAG language models" --max 50
  scirag ask "What is Retrieval-Augmented Generation?"
  scirag evaluate --benchmark ./data/benchmarks/bench.json
  scirag stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="scirag", help="SciRAG-UQ — Uncertainty-Aware Scientific RAG")
console = Console()


def _load_components():
    """Lazily import and initialise all components."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import get_settings
    from src.embeddings.chunker import RecursiveChunker
    from src.embeddings.embedder import Embedder
    from src.generation.groq_client import GroqClient
    from src.generation.rag_chain import RAGChain
    from src.ingestion.pipeline import IngestionPipeline
    from src.retrieval.retriever import HybridRetriever
    from src.uncertainty.abstention import CascadePolicy
    from src.uncertainty.estimator import UncertaintyEstimator
    from src.vectorstore.chroma_manager import ChromaManager

    s = get_settings()
    embedder = Embedder(model_name=s.embedding_model)
    chroma = ChromaManager(persist_dir=s.chroma_persist_dir, collection=s.chroma_collection)
    retriever = HybridRetriever(chroma=chroma, embedder=embedder,
                                dense_weight=s.dense_weight, mmr_lambda=s.mmr_lambda)
    llm = GroqClient(api_key=s.groq_api_key, model=s.llm_model)
    uq = UncertaintyEstimator(embedder=embedder)
    policy = CascadePolicy()
    chain = RAGChain(retriever=retriever, llm=llm, uncertainty=uq,
                     abstention=policy, top_k=s.top_k)
    pipeline = IngestionPipeline(download_dir=s.data_raw_dir,
                                 processed_dir=s.data_processed_dir,
                                 s2_api_key=s.s2_api_key)
    chunker = RecursiveChunker(chunk_size=512, overlap=64)
    return dict(s=s, embedder=embedder, chroma=chroma, chain=chain,
                pipeline=pipeline, chunker=chunker)


@app.command()
def ingest(
    query: str = typer.Option(..., "--query", "-q", help="arXiv search query"),
    max_papers: int = typer.Option(20, "--max", "-n", help="Max papers to fetch"),
    download_pdfs: bool = typer.Option(False, "--pdfs", help="Download full PDFs"),
    no_s2: bool = typer.Option(False, "--no-s2", help="Skip Semantic Scholar enrichment"),
):
    """Ingest papers from arXiv into the vector store."""
    c = _load_components()
    with console.status(f"Ingesting up to {max_papers} papers for: [bold]{query}[/bold]"):
        docs = c["pipeline"].run(
            query=query, max_papers=max_papers,
            download_pdfs=download_pdfs, enrich_s2=not no_s2,
        )
    ingested = 0
    for doc in docs:
        try:
            chunks = c["chunker"].split(
                text=doc.full_text or doc.abstract,
                source_id=doc.arxiv_id, title=doc.title,
                extra_metadata=doc.to_metadata(),
            )
            embs = c["embedder"].embed([ch.text for ch in chunks])
            c["chroma"].upsert_chunks(chunks, embs)
            ingested += 1
        except Exception as exc:
            console.print(f"[red]Failed {doc.arxiv_id}: {exc}[/red]")
    console.print(f"[green]✓ Ingested {ingested}/{len(docs)} papers[/green]")
    console.print(f"[blue]Total corpus: {c['chroma'].count()} chunks[/blue]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the corpus"),
    cot: bool = typer.Option(False, "--cot", help="Use chain-of-thought"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Ask a question about the ingested corpus."""
    c = _load_components()
    c["chain"].use_cot = cot

    with console.status("Retrieving and generating…"):
        resp = c["chain"].query(question)

    if json_out:
        print(json.dumps(resp.to_dict(), indent=2))
        return

    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red", "VERY LOW": "magenta"}
    from src.uncertainty.estimator import UncertaintyEstimator
    label = UncertaintyEstimator.confidence_to_label(resp.confidence)
    color = conf_color.get(label, "white")

    console.rule("[bold blue]SciRAG-UQ Answer[/bold blue]")
    console.print(f"\n[bold]Q:[/bold] {question}\n")
    if resp.abstained:
        console.print(f"[yellow]⚠ ABSTAINED:[/yellow] {resp.abstention_reason}\n")
    console.print(f"[bold]A:[/bold] {resp.answer}\n")
    console.print(f"Confidence: [{color}]{label} ({resp.confidence:.1%})[/{color}]")

    if resp.sources:
        table = Table(title="Sources", show_header=True)
        table.add_column("Title", max_width=50)
        table.add_column("arXiv ID", max_width=15)
        table.add_column("Score")
        for src in resp.sources:
            table.add_row(src["title"][:48], src.get("arxiv_id",""), f"{src['score']:.3f}")
        console.print(table)


@app.command()
def stats():
    """Show corpus statistics."""
    c = _load_components()
    sources = c["chroma"].list_sources()
    console.print(f"\n[bold]Corpus Statistics[/bold]")
    console.print(f"  Total chunks : {c['chroma'].count():,}")
    console.print(f"  Unique sources: {len(sources)}")
    console.print(f"\n  Sources:")
    for src in sources[:20]:
        console.print(f"    • {src}")
    if len(sources) > 20:
        console.print(f"    … and {len(sources)-20} more")


@app.command()
def evaluate(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="Path to benchmark JSON"),
    run_name: str = typer.Option("eval", "--name", help="Run identifier"),
    output_dir: str = typer.Option("./results", "--output", "-o"),
):
    """Run evaluation on a benchmark file."""
    from src.evaluation.benchmark import BenchmarkBuilder
    from src.evaluation.metrics import Evaluator
    from src.evaluation.runner import EvaluationRunner

    c = _load_components()
    builder = BenchmarkBuilder()
    items = builder.load(benchmark)
    console.print(f"[blue]Loaded {len(items)} benchmark items[/blue]")

    evaluator = Evaluator(embedder=c["embedder"])
    runner = EvaluationRunner(c["chain"], evaluator, output_dir=output_dir)
    report = runner.run(items, run_name=run_name)
    console.print(f"\n[green]✓ Evaluation complete. Results saved to {output_dir}/[/green]")


if __name__ == "__main__":
    app()
