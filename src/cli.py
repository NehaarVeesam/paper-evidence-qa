from pathlib import Path

import typer

from config import PDF_PATH

app = typer.Typer()


@app.command()
def index(
    pdf_path: Path = typer.Argument(
        PDF_PATH,
        help="Path to the PDF (default: data/paper.pdf)",
    ),
):
    from index import build_chroma_index

    n = build_chroma_index(pdf_path)
    print(f"Indexed {n} chunks into Chroma Collection")


@app.command()
def ask(question: str):
    from generate import ask_paper

    print(ask_paper(question))

if __name__ == "__main__":
    app()