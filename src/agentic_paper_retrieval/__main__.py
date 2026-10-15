"""Command-line entry point for the project scaffold."""

import argparse

from agentic_paper_retrieval import __version__


def main() -> None:
    """Expose starter help and version information."""
    parser = argparse.ArgumentParser(
        prog="paper-retriever",
        description=(
            "Assisted Agentic RAG Paper Retriever. "
            "Project scaffold; paper retrieval is not implemented yet."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
