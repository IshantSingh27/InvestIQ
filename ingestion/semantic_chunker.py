from pathlib import Path
import os

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv


# Load .env
load_dotenv()


def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.

    Args:
        markdown_file: Markdown file path.

    Returns:
        Markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.

    Args:
        markdown_file: Markdown file path.
        embeddings: Azure OpenAI embedding model.

    Returns:
        List of semantic chunks.
    """

    markdown_content = read_markdown(markdown_file)

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile"
    )

    return splitter.create_documents([markdown_content])


if __name__ == "__main__":

    # --------------------------------------------------
    # Azure OpenAI configuration
    # --------------------------------------------------

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    api_version = os.getenv(
        "AZURE_OPENAI_API_EMBEDDING_VERSION",
        "2023-05-15"
    )

    embedding_model = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )


    # --------------------------------------------------
    # Check configuration
    # --------------------------------------------------

    if not endpoint or not api_key or not embedding_model:
        raise RuntimeError(
            "Missing Azure OpenAI configuration.\n"
            "Check your .env file for:\n"
            "AZURE_OPENAI_ENDPOINT\n"
            "AZURE_OPENAI_API_KEY\n"
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )


    # --------------------------------------------------
    # Create embedding model
    # --------------------------------------------------

    embeddings = AzureOpenAIEmbeddings(
        model=embedding_model,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )


    # --------------------------------------------------
    # Test embedding with small Markdown file
    # --------------------------------------------------

    markdown_file = "../data/markdown/embedding_test_1_page.md"

    print("Reading Markdown file...")
    print(f"File: {markdown_file}\n")


    # --------------------------------------------------
    # Create semantic chunks
    # --------------------------------------------------

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks):

        print("=" * 80)
        print(f"Chunk {index + 1}")
        print("=" * 80)

        print(chunk.page_content[:1000])
        print()