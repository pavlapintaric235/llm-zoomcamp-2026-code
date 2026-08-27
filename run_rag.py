import os

from dotenv import load_dotenv
from openai import OpenAI

from ingest import build_index, load_faq_data
from rag_helper import RAGBase


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Make sure your .env file contains "
            "OPENAI_API_KEY=your-real-api-key"
        )

    print("Downloading FAQ documents...")

    documents = load_faq_data()

    print(f"Downloaded {len(documents)} documents.")
    print("Building the search index...")

    index = build_index(documents)
    openai_client = OpenAI()

    assistant = RAGBase(
        index=index,
        llm_client=openai_client,
    )

    question = "I just discovered the course. Can I join now?"

    print(f"\nQuestion: {question}")
    print("\nGenerating answer...\n")

    answer = assistant.rag(question)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()