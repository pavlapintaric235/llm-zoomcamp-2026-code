import os

from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import TextSearchIndex

from rag_helper import RAGBase


DATABASE_PATH = "faq.db"


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Make sure your .env file contains "
            "OPENAI_API_KEY=your-real-api-key"
        )

    print(f"Opening persistent index: {DATABASE_PATH}")

    sqlite_index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=DATABASE_PATH,
    )

    try:
        document_count = sqlite_index.count()

        print(f"Documents currently in the index: {document_count}")

        if document_count == 0:
            print(
                "\nThe database is empty. "
                "Run sqlite_ingest.py before running this program."
            )
            return

        question = "Can I still join the course after it started?"

        print(f"\nSearching for: {question}")

        search_results = sqlite_index.search(
            question,
            num_results=5,
        )

        print("\nRetrieved FAQ questions:")

        for number, document in enumerate(search_results, start=1):
            print(f"{number}. {document['question']}")

        openai_client = OpenAI()

        assistant = RAGBase(
            index=sqlite_index,
            llm_client=openai_client,
        )

        print("\nGenerating the RAG answer...\n")

        answer = assistant.rag(question)

        print("Answer:")
        print(answer)

    finally:
        sqlite_index.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()