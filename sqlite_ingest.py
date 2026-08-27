import time

from sqlitesearch import TextSearchIndex

from ingest import load_faq_data


DATABASE_PATH = "faq.db"


def main() -> None:
    print("Downloading FAQ documents...")

    documents = load_faq_data()

    print(f"Loaded {len(documents)} total documents.")

    docs_llm = [
        document
        for document in documents
        if document["course"] == "llm-zoomcamp"
    ]

    print(f"Found {len(docs_llm)} LLM Zoomcamp documents.")
    print(f"Creating persistent index: {DATABASE_PATH}\n")

    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=DATABASE_PATH,
    )

    try:
        for number, document in enumerate(docs_llm, start=1):
            index.add(document)

            shortened_question = document["question"][:60]

            print(
                f"Added {number}/{len(docs_llm)}: "
                f"{shortened_question}..."
            )

            # Artificial delay used by the course to simulate slow ingestion.
            time.sleep(0.5)

    finally:
        index.close()

    print(f"\nDone. Index saved to {DATABASE_PATH}")


if __name__ == "__main__":
    main()
    