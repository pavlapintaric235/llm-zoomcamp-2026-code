import requests
from minsearch import Index


def load_documents() -> list[dict]:
    """
    Download FAQ documents from all DataTalks.Club courses.
    """

    docs_url = "https://datatalks.club/faq/json/courses.json"

    response = requests.get(docs_url, timeout=30)
    response.raise_for_status()

    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"{url_prefix}{course['path']}"

        course_response = requests.get(course_url, timeout=30)
        course_response.raise_for_status()

        course_data = course_response.json()
        documents.extend(course_data)

    return documents


def build_index(documents: list[dict]) -> Index:
    """
    Create and fill the minsearch index.
    """

    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    )

    index.fit(documents)

    return index


# Download the documents and build the index.
documents = load_documents()
index = build_index(documents)


def search(
    question: str,
    course: str = "llm-zoomcamp",
) -> list[dict]:
    """
    Search the FAQ documents.

    By default, only return documents from LLM Zoomcamp.
    """

    boost_dict = {
        "question": 2.0,
        "section": 0.5,
    }

    filter_dict = {
        "course": course,
    }

    search_results = index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5,
    )

    return search_results


def main() -> None:
    question = "I just discovered the course. Can I join now?"

    print(f"Number of downloaded documents: {len(documents)}")
    print(f"\nQuestion: {question}")

    search_results = search(question)

    print("\nTop five matching FAQ questions:")

    for number, document in enumerate(search_results, start=1):
        print(f"{number}. {document['question']}")

    if search_results:
        best_result = search_results[0]

        print("\nBest matching document:")
        print(f"Course: {best_result['course']}")
        print(f"Section: {best_result['section']}")
        print(f"Question: {best_result['question']}")
        print(f"Answer: {best_result['answer']}")
    else:
        print("\nNo matching documents were found.")


if __name__ == "__main__":
    main()