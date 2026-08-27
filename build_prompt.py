import requests
from minsearch import Index


INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""


USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


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
    Create the minsearch index.
    """

    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    )

    index.fit(documents)

    return index


# Download the FAQ FAQ documents and build the search index.
documents = load_documents()
index = build_index(documents)


def search(
    question: str,
    course: str = "llm-zoomcamp",
) -> list[dict]:
    """
    Return the five most relevant FAQ documents.
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


def build_context(search_results: list[dict]) -> str:
    """
    Turn the list of search-result dictionaries into one string.
    """

    lines = []

    for document in search_results:
        lines.append(document["section"])
        lines.append("Q: " + document["question"])
        lines.append("A: " + document["answer"])
        lines.append("")

    context = "\n".join(lines).strip()

    return context


def build_prompt(
    question: str,
    search_results: list[dict],
) -> str:
    """
    Combine the user's question and the retrieved context.
    """

    context = build_context(search_results)

    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    return prompt.strip()


def main() -> None:
    question = "I just discovered the course. Can I join now?"

    # Step 1: Retrieve the relevant FAQ documents.
    search_results = search(question)

    # Step 2: Turn the results into a prompt.
    prompt = build_prompt(question, search_results)

    print("INSTRUCTIONS:")
    print(INSTRUCTIONS.strip())

    print("\n" + "=" * 70)

    print("\nUSER PROMPT:")
    print(prompt)


if __name__ == "__main__":
    main()