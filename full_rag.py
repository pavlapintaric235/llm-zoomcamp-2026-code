import os

import requests
from dotenv import load_dotenv
from minsearch import Index
from openai import OpenAI


# Load OPENAI_API_KEY from the .env file.
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        "Create a .env file containing OPENAI_API_KEY=your-key"
    )

openai_client = OpenAI()


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
    Download all FAQ documents from DataTalks.Club.
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
    Build a searchable minsearch index.
    """

    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
    )

    index.fit(documents)

    return index


# Load the documents and create the index when the program starts.
documents = load_documents()
index = build_index(documents)


def search(
    question: str,
    course: str = "llm-zoomcamp",
) -> list[dict]:
    """
    Find the five most relevant FAQ documents.
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
    Convert the retrieved documents into one context string.
    """

    lines = []

    for document in search_results:
        lines.append(document["section"])
        lines.append("Q: " + document["question"])
        lines.append("A: " + document["answer"])
        lines.append("")

    return "\n".join(lines).strip()


def build_prompt(
    question: str,
    search_results: list[dict],
) -> str:
    """
    Combine the question and retrieved context.
    """

    context = build_context(search_results)

    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    return prompt.strip()


def llm(
    instructions: str,
    user_prompt: str,
    model: str = "gpt-5.4-mini",
) -> str:
    """
    Send the instructions and user prompt to OpenAI.
    """

    message_history = [
        {
            "role": "developer",
            "content": instructions,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = openai_client.responses.create(
        model=model,
        input=message_history,
    )

    return response.output_text


def rag(
    query: str,
    model: str = "gpt-5.4-mini",
) -> str:
    """
    Run the complete RAG pipeline.
    """

    search_results = search(query)

    prompt = build_prompt(
        question=query,
        search_results=search_results,
    )

    answer = llm(
        instructions=INSTRUCTIONS,
        user_prompt=prompt,
        model=model,
    )

    return answer


def main() -> None:
    question = "I just discovered the course. Can I join now?"

    print(f"Question: {question}")
    print("\nSearching the FAQ and generating an answer...\n")

    answer = rag(question)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()