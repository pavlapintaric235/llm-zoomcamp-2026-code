import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import TextSearchIndex


DATABASE_PATH = "faq.db"
MODEL = "gpt-5.4-mini"
MAX_ITERATIONS = 5


INSTRUCTIONS = """
You're a course teaching assistant.
You're given a question from a course student, and your task is
to answer it.

Use the search function to find information in the course FAQ.

For the first search, use as many useful keywords from the user's
question as possible.

Make multiple searches when necessary. Analyze the results from
each search and perform additional searches using new keywords.

If a search returns poor results, consider whether the user's
question contains a spelling mistake. Try searching again with
corrected or alternative terms.

The question must be about the course or its logistics.
Do not answer off-topic questions.

If you cannot answer using the FAQ database, say that you don't know.
Do not answer using your own general knowledge.

At the end, ask whether the user wants to explore another
course-related area.
""".strip()


SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": (
        "Search the LLM Zoomcamp FAQ database for information "
        "needed to answer a student's course-related question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keywords or a question to search for in "
                    "the course FAQ database."
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


def search(
    index: TextSearchIndex,
    query: str,
) -> list[dict]:
    """
    Search the LLM Zoomcamp FAQ database.
    """

    boost_dict = {
        "question": 3.0,
        "section": 0.5,
    }

    filter_dict = {
        "course": "llm-zoomcamp",
    }

    search_results = index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
    )

    return search_results


def make_call(
    call,
    index: TextSearchIndex,
) -> dict:
    """
    Execute one function call requested by the model.

    Return the result in the format expected by
    the Responses API.
    """

    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError as error:
        result = {
            "error": f"Invalid JSON arguments: {error}",
        }

    else:
        if call.name == "search":
            result = search(
                index=index,
                **arguments,
            )
        else:
            result = {
                "error": f"Unknown function: {call.name}",
            }

    result_json = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }


def agent_loop(
    openai_client: OpenAI,
    index: TextSearchIndex,
    instructions: str,
    question: str,
    model: str = MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """
    Run the LLM and its tools until it returns a final answer.
    """

    messages = [
        {
            "role": "developer",
            "content": instructions,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration #{iteration}")

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[SEARCH_TOOL],
        )

        if response.usage is not None:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

        # Preserve every output item in the conversation history.
        # This includes messages, reasoning items and function calls.
        messages.extend(response.output)

        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        # Display any text the model produced during this iteration.
        if response.output_text:
            print("\nASSISTANT:")
            print(response.output_text)

        # No function calls means the model has finished.
        if not function_calls:
            final_answer = response.output_text

            if not final_answer:
                final_answer = (
                    "The agent stopped without producing "
                    "a text answer."
                )

            print("\nAgent finished.")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")

            return final_answer

        # Execute every tool call requested during this iteration.
        for function_call in function_calls:
            print("\nFUNCTION CALL:")
            print(f"Name: {function_call.name}")
            print(f"Arguments: {function_call.arguments}")

            call_output = make_call(
                call=function_call,
                index=index,
            )

            messages.append(call_output)

            tool_result = json.loads(call_output["output"])

            if isinstance(tool_result, list):
                print(
                    f"Search returned "
                    f"{len(tool_result)} documents."
                )
            else:
                print(f"Tool result: {tool_result}")

    raise RuntimeError(
        f"The agent did not finish after "
        f"{max_iterations} iterations. "
        "The safety limit stopped the loop."
    )


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Make sure your .env file contains "
            "OPENAI_API_KEY=your-real-api-key"
        )

    sqlite_index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=DATABASE_PATH,
    )

    try:
        document_count = sqlite_index.count()

        if document_count == 0:
            print(
                "The FAQ database is empty. "
                "Run sqlite_ingest.py first."
            )
            return

        print(f"Documents in FAQ database: {document_count}")

        openai_client = OpenAI()

        question = "How do I run Olama locally?"

        print(f"\nUSER:")
        print(question)

        final_answer = agent_loop(
            openai_client=openai_client,
            index=sqlite_index,
            instructions=INSTRUCTIONS,
            question=question,
        )

        print("\nFINAL ANSWER:")
        print(final_answer)

    finally:
        sqlite_index.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()