import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from sqlitesearch import TextSearchIndex


DATABASE_PATH = "faq.db"
MODEL = "gpt-5.4-mini"

INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": (
        "Search the LLM Zoomcamp FAQ database for entries "
        "matching the given query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query text to look up in the course FAQ."
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

    results = index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
    )

    return results


def calculate_gpt54mini_price(
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """
    Calculate the approximate API cost for GPT-5.4 Mini.

    This calculation uses the standard uncached token prices.
    """

    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_PRICE_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MILLION

    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def get_usage(response) -> tuple[int, int]:
    """
    Get input and output token counts from an API response.
    """

    if response.usage is None:
        return 0, 0

    return (
        response.usage.input_tokens,
        response.usage.output_tokens,
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
                "Run sqlite_ingest.py before running this file."
            )
            return

        print(f"Documents in FAQ database: {document_count}")

        openai_client = OpenAI()

        question = "I just discovered the course. Can I join it?"

        messages = [
            {
                "role": "user",
                "content": question,
            }
        ]

        print(f"\nUser question: {question}")
        print("\nSending the first request to the LLM...")

        # First API call:
        # The model decides whether it needs the search tool.
        first_response = openai_client.responses.create(
            model=MODEL,
            input=messages,
            tools=[SEARCH_TOOL],
        )

        first_input_tokens, first_output_tokens = get_usage(
            first_response
        )

        # Add everything produced by the model to the history.
        # This includes its function-call request.
        messages.extend(first_response.output)

        function_calls = [
            item
            for item in first_response.output
            if item.type == "function_call"
        ]

        if not function_calls:
            print("\nThe model did not request a tool.")
            print("\nAnswer:")
            print(first_response.output_text)

            price = calculate_gpt54mini_price(
                first_input_tokens,
                first_output_tokens,
            )

            print("\nToken usage:")
            print(f"Input tokens: {first_input_tokens}")
            print(f"Output tokens: {first_output_tokens}")
            print(f"Approximate cost: ${price['total_cost']:.8f}")
            return

        # Execute every function call requested by the model.
        for function_call in function_calls:
            print("\nThe model requested a function call.")
            print(f"Function name: {function_call.name}")

            arguments = json.loads(function_call.arguments)

            print(f"Arguments: {arguments}")

            if function_call.name == "search":
                results = search(
                    index=sqlite_index,
                    **arguments,
                )

                print(
                    f"Search returned {len(results)} documents."
                )

                result_json = json.dumps(
                    results,
                    indent=2,
                    ensure_ascii=False,
                )
            else:
                result_json = json.dumps(
                    {
                        "error": (
                            f"Unknown function: "
                            f"{function_call.name}"
                        )
                    }
                )

            # Link the result to the function call using call_id.
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": result_json,
                }
            )

        print("\nSending the search results back to the LLM...")

        # Second API call:
        # The model receives the question, its function call,
        # and the search results.
        second_response = openai_client.responses.create(
            model=MODEL,
            input=messages,
            tools=[SEARCH_TOOL],
        )

        second_input_tokens, second_output_tokens = get_usage(
            second_response
        )

        total_input_tokens = (
            first_input_tokens + second_input_tokens
        )

        total_output_tokens = (
            first_output_tokens + second_output_tokens
        )

        total_price = calculate_gpt54mini_price(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        print("\nAnswer:")
        print(second_response.output_text)

        print("\nTotal token usage for both API calls:")
        print(f"Input tokens: {total_input_tokens}")
        print(f"Output tokens: {total_output_tokens}")

        print("\nApproximate cost for both API calls:")
        print(f"Input cost: ${total_price['input_cost']:.8f}")
        print(f"Output cost: ${total_price['output_cost']:.8f}")
        print(f"Total cost: ${total_price['total_cost']:.8f}")

    finally:
        sqlite_index.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()