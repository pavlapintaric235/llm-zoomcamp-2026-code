from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

openai_client = OpenAI()


def llm(prompt: str) -> str:
    response = openai_client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
    )
    return response.output_text


def main() -> None:
    question = "I just discovered the course. Can I join now?"
    answer = llm(question)
    print(answer)


if __name__ == "__main__":
    main()