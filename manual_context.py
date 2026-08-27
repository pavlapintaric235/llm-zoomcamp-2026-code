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

    context = """
I just discovered the course. Can I still join?
Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions.

Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
You don't need it. You're accepted. You can also just start learning and submitting homework while the form is open without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.

What is the video/Zoom link to the stream for the Office Hours or live workshop sessions?
The Zoom link is only published to instructors, presenters, and teaching assistants. Students participate via YouTube Live and submit questions to Slido.

Cloud alternatives with GPU
Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, and Databricks.
"""

    prompt = f"""
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide an accurate
answer. If the answer is not found in the context,
respond with "I don't know."

Question:
{question}

Context:
{context}
"""

    answer = llm(prompt)

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()