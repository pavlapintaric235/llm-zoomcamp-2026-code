INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""


PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        course: str = "llm-zoomcamp",
        model: str = "gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> list[dict]:
        """
        Retrieve the most relevant FAQ documents.
        """

        boost_dict = {
            "question": 3.0,
            "section": 0.5,
        }

        filter_dict = {
            "course": self.course,
        }

        search_results = self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

        return search_results

    def build_context(
        self,
        search_results: list[dict],
    ) -> str:
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
        self,
        query: str,
        search_results: list[dict],
    ) -> str:
        """
        Combine the user's question and retrieved context.
        """

        context = self.build_context(search_results)

        prompt = self.prompt_template.format(
            question=query,
            context=context,
        )

        return prompt

    def llm(self, prompt: str) -> str:
        """
        Send the prompt to the configured LLM.
        """

        input_messages = [
            {
                "role": "developer",
                "content": self.instructions,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

        return response.output_text

    def rag(self, query: str) -> str:
        """
        Run the complete RAG pipeline.
        """

        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        return answer