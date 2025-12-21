import asyncio
from typing import Iterable

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

BASE_URL: str = "https://openrouter.ai/api/v1"
MAX_COMPLETION_TOKENS: int = 500
TEMPERATURE: float = 0.5
CONCURRENCY_LIMIT: int = 5

SYSTEM_MESSAGE: str = """
You are a code completion software LLM. When you receive an input code,
complete it to the best of your knowledge, while keeping the code correct,
readable and formatted according to best practices.
IMPORTANT: Only return the rest of the code, *not* any other explanation,
comments, tests, etc. Complete the code (possibly with code comments). Your entire ouput
result will be concatenated to the input string and it needs to be valid, runnable code.
"""


class CodeCompleter:
    def __init__(self, api_key: str, model: str, verbose_failures=False):
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
        self.model: str = model
        self.verbose_failures: bool = verbose_failures
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    def _verify_response(self, input_string: str, response) -> bool:
        if not response.choices:
            print(f"No choices return from {input_string}")
            return False

        choice = response.choices[0]

        if choice.finish_reason == "content_filter":
            print(f"Content filter triggered for {input_string}")
            return False

        if not choice.message.content:
            print(f"Empty content returned for {input_string}")
            return False

        return True

    async def complete(self, input_string: str) -> str:

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": input_string},
        ]
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
            except Exception as e:
                print(f"Got exception {e} for {input_string}")
                return ""

        if not self._verify_response(input_string, response):
            if self.verbose_failures:
                print(response)
            return ""

        return response.choices[0].message.content  # type: ignore

    async def complete_many(self, inputs: Iterable[str]) -> dict[str, str]:
        tasks = [self.complete(input_string) for input_string in inputs]

        completions = await asyncio.gather(*tasks)

        response = {}
        for input_string, completion in zip(inputs, completions):
            response[input_string] = completion

        return response
