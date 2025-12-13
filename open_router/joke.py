import asyncio
from openai import AsyncOpenAI

MODELS = {
    "chatgpt": "openai/gpt-5.2-chat",
    "gemini": "google/gemini-3-pro-preview",
    "mistal": "mistralai/devstral-2512:free",
    "claude": "anthropic/claude-sonnet-4.5",
}


async def get_joke(client: AsyncOpenAI, model_url: str) -> str:
    response = await client.chat.completions.create(
        model=model_url,
        extra_body={
            "reasoning": {
                "effort": "low",
            }
        },
        messages=[
            {
                "role": "system",
                "content": "You are a funny comedian and always strives to make very varied jokes. Your jokes should be original and not repeat any previous jokes.",
            },
            {"role": "user", "content": "Tell me a one liner joke."},
        ],
        temperature=1.0,
        max_tokens=200,
    )

    if not response.choices or not response.choices[0].message.content:
        print("Could not generate joke for ", model_url, ", received: ", response)
    return response.choices[0].message.content


async def main():
    api_key = open("open_router_token.txt", "r", encoding="utf-8").read().strip()

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    tasks = []
    for model_name, model_url in MODELS.items():
        tasks.append(asyncio.create_task(get_joke(client, model_url)))

    jokes = await asyncio.gather(*tasks)

    for model_name, joke in zip(MODELS.keys(), jokes):
        print(f"Joke from {model_name}:\n{joke}\n")


if __name__ == "__main__":
    asyncio.run(main())
