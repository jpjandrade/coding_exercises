import json
from typing import Callable
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-3.5-haiku"

TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    },
]


def get_weather(location: str) -> dict:
    mock_data = {
        "Zürich": {"temp": 5, "condition": "foggy"},
        "Paris": {"temp": 12, "condition": "sunny"},
    }

    return mock_data.get(location, {"temp": 10, "condition": "unknown"})


def get_time(location: str) -> str:
    return "16:30"


TOOL_FUNCTIONS: dict[str, Callable] = {"get_weather": get_weather, "get_time": get_time}


def process_stream(
    client: OpenAI, messages: list[ChatCompletionMessageParam]
) -> tuple[str | None, list[dict]]:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        tools=TOOLS,
        max_completion_tokens=500,
    )

    content_parts = []
    tool_calls = {}
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}

                if tc_chunk.id:
                    tool_calls[idx]["id"] = tc_chunk.id
                if tc_chunk.function and tc_chunk.function.name:
                    tool_calls[idx]["name"] += tc_chunk.function.name
                if tc_chunk.function and tc_chunk.function.arguments:
                    tool_calls[idx]["arguments"] += tc_chunk.function.arguments

    content = "".join(content_parts) if content_parts else None
    return content, list(tool_calls.values())


def execute_tool_calls(tool_calls: list[dict]) -> list[ChatCompletionMessageParam]:
    results: list[ChatCompletionMessageParam] = []

    for tc in tool_calls:
        func: Callable | None = TOOL_FUNCTIONS.get(tc["name"], None)

        if not func:
            print(f" \n [Error: Unknown function {tc["name"]}. Skipping..]")
            continue

        args = json.loads(tc["arguments"])
        response = func(**args)
        print(f"\n [Executed {tc["name"]}({args}) -> {response}])")

        results.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(response)}
        )

    return results


def interact(
    input_string, client: OpenAI
) -> tuple[str, list[ChatCompletionMessageParam]]:

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You're a friendly assistant."},
        {"role": "user", "content": input_string},
    ]
    while True:
        content, tool_calls = process_stream(client, messages)
        print()  # New line.

        if not tool_calls:
            content = content or ""
            return content, messages

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )

        tool_results = execute_tool_calls(tool_calls)
        messages.extend(tool_results)


def main() -> None:
    try:
        api_key = open("open_router_token.txt", "r", encoding="utf-8").read().strip()
    except FileNotFoundError:
        print("API Key file not found!")
        return

    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    _, messages = interact(
        "What's the weather right now in Paris and Zürich and what's the time there",
        client,
    )

    print(messages)


if __name__ == "__main__":
    main()
