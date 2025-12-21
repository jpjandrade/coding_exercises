import asyncio
from openai import AsyncOpenAI

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-3-flash-preview"

MAX_EXAMPLES = 2  # Number of examples to generate explanations for.

CONCURRENCY_LIMIT = 5
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

eval_dataset_explanation = [
    {
        "id": "101",
        "input_code": "print(lambda x: x*2 for x in range(5))",
        "ground_truth": "This code creates a generator object that doubles numbers from 0 to 4. However, it will print the generator object representation (e.g., <generator object...>), not the actual values, because the generator hasn't been iterated over or converted to a list.",
    },
    {
        "id": "102",
        "input_code": "df.groupby('category')['value'].transform(lambda x: x.fillna(x.mean()))",
        "ground_truth": "This Pandas code fills missing values (NaNs) in the 'value' column. It does this intelligently by grouping the data by 'category' first, calculating the mean for *that specific category*, and using that mean to fill the holes in that group.",
    },
    {
        "id": "103",
        "input_code": "while True: fork()",
        "ground_truth": "This is a 'fork bomb'. It creates an infinite loop where the process continually replicates itself (forks), rapidly exhausting system resources (process table entries) and likely causing the system to crash or freeze.",
    },
    {
        "id": "104",  # Language Quirk (Mutable Default Arguments)
        "input_code": "def add_item(item, box=[]):\n    box.append(item)\n    return box",
        "ground_truth": "This code defines a function with a mutable default argument (`box=[]`). This is a common Python trap: the list `box` is created once when the function is defined, not every time it's called. As a result, subsequent calls without a list argument will append items to the *same* persistent list rather than starting with a new empty one.",
    },
    {
        "id": "105",  # Security (SQL Injection)
        "input_code": "cursor.execute(f\"SELECT * FROM users WHERE username = '{user_input}'\")",
        "ground_truth": "This code executes a SQL query using an f-string to insert user input directly. This is a severe security vulnerability known as SQL Injection. A malicious user could craft `user_input` to manipulate the query (e.g., `' OR '1'='1`) to bypass authentication or delete data. It should use parameterized queries instead.",
    },
    {
        "id": "106",  # Regular Expressions
        "input_code": "import re\npattern = re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b')",
        "ground_truth": "This regex pattern is designed to validate email addresses. It checks for a sequence of alphanumeric characters (including dots and underscores), followed by an '@' symbol, a domain name, a dot, and a top-level domain (like .com or .org) of at least 2 characters.",
    },
    {
        "id": "107",  # Algorithmic (Memoization)
        "input_code": "memo = {}\ndef fib(n):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1) + fib(n-2)\n    return memo[n]",
        "ground_truth": "This is an implementation of the Fibonacci sequence using Dynamic Programming (Memoization). It stores previously calculated results in a dictionary (`memo`) to avoid redundant calculations, changing the time complexity from exponential O(2^n) to linear O(n).",
    },
    {
        "id": "108",  # JavaScript / React (Frontend Context)
        "input_code": "useEffect(() => {\n  const id = setInterval(tick, 1000);\n  return () => clearInterval(id);\n}, []);",
        "ground_truth": "This is a React `useEffect` hook. It sets up a timer (`setInterval`) when the component mounts. Crucially, it returns a cleanup function that clears the interval when the component unmounts to prevent memory leaks. The empty dependency array `[]` ensures this runs only once.",
    },
    {
        "id": "109",  # Bitwise Logic (Optimization)
        "input_code": "def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0",
        "ground_truth": "This function checks if a number is a power of two using bitwise operations. If `n` is a power of two, its binary representation has exactly one '1' bit. Subtracting 1 flips all bits up to that '1'. The bitwise AND (`&`) results in 0 only for powers of two.",
    },
    {
        "id": "110",  # Pandas (Data Science)
        "input_code": "df.loc[df['grade'] < 50, 'status'] = 'fail'",
        "ground_truth": "This Pandas code performs a conditional assignment. It looks for rows where the 'grade' column is less than 50, and for those specific rows, it sets the value of the 'status' column to 'fail'. It uses `.loc` for proper label-based indexing.",
    },
    {
        "id": "111",  # Shell/Bash (DevOps)
        "input_code": "chmod 755 script.sh",
        "ground_truth": "This command changes the file permissions of `script.sh`. The code `755` means the Owner has Read/Write/Execute permissions (7), while the Group and Others have only Read/Execute permissions (5).",
    },
    {
        "id": "112",  # Python Comprehension (Brevity)
        "input_code": "flattened = [item for sublist in matrix for item in sublist]",
        "ground_truth": "This is a nested list comprehension used to flatten a 2D list (a matrix) into a 1D list. It iterates through every `sublist` in `matrix`, and then through every `item` in that `sublist`, collecting them into a single list.",
    },
    {
        "id": "113",  # Dangerous/Malicious (System Deletion)
        "input_code": "import os\nos.system('rm -rf /')",
        "ground_truth": "This is a highly dangerous command that attempts to delete the root directory and everything inside it on a Linux/Unix system. `rm` is remove, `-r` is recursive, and `-f` is force. Running this would wipe the operating system.",
    },
]

PROMPT = """
"You are a code explainer, you will receive as user input a snippet of code 
and it's your job to explain what it does in two or three sentences, more if the code is very complex."
"""


async def generate_explanation(client: AsyncOpenAI, input_code) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": PROMPT,
                    },
                    {"role": "user", "content": input_code},
                ],
                temperature=1.0,
                reasoning_effort="low",
                max_completion_tokens=500,
            )
        except Exception as e:
            print(f"Error generating explanation: {e}")

    if not response.choices or not response.choices[0].message.content:
        raise ValueError(f"No content in response: {response}")

    return response.choices[0].message.content


async def main():
    try:
        api_key = open("open_router_token.txt", "r", encoding="utf-8").read().strip()
    except FileNotFoundError:
        print("API Key file not found!")
        return

    async with AsyncOpenAI(base_url=BASE_URL, api_key=api_key) as client:
        tasks = {
            item["id"]: generate_explanation(client, item["input_code"])
            for item in eval_dataset_explanation[:MAX_EXAMPLES]
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        generated_explanations = {}
        for task_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                print(f"Error for id: {task_id}: {result}")
            else:
                generated_explanations[task_id] = result
                print(f"ID {task_id}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
