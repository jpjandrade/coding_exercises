import argparse
import asyncio
import random

from code_completion_eval import eval_results
from code_completion import CodeCompleter

MODEL = "google/gemini-3-flash-preview"
MIN_SPLIT_PADDING = (
    10  # Minimum amount of chars to leave as input string / string to be completed.
)

corpus_simple = [
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def sum_list(items):\n    total = 0\n    for x in items:\n        total += x\n    return total",
    # Prime Check
    """def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
    # Reverse String
    """def reverse_string(s):
    return s[::-1]""",
    # FizzBuzz
    """def fizz_buzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result""",
    # Palindrome Check
    """def is_palindrome(s):
    clean_s = ''.join(c.lower() for c in s if c.isalnum())
    return clean_s == clean_s[::-1]""",
    # Convert Celsius to Fahrenheit
    """def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32""",
]

corpus_medium = [
    # Binary Search
    """def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",
    # Valid Parentheses (Stack)
    """def is_valid_parentheses(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack""",
    # Two Sum (Hashmap)
    """def two_sum(nums, target):
    prev_map = {}  # val : index
    for i, n in enumerate(nums):
        diff = target - n
        if diff in prev_map:
            return [prev_map[diff], i]
        prev_map[n] = i
    return []""",
    # Merge Sort (Divide and Conquer)
    """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result""",
]

corpus_complex = [
    # Simple LRU Cache implementation
    """class ListNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {} # Map key to node
        # Dummy head and tail
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add(self, node):
        prev_node = self.tail.prev
        prev_node.next = node
        self.tail.prev = node
        node.prev = prev_node
        node.next = self.tail

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        
        node = ListNode(key, value)
        self._add(node)
        self.cache[key] = node
        
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]""",
    # Simple Event/Task Manager (Observer Pattern-ish)
    """import datetime

class Task:
    def __init__(self, title, description, due_date):
        self.title = title
        self.description = description
        self.due_date = due_date
        self.completed = False
        self.created_at = datetime.datetime.now()

    def mark_complete(self):
        self.completed = True

    def __repr__(self):
        status = "[x]" if self.completed else "[ ]"
        return f"{status} {self.title} (Due: {self.due_date})"

class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, title, description, due_date):
        new_task = Task(title, description, due_date)
        self.tasks.append(new_task)
        return new_task

    def get_pending_tasks(self):
        return [t for t in self.tasks if not t.completed]

    def get_overdue_tasks(self):
        now = datetime.datetime.now()
        # Assuming due_date is comparable
        return [t for t in self.tasks if not t.completed and t.due_date < now]

    def bulk_complete(self, titles):
        for task in self.tasks:
            if task.title in titles:
                task.mark_complete()""",
    # Basic Trie (Prefix Tree) implementation
    """class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True""",
]


def split_at_random(input_text: str) -> tuple[str, str]:
    # Guard for when the text is too small.
    split_padding = min(MIN_SPLIT_PADDING, len(input_text) // 2)

    i = random.randint(split_padding, len(input_text) - split_padding)
    return input_text[:i], input_text[i:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit the number of corpus entries to process (default: process all)"
    )
    args = parser.parse_args()

    try:
        api_key = open("open_router_token.txt", "r", encoding="utf-8").read().strip()
    except FileNotFoundError:
        print("API Key file not found!")
        return

    cc = CodeCompleter(api_key, MODEL, verbose_failures=True)

    # TODO: split evals into complexity.
    corpus = corpus_simple + corpus_medium + corpus_complex

    if args.limit is not None:
        corpus = corpus[:args.limit]
        print(f"Processing {len(corpus)} entries (limited by --limit flag)")
    else:
        print(f"Processing all {len(corpus)} entries")

    split_corpus: dict[str, str] = {
        k: v for k, v in (split_at_random(entry) for entry in corpus)
    }
    print("Generating completions...")
    results = asyncio.run(cc.complete_many(split_corpus.keys()))
    print("Done!")

    evaluation_results = eval_results(split_corpus, results)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80 + "\n")

    for result in evaluation_results:
        print(f"Input: {result['input'][:60]}...")
        print(f"Generated: {result['output'][:60]}...")
        print(f"Original:  {result['original_completion'][:60]}...")
        print(f"Scores --> Parser: {result['parser_score']}, "
              f"Linter: {result['linter_score']:.2f}, "
              f"BLEU: {result['bleu_score']:.3f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
