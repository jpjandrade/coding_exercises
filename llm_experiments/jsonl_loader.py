import json
import os
import queue
import threading

from typing import List, Generator
from itertools import islice

# ==========================================
# SETUP: Creating dummy data for the exercise
# ==========================================
FILENAME = "training_data.jsonl"
BATCH_SIZE = 4
SEQ_LENGTH = 4


def create_dummy_data():
    data = [
        {
            "id": "101",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
        {
            "id": "102",
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
            ],
        },
        "CORRUPT_JSON_LINE_ERROR_{}",  # <--- You must handle this without crashing
        {
            "id": "103",
            "messages": [
                {"role": "user", "content": "Count to 3"},
                {"role": "assistant", "content": "1, 2, 3"},
            ],
        },
        {
            "id": "104",
            "messages": [
                {"role": "user", "content": "Color of sky?"},
                {"role": "assistant", "content": "Blue."},
            ],
        },
        {
            "id": "105",
            "messages": [
                {"role": "user", "content": "Next line?"},
                {"role": "assistant", "content": "End."},
            ],
        },
        "",
        "",
        {"id": "106", "a": "also corrupt"},
    ]

    with open(FILENAME, "w") as f:
        for entry in data:
            if isinstance(entry, dict):
                f.write(json.dumps(entry) + "\n")
            else:
                f.write(entry + "\n")


# Run setup
create_dummy_data()


# ==========================================
# HELPER: Mock Tokenizer
# ==========================================
def mock_tokenize(text: str) -> List[int]:
    """
    Simulates a tokenizer.
    Splits by space and returns arbitrary integers.
    """
    # Simple strategy: sum of ASCII values of the word for a 'token ID'
    return [sum(ord(c) for c in word) for word in text.split()]


# ==========================================
# YOUR TASK
# ==========================================


class BatchLoader:
    def __init__(self, file_path: str, batch_size: int, sequence_length: int = 16):
        self.file_path = file_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def _format_conversation(self, messages: List[dict]) -> str:
        """
        Helper to convert message list to training string.
        Format: <user> {val} \n <assistant> {val} \n <EOS>
        """

        conversation_chunks = [
            f"<{msg["role"]}> {msg["content"]} \n" for msg in messages
        ]
        conversation_chunks.append("<EOS>")
        return " ".join(conversation_chunks)

    def _parse_lines(self):
        with open(self.file_path, "r") as f:
            for line in f:
                try:
                    data: dict = json.loads(line)
                    if data.get("messages"):
                        yield data
                    else:
                        print(f"No message found in {line}, skipping...")

                except json.JSONDecodeError as e:
                    print(f"Errors parsing line {line}: {e.msg}, skipping...")
                    continue

    def _tokenized_sequences(self) -> Generator[List[int], None, None]:
        for data in self._parse_lines():
            tokens = mock_tokenize(self._format_conversation(data["messages"]))
            padding_needed = (
                self.sequence_length - len(tokens) % self.sequence_length
            ) % self.sequence_length
            tokens += [0] * padding_needed
            assert len(tokens) % self.sequence_length == 0

            for i in range(0, len(tokens), self.sequence_length):
                yield tokens[i : i + self.sequence_length]

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """
        Yields batches of tokenized data.
        Return type should be a List of lists (e.g., [[101, 20], [30, 40]])
        """
        example_iterator: Generator[List[int], None, None] = self._tokenized_sequences()
        while True:
            batch = list(islice(example_iterator, self.batch_size))
            if not batch:
                break
            yield batch


class BackgroundPrefetcher:
    def __init__(self, loader_iterator, buffer_size=3):
        self.loader_iterator = loader_iterator
        self.buffer_size = buffer_size
        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._stop_sentinel = object()  # To detect when done.
        self._worker_thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _producer(self):
        try:
            for batch in self.loader_iterator:
                if self._stop_event.is_set():
                    break
                # Blocks if the queue is full.
                self.queue.put(batch)
        except Exception as e:
            self._error = e

        finally:
            # Adds sentinel at the end
            self.queue.put(self._stop_sentinel)

    def __iter__(self):
        self._worker_thread = threading.Thread(target=self._producer, daemon=True)
        self._worker_thread.start()

        while True:
            batch = self.queue.get()
            if batch is self._stop_sentinel:
                if self._error:
                    raise self._error
                break

            yield batch

    def stop(self):
        self._stop_event.set()


# ==========================================
# EXECUTION
# ==========================================
# Test your code
loader = BatchLoader(FILENAME, batch_size=BATCH_SIZE, sequence_length=SEQ_LENGTH)
prefetcher = BackgroundPrefetcher(loader)

print("Starting Training Loop...")
for i, batch in enumerate(prefetcher):
    assert len(batch) <= BATCH_SIZE
    for example in batch:
        assert len(example) == SEQ_LENGTH
    print(f"Batch {i + 1}: {batch}")


# Cleanup
if os.path.exists(FILENAME):
    os.remove(FILENAME)
