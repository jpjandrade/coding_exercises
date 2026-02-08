import asyncio
import time

from dataclasses import dataclass, field


@dataclass(order=True)
class Request:
    num_tokens: int
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    arrival_time: float = field(compare=False, default_factory=time.monotonic)


class DynamicBatcher:
    """
    Dynamic batcher that groups incoming requests to maximize utlization.

    Dispatch triggers:
    1. Buffer reaches max_batch_size
    2. Buffer tokens reach max_batch_tokens
    3. Oldest request has waited >= max_wait_time_ms
    """

    def __init__(
        self,
        max_batch_tokens: int = 4096,
        max_batch_size: int = 32,
        max_wait_time_ms: float = 50.0,
    ):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.max_wait_time_s = max_wait_time_ms / 1000.0

        self._buffer: list[Request] = []
        self._batch_queue: asyncio.Queue[list[dict]] = asyncio.Queue()
        self._new_request_event = asyncio.Event()
        self._closed = False

    @staticmethod
    def _token_count(prompt: str) -> int:
        """Simplified tokenizer logic."""
        return len(prompt.split())

    def submit(self, request_id: str, prompt: str) -> None:
        """
        Called by clients when a new request arrives.
        Can be async if we want to add a semaphore to buffer.
        """
        num_tokens = self._token_count(prompt)
        self._buffer.append(Request(num_tokens, request_id, prompt))
        self._new_request_event.set()  # wake up the scheduler

    async def get_next_batch(self) -> list[dict]:
        """Blocks until a batch is ready."""
        return await self._batch_queue.get()

    def _time_until_deadline(self) -> float | None:
        """Time in seconds until the oldest request hits its max wait time."""
        if not self._buffer:
            return None

        oldest = min(r.arrival_time for r in self._buffer)
        remaining = self.max_wait_time_s - (time.monotonic() - oldest)
        return max(0.0, remaining)

    def _should_dispatch(self) -> bool:
        if not self._buffer:
            return False

        # Condition 1: enough requests.
        if len(self._buffer) >= self.max_batch_size:
            return True

        # Condition 2: enough tokens.
        total_tokens = sum(r.num_tokens for r in self._buffer)
        if total_tokens >= self.max_batch_tokens:
            return True

        # Condition 3: timeout.
        oldest = min(r.arrival_time for r in self._buffer)
        if time.monotonic() - oldest >= self.max_wait_time_s:
            return True

        return False

    def _form_batch(self) -> list[dict]:
        self._buffer.sort()  # Sort by num_tokens

        batch: list[Request] = []
        remaining: list[Request] = []
        batch_max_len = 0

        for req in self._buffer:
            new_max_len = max(batch_max_len, req.num_tokens)
            new_cost = new_max_len * (len(batch) + 1)

            if len(batch) < self.max_batch_size and new_cost <= self.max_batch_tokens:
                batch.append(req)
                batch_max_len = new_max_len
            else:
                remaining.append(req)
        self._buffer = remaining

        return [
            {
                "request_id": r.request_id,
                "prompt": r.prompt,
                "num_tokens": r.num_tokens,
                "wait_time_ms": (time.monotonic() - r.arrival_time) * 1000,
            }
            for r in batch
        ]

    async def run_scheduler(self) -> None:
        """
        Main loop, running as a background task.
        """
        while not self._closed:
            timeout = self._time_until_deadline()
            try:
                # Wait for new request, up until the timeout.
                self._new_request_event.clear()
                await asyncio.wait_for(self._new_request_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass  # Stop when deadline exceeded.

            if self._should_dispatch():
                batch = self._form_batch()
                if batch:
                    await self._batch_queue.put(batch)

    async def shutdown(self) -> None:
        """Flush the buffer and stop"""
        self._closed = True
        if self._buffer:
            batch = self._form_batch()
            if batch:
                await self._batch_queue.put(batch)


# ─── Usage ───────────────────────────────────────────────────────────────────


async def main():
    batcher = DynamicBatcher(
        max_batch_tokens=16,
        max_batch_size=4,
        max_wait_time_ms=200,
    )

    scheduler_task = asyncio.create_task(batcher.run_scheduler())

    # Simulate async request arrival
    prompts = [
        ("r1", "the quick brown fox"),  # 4 tokens
        ("r2", "jumps over the lazy dog by the river on a sunny day"),  # 12
        ("r3", "hello world"),  # 2
        ("r4", "short"),  # 1
        ("r5", "a slightly longer prompt with several words in it"),  # 9
        ("r6", "medium length prompt here"),  # 4
        (
            "r7",
            "long length long length long length long length long length long length long length long length",
        ),  # 14
    ]

    for rid, prompt in prompts:
        batcher.submit(rid, prompt)
        await asyncio.sleep(0.02)  # 20ms between requests

    # Consume batches
    while True:  # expect a few batches
        try:
            batch = await asyncio.wait_for(batcher.get_next_batch(), timeout=1.0)
            print(f"\n📦 Batch ({len(batch)} items):")
            for item in batch:
                print(
                    f"  {item['request_id']:>3}: {item['num_tokens']:>3} tokens, "
                    f"waited {item['wait_time_ms']:.1f}ms — \"{item['prompt']}\""
                )
        except asyncio.TimeoutError:
            break

    await batcher.shutdown()
    scheduler_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
