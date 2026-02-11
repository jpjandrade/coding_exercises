import sys
import time
import threading


class TTLCache:
    def __init__(self, default_ttl=60):
        self.default_ttl = default_ttl
        self._cache = {}
        self._lock = threading.RLock()

        self._cleaner_thread = threading.Thread(target=self._clean, daemon=True)
        self._cleaner_stop_event: threading.Event = threading.Event()
        self._cleaner_thread.start()

    def _clean(self):
        while not self._cleaner_stop_event.wait(timeout=4):
            current_time = time.time()

            # Phase 1: Quick snapshot (lock held briefly)
            with self._lock:
                items_snapshot = list(self._cache.items())

            # Phase 2: Process snapshot (NO lock - other threads can proceed)
            expired_keys = [
                k for k, (_, expiry) in items_snapshot if expiry < current_time
            ]

            # Phase 3: Delete expired keys (lock held briefly)
            if expired_keys:
                with self._lock:
                    for k in expired_keys:
                        # Double-check: key might have been updated/deleted
                        if k in self._cache:
                            _, expiry = self._cache[k]
                            if expiry < time.time():
                                del self._cache[k]

    def put(self, key, value, ttl):
        expiry = time.time() + ttl
        with self._lock:
            self._cache[key] = (value, expiry)

    def get(self, key):

        with self._lock:
            if key not in self._cache:
                return None

            val, expiry = self._cache[key]
            if expiry < time.time():
                del self._cache[key]
                return None

            return val

    def delete(self, key):
        with self._lock:
            if key not in self._cache:
                return False
            del self._cache[key]

        return True

    def size(self):
        with self._lock:
            return len(self._cache)

    def stop(self):
        self._cleaner_stop_event.set()
        self._cleaner_thread.join()


def run_tests():
    """Comprehensive test suite for TTLCache"""

    print("=" * 60)
    print("TTLCache Test Suite")
    print("=" * 60)

    cache = TTLCache(default_ttl=10)
    passed = 0
    failed = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, failed
        if condition:
            print(f"✓ {name}")
            passed += 1
        else:
            print(f"✗ {name} - {details}")
            failed += 1

    # ========================================
    # Test 1: Basic Put/Get
    # ========================================
    print("\n[Test 1: Basic Operations]")
    cache.put("key1", "value1", ttl=10)
    assert_test("Put and Get", cache.get("key1") == "value1")
    assert_test("Size after put", cache.size() == 1)

    # ========================================
    # Test 2: Get Non-Existent Key
    # ========================================
    print("\n[Test 2: Non-Existent Keys]")
    assert_test("Get missing key returns None", cache.get("missing") is None)

    # ========================================
    # Test 3: Delete Operations
    # ========================================
    print("\n[Test 3: Delete Operations]")
    cache.put("to_delete", "temp", ttl=10)
    assert_test("Delete existing key", cache.delete("to_delete") == True)
    assert_test("Get deleted key returns None", cache.get("to_delete") is None)
    assert_test("Delete non-existent key", cache.delete("missing") == False)

    # ========================================
    # Test 4: TTL Expiration (Lazy)
    # ========================================
    print("\n[Test 4: TTL Expiration - Lazy Deletion]")
    cache.put("expires_fast", "temporary", ttl=1)
    assert_test("Get before expiry", cache.get("expires_fast") == "temporary")
    time.sleep(1.2)
    assert_test("Get after expiry returns None", cache.get("expires_fast") is None)

    # ========================================
    # Test 5: Update Existing Key
    # ========================================
    print("\n[Test 5: Update Existing Key]")
    cache.put("update_me", "old_value", ttl=10)
    cache.put("update_me", "new_value", ttl=10)
    assert_test("Updated value retrieved", cache.get("update_me") == "new_value")

    # ========================================
    # Test 6: Background Cleanup
    # ========================================
    print("\n[Test 6: Background Cleanup Process]")
    initial_size = cache.size()
    # Add keys that expire quickly
    for i in range(5):
        cache.put(f"cleanup_test_{i}", f"value_{i}", ttl=1)

    assert_test("Size increased after puts", cache.size() == initial_size + 5)

    print("  Waiting 5s for background cleanup...")
    time.sleep(5)  # Wait for expiry + cleanup cycle

    assert_test(
        "Background cleanup removed expired keys",
        cache.size() == initial_size,
        f"Expected {initial_size}, got {cache.size()}",
    )

    # ========================================
    # Test 7: Concurrent Writes
    # ========================================
    print("\n[Test 7: Concurrent Writes]")

    def concurrent_writer(cache, thread_id, num_writes):
        for i in range(num_writes):
            cache.put(f"thread_{thread_id}_key_{i}", f"value_{i}", ttl=10)

    threads = []
    num_threads = 10
    writes_per_thread = 20

    for i in range(num_threads):
        t = threading.Thread(
            target=concurrent_writer, args=(cache, i, writes_per_thread)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    expected_size = initial_size + (num_threads * writes_per_thread)
    assert_test(
        f"All {num_threads * writes_per_thread} concurrent writes succeeded",
        cache.size() == expected_size,
        f"Expected {expected_size}, got {cache.size()}",
    )

    # ========================================
    # Test 8: Concurrent Reads
    # ========================================
    print("\n[Test 8: Concurrent Reads]")
    cache.put("shared_key", "shared_value", ttl=10)

    results = []

    def concurrent_reader(cache, results, num_reads):
        for _ in range(num_reads):
            val = cache.get("shared_key")
            results.append(val)

    threads = []
    for i in range(10):
        t = threading.Thread(target=concurrent_reader, args=(cache, results, 10))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert_test(
        "All concurrent reads got correct value",
        all(v == "shared_value" for v in results),
        f"Got {len([v for v in results if v != 'shared_value'])} incorrect reads",
    )

    # ========================================
    # Test 9: Mixed Read/Write/Delete
    # ========================================
    print("\n[Test 9: Mixed Operations Under Contention]")
    stop_flag = threading.Event()

    def random_operations(cache, stop_flag, op_counts):
        import random

        while not stop_flag.is_set():
            op = random.choice(["put", "get", "delete"])
            key = f"mixed_{random.randint(0, 20)}"

            if op == "put":
                cache.put(key, f"value_{time.time()}", ttl=5)
                op_counts[0] += 1
            elif op == "get":
                cache.get(key)
                op_counts[1] += 1
            else:
                cache.delete(key)
                op_counts[2] += 1

    op_counts = [0, 0, 0]  # [puts, gets, deletes]
    threads = []

    for i in range(5):
        t = threading.Thread(
            target=random_operations, args=(cache, stop_flag, op_counts)
        )
        threads.append(t)
        t.start()

    time.sleep(2)  # Run for 2 seconds
    stop_flag.set()

    for t in threads:
        t.join()

    total_ops = sum(op_counts)
    assert_test(
        f"Mixed operations completed without crashes ({total_ops} ops)",
        total_ops > 0,
        f"Puts: {op_counts[0]}, Gets: {op_counts[1]}, Deletes: {op_counts[2]}",
    )

    # ========================================
    # Test 10: Very Short TTL
    # ========================================
    print("\n[Test 10: Very Short TTL (100ms)]")
    cache.put("very_short", "quick", ttl=0.1)
    time.sleep(0.15)
    assert_test("100ms TTL expired correctly", cache.get("very_short") is None)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    cache.stop()
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
