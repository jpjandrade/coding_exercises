import logging
import threading
import queue
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import random

MAX_URLS = 50
N_FETCHERS = 3
N_PARSERS = 2

fetch_queue = queue.Queue(maxsize=MAX_URLS)
parse_queue = queue.Queue()
visited_urls = set()
visited_lock = threading.Lock()
counter_lock = threading.Lock()
dropped_urls = 0
shutdown_event = threading.Event()


def fetcher():
    while not shutdown_event.is_set():
        try:
            url = fetch_queue.get(timeout=1)
        except queue.Empty:
            continue

        with visited_lock:
            if url in visited_urls or len(visited_urls) >= MAX_URLS:
                fetch_queue.task_done()
                continue
            visited_urls.add(url)
            count = len(visited_urls)

        logging.info(
            f"[Fetcher {threading.current_thread().name}] Fetching ({count}/{MAX_URLS}): {url}"
        )

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            parse_queue.put((url, response.text))
        except Exception as e:
            logging.error(
                f"[Fetcher {threading.current_thread().name}] Error fetching {url}: {e}"
            )

        fetch_queue.task_done()

        with visited_lock:
            if len(visited_urls) >= MAX_URLS:
                shutdown_event.set()


def parser():
    global dropped_urls

    while not shutdown_event.is_set():
        try:
            url, html = parse_queue.get(timeout=1)
        except queue.Empty:
            continue

        logging.info(f"[Parser {threading.current_thread().name}] Parsing {url}")

        try:
            soup = BeautifulSoup(html, "html.parser")

            title = soup.title.string if soup.title else "No title"
            logging.info(f"  -> Title: {title[:60]}...")

            for link in soup.find_all("a", href=True):
                new_url = urljoin(url, link["href"])
                new_url = new_url.split("#")[0]

                with visited_lock:
                    if new_url not in visited_urls and len(visited_urls) < MAX_URLS:
                        try:
                            fetch_queue.put_nowait(new_url)
                        except queue.Full:
                            with counter_lock:
                                dropped_urls += 1

        except Exception as e:
            logging.error(
                f"[Parser {threading.current_thread().name}] Error parsing {url}: {e}"
            )

        parse_queue.task_done()


def monitor():
    while not shutdown_event.is_set():
        print(
            f"[Stats] Fetch queue: {fetch_queue.qsize()}, parse queue: {parse_queue.qsize()}, visited: {len(visited_urls)}"
        )
        time.sleep(2)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seed_file = os.path.join(script_dir, "seed_urls.txt")
    with open(seed_file, "r") as f:
        seed_urls = [line.strip() for line in f if line.strip()]

    random.shuffle(seed_urls)
    for url in seed_urls:
        fetch_queue.put(url)

    fetchers = [
        threading.Thread(target=fetcher, daemon=True, name=f"F-{i}")
        for i in range(N_FETCHERS)
    ]
    parsers = [
        threading.Thread(target=parser, daemon=True, name=f"P-{i}")
        for i in range(N_PARSERS)
    ]

    t = threading.Thread(target=monitor, daemon=True)
    t.start()

    for t in fetchers:
        t.start()

    for t in parsers:
        t.start()

    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        logging.info("\n\nInterrupted by user (Ctrl+C). Shutting down...")
        shutdown_event.set()

    # Give threads a moment to finish current work
    logging.info("Shutting down...")

    logging.info(f"Done! Visited {len(visited_urls)} URLs:")
    with open("visited_urls.txt", "w") as f:
        for url in visited_urls:
            print(f"  - {url}")
            f.write(url + "\n")
    logging.info(f"Dropped URLs due to full queue: {dropped_urls}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
