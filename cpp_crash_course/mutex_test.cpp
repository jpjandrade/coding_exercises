// mutex example
#include <iostream>  // std::cout
#include <mutex>     // std::mutex
#include <thread>    // std::thread

std::mutex mtx;  // mutex for critical section

void print_block(int n, char c, bool use_locks) {
  // critical section (exclusive access to std::cout signaled by locking mtx):
  // comment and uncomment locks to see mutex effect
  if (use_locks) mtx.lock();
  for (int i = 0; i < n; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::cout << c;
  }
  std::cout << '\n';
  if (use_locks) mtx.unlock();
}

void multi_thread_prints(bool use_locks) {
  std::thread th1(print_block, 50, '*', use_locks);
  std::thread th2(print_block, 50, '$', use_locks);
  std::thread th3(print_block, 50, '%', use_locks);

  th1.join();
  th2.join();
  th3.join();
}

int main() {
  std::cout << "Without locks" << std::endl;
  multi_thread_prints(false);
  std::cout << std::endl;
  std::cout << "With locks" << std::endl;
  multi_thread_prints(true);

  return 0;
}