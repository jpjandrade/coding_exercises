#include <iostream>

int main() {
  std::cout << "Do you want to proceed? (Y/n)\n";
  char answer = 0;
  std::cin >> answer;

  if (answer == 'Y' | answer == 'y') {
    std::cout << "True!\n";
  } else {
    if (answer != 'N' & answer != 'n') {
      std::cout << "I'll take that as a no.\n";
    }
    std::cout << "False!\n";
  }

  return 0;
}