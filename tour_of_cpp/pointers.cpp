#include <iostream>

int main() {
  int x = 2;
  int y = 3;
  int* p = &x;
  int* q = &y;

  int& r = x;
  int& r2 = y;
  std::cout << *p << "\n" << *q << "\n";
  std::cout << r << "\n" << r2 << "\n";
  std::cout << p << "\n" << q << "\n";

  p = q;

  std::cout << *p << "\n" << *q << "\n";
  std::cout << r << "\n" << r2 << "\n";
  std::cout << p << "\n" << q << "\n";

  char greeting[] = "Hello";
  char* pt = greeting;

  std::cout << pt << std::endl;
  std::cout << *pt << std::endl;

  return 0;
}
