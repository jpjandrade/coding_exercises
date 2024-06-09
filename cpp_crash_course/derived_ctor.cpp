#include <iostream>

class Base {
 public:
  Base(int x) { std::cout << "Base called with " << x << std::endl; }
};

class Derived : private Base {
 public:
  Derived(int y) : Base(y) { std::cout << "Derived called with " << y << std::endl; }
};

int main() {
  Derived d(10);
  return 0;
}