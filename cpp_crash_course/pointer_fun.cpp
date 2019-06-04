#include <iostream>
using namespace std;

void DoIt(int &foo, int goo);

int main() {
  int *foo, *goo;
  foo = new int;
  *foo = 1;
  goo = new int;
  *goo = 3;
  *foo = *goo + 3;
  foo = goo;
  *goo = 5;
  *foo = *goo + *foo; // foo and goo point to 10
  DoIt(*foo, *goo);
  cout << (*foo) << endl;
}

void DoIt(int &foo, int goo) {
  foo = goo + 3;
  goo = foo + 4;
  foo = goo + 3;
  goo = foo;
} 