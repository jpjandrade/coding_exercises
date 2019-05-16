#include <iostream>

using namespace std;

int gcd(int a, int b) {
  int temp;
  if (a < b) {
    temp = b;
    b = a;
    a = b;
  }

  while (b != 0) {
    temp = b;
    b = a % b;
    a = temp;
  }

  return a;
}
int main () {
  int a, b, res;
  cout << "Please type a and b" << endl;
  cin >> a;
  cin >> b;

  res = gcd(a, b);

  cout << "gcd between " << a << " and " << b << " is " << res << endl;
  return 0;
}
