#include <iostream>
#include "Vector.cpp"

using namespace std;

class Container {
 public:
  virtual double& operator[](int) = 0;
  virtual int size() const = 0;
  virtual ~Container() {}
};

class Vector_container : public Container {
 public:
  Vector_container(int s) : v(s) {}
  ~Vector_container() {}

  double& operator[](int i) override { return v[i]; }
  int size() const override { return v.size(); }

 private:
  Vector v;
};

void use(Container& c) {
  const int sz = c.size();
  for (int i = 0; i != sz; i++) {
    cout << c[i] << " ";
  }
  cout << endl;
}

double read_and_sum_with_use(int s) {
  Vector_container vc = Vector_container(s);
  for (int i = 0; i != vc.size(); ++i) std::cin >> vc[i];

  double sum = 0;
  for (int i = 0; i != vc.size(); ++i) sum += vc[i];

  cout << "Summed the following vector: \n";
  use(vc);

  return sum;
}

int this_main() {
  double res = read_and_sum_with_use(5);
  cout << "Total sum for that is " << res << "\n";
  return 0;
}