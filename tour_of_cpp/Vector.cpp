#include <iostream>

class Vector {
 public:
  Vector(int s) : elem{new double[s]}, sz{s} {}
  ~Vector() { delete[] elem; }

  Vector(const Vector& a);
  Vector& operator=(const Vector& a);

  Vector(Vector&& a);
  Vector& operator=(Vector&& a);

  double& operator[](int i) { return elem[i]; }
  int size() const { return sz; }

 private:
  double* elem;
  int sz;
};

Vector::Vector(const Vector& a) : elem{new double[a.sz]}, sz{a.sz} {
  for (int i = 0; i != sz; i++) elem[i] = a.elem[i];
}

Vector& Vector::operator=(const Vector& a) {
  double* p = new double[a.sz];
  for (int i = 0; i != a.sz; ++i) p[i] = a.elem[i];
  delete[] elem;
  elem = p;
  sz = a.sz;
  return *this;
}

Vector::Vector(Vector&& a) : elem{a.elem}, sz{a.sz} {
  a.elem = nullptr;
  a.sz = 0;
}

double read_and_sum(int s) {
  Vector v(s);
  for (int i = 0; i != v.size(); ++i) std::cin >> v[i];

  double sum = 0;
  for (int i = 0; i != v.size(); ++i) sum += v[i];

  std::cout << "Summed the following vector: \n";
  for (int i = 0; i != v.size(); ++i) std::cout << v[i] << " ";

  std::cout << "\n";
  return sum;
}

int main() {
  double res;

  res = read_and_sum(5);
  std::cout << "Total sum for that is " << res << "\n";
  return 0;
}