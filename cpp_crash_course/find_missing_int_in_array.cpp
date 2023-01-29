#include <vector>
#include <iostream>

int find_missing_num_in_array(std::vector<int> arr) {
  int xor_res = 0;
  for(const int& i : arr) {
    xor_res ^= i;
  }

  for(int i = 1; i <= arr.size() + 1; ++i) {
    xor_res ^= i;
  }
  return xor_res;
}

int main() {
  std::vector a = {1, 2, 3, 5};
  std::cout << find_missing_num_in_array(a) << std::endl;
  return 0;
}