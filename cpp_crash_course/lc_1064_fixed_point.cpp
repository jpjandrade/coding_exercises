#include <vector>
#include <iostream>
#include <set>
using namespace std;

class Solution {
public:
  int fixedPoint(vector<int>& A) {
    int low = 0;
    int high = A.size();
    int middle;
    while (low <= high) {
      middle = (low + high) / 2;
      if (A[middle] == middle && (middle == 0 || A[middle - 1] < middle - 1)) 
        return middle;
      if (A[middle] > middle) {
        low = middle + 1;
      }
      else {
        high = middle - 1;
      }
    }
    return -1;
  }
};