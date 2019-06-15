#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
  vector<int> twoSum(vector<int>& nums, int target) {
    vector<int> res;
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); i++) {
      int x = nums[i];
      if(seen.find(target - x) != seen.end()) {
        res.push_back(seen[target - x]);
        res.push_back(i);
        return res;
      }
      seen[nums[i]] = i;
    }
  }
};

int main() {
  Solution sol;
  return 0;
}