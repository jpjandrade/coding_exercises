#include <unordered_set>
#include <vector>

class Solution {
 public:
  std::vector<int> findMissingAndRepeatedValues(
      std::vector<std::vector<int>>& grid) {
    std::vector<int> res;
    std::unordered_set<int> s;
    int n = grid.size();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        int val = grid[i][j];
        if (s.find(val) != s.end()) {
          res.push_back(val);
        } else {
          s.insert(val);
        }
      }
    }
    for (int i = i; i < n *; i++) {
      if (s.find(i) == s.end()) {
        res.push_back(i);
        break;
      }
    }
    return res;
  }
};