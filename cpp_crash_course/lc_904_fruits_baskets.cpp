#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;
class Solution {
 public:
  int totalFruit(vector<int>& tree) {
    unordered_map<int, int> fruit_cts;
    int res = 0;
    int p = 0;
    for (int i = 0; i < tree.size(); i++) {
      add_or_initialize(fruit_cts, tree[i]);
      while (fruit_cts.size() > 2) {
        fruit_cts[tree[p]]--;
        if (fruit_cts[tree[p]] == 0) fruit_cts.erase(tree[p]);
        p++;
      }
      res = max(res, i + 1 - p);
    }
    return max(res, (int)tree.size() - p);
  }

 private:
  void add_or_initialize(unordered_map<int, int>& d, int k) {
    if (d.count(k) > 0)
      d[k]++;
    else
      d[k] = 1;
  }
};