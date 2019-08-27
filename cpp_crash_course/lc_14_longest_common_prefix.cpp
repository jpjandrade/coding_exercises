#include <string>
#include <vector>
using namespace std;

class Solution {
 public:
  string longestCommonPrefix(vector<string>& strs) {
    if (strs.size() == 0)
      return "";
    bool done = false;
    int l = 0;

    while (!done) {
      if (all_chars_match(strs, l))
        l++;
      else
        done = true;
    }
    return strs[0].substr(0, l);
  }

  private:
    bool all_chars_match(vector<string>& strs, int l) {
      int n = strs.size();
      if (strs[0].size() < l)
        return false;
      for (int i = 1; i < n; i++) {
        if (strs[i].size() < l || strs[i][l] != strs[i - 1][l])
          return false;
      }
      return true;
    }
  };