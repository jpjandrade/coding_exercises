#include <string>
#include <vector>
using namespace std;

class Solution {
 public:
  int strStr(string haystack, string needle) {
    if (needle.size() == 0) return 0;
    for (int i = 0; i < haystack.size(); i++) {
      if (found_str(haystack, needle, i)) return i;
    }
    return -1;
  }

 private:
  bool found_str(string haystack, string needle, int i) {
    for (int j = 0; j < needle.size(); j++) {
      if (i + j > haystack.size() | haystack[i + j] != needle[j]) return false;
    }
    return true;
  }
};