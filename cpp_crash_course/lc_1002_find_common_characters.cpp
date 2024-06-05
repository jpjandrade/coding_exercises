#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

class Solution {
 public:
  vector<string> commonChars(vector<string>& words) {
    if (words.empty()) {
      return {};
    }
    unordered_map<char, int> common_chars;
    for (const char& c : words.at(0)) {
      common_chars[c]++;
    }

    for (int i = 1; i < words.size(); ++i) {
      unordered_map<char, int> curr_word;
      for (const auto& c : words.at(i)) {
        curr_word[c]++;
      }

      for (const auto& [c, count] : common_chars) {
        if (curr_word[c] < common_chars[c]) {
          common_chars[c] = curr_word[c];
        }
      }
    }

    vector<std::string> res;
    for (const auto& [c, count] : common_chars) {
      for (int i = 0; i < count; ++i) {
        res.push_back(string(1, c));
      }
    }
    return res;
  }
};

int main() {
  vector<string> words = {"hello", "helicopter", "horizon"};
  Solution sol;
  vector<string> res = sol.commonChars(words);
  for (const string& s : res) {
    std::cout << s << std::endl;
  }

  std::cout << std::endl;
}