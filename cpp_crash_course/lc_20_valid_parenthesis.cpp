#include <stack>
#include <string>
#include <unordered_map>
using namespace std;

class Solution {
 public:
  bool isValid(string s) {
    stack<char> st {};
    unordered_map<char, char> complements = {{')', '('}, {']', '['}, {'}', '{'}};

    for (auto c: s) {
      if (complements.count(c)) {
        if (st.empty() || st.top() != complements[c]) {
          return false;
        }
        else {
          st.pop();
        }
      }
      else {
        st.push(c);
      }
    }
    return st.empty();
  }
};