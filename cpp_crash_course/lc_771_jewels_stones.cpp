#include <string>
#include <iostream>
#include <set>
using namespace std;

class Solution {
public:
    int numJewelsInStones(string J, string S) {
        set<char> jewel_set;
        for (auto& c : J) {
          jewel_set.insert(c);
        }
        int res = 0;
        for (auto& c: S) {
          if (jewel_set.count(c)) res++;
        }
        return res;
    }
};