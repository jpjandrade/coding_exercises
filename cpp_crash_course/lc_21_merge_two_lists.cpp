#include <iostream>
#include <unordered_map>
#include <vector>

struct ListNode {
  int val;
  ListNode* next;
  ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
 public:
  ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* res = new ListNode(-1);
    ListNode* head = res;
    while (l1 != nullptr && l2 != nullptr) {
      if (l1->next->val <= l2->next->val) {
        res->next = new ListNode(l1->val);
        l1 = l1->next;
      } else {
        res->next = new ListNode(l2->val);
        l2 = l2->next;
      }
      res = res->next;
    }

    while (l1 != nullptr) {
      res->next = new ListNode(l1->val);
      l1 = l1->next;
      res = res->next;
    }

    while (l2 != nullptr) {
      res->next = new ListNode(l2->val);
      l2 = l2->next;
      res = res->next;
    }

    return head->next;
  }
};