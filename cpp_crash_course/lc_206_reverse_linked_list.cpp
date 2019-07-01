

struct ListNode {
  int val;
  ListNode* next;
  ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
 public:
  ListNode* reverseList(ListNode* head) {
    if (head == nullptr || head -> next == nullptr) {
      return head;
    }

    ListNode* prev = nullptr;
    ListNode *n = head;
    ListNode * temp = nullptr;
    while (n != nullptr) {
      temp = n -> next;
      n -> next = prev;
      prev = n;
      n = temp;
    }
    return prev;
  }
};