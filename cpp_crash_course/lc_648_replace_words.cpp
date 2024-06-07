#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Mostly copied from editorial with some changes on my side to learn Tries in
// C++.
class TrieNode {
 public:
  bool is_end;
  std::vector<std::unique_ptr<TrieNode>> children;
  TrieNode() : children(26), is_end(false) {}
};
class Trie {
 public:
  Trie() : root_(std::make_unique<TrieNode>()) {}

  void insert(const std::string& word) {
    TrieNode* current = root_.get();

    for (const char& c : word) {
      int index = c - 'a';
      if (current->children[index] == nullptr) {
        current->children[index] = std::make_unique<TrieNode>();
      }
      std::cout << (current->children.size());
      current = current->children[index].get();
    }
    current->is_end = true;
  }

  std::string shortest_root(const std::string& word) {
    TrieNode* current = root_.get();
    for (int i = 0; i < word.size(); ++i) {
      const char& c = word[i];
      if (current->children[c - 'a'] == nullptr) {
        // Couldn't find a stem in the dict.
        return word;
      }
      current = current->children[c - 'a'].get();

      if (current->is_end) {
        return word.substr(0, i + 1);
      }
    }

    return word;
  }

 private:
  std::unique_ptr<TrieNode> root_;
};

// Copied from the editorial to learn tries in C++.
class Solution {
 public:
  std::string replaceWords(std::vector<std::string>& dictionary,
                           std::string sentence) {
    Trie trie = build_trie(dictionary);

    std::stringstream stream(sentence);
    std::string word;
    std::stringstream new_sentence;

    while (getline(stream, word, ' ')) {
      new_sentence << trie.shortest_root(word) << " ";
    }

    std::string final_sentence = new_sentence.str();
    final_sentence.pop_back();
    return final_sentence;
  }

 private:
  Trie build_trie(std::vector<std::string> dictionary) {
    Trie trie;
    for (const std::string& word : dictionary) {
      trie.insert(word);
    }
    return trie;
  }
};

int main() {
  Solution solution;
  std::vector<std::string> dictionary{"cat", "bat", "rat"};
  std::string sentence = "the cat was rat by the bat";

  std::string new_sentence = solution.replaceWords(dictionary, sentence);

  std::cout << new_sentence << std::endl;
}