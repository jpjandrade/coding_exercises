#include <iostream>
#include <ctime>
using namespace std;

int get_random_num() { 
  int k = rand() % 100 + 1;
  return k;
}

int process_guess() {
  int guess;
  cout << "Please type a your guess between 1 and 100 (negative numbers to terminate)" << endl;
  while(!(cin >> guess)) {
    cout << "Please type a numeric value" << endl;
    cin.clear();
    cin.ignore(100, '\n');
  }
  return guess;
}

bool parse_guess(int guess, int target) {
  if (guess == -1 || guess == target) {
    return true;
  }
  if (guess < -1 || guess == 0 || guess > 100) {
    cout << "Please type a number between 1 and 100!" << endl;
  }
  else if (guess < target) {
    cout << "Too low!" << endl;
  }
  else {
    cout << "Too high!" << endl;
  }

  return false; 
}
int main() {
  srand(time(NULL));

  int k = get_random_num();
  cout << k << endl;
  bool done = false;
  int guess;
  int attempts = 0;
  while (!done){
    guess = process_guess();
    done = parse_guess(guess, k);
    attempts++;
  }

  if (guess == k) {
    cout << "*** NICE! You got it! ***" << endl;
    cout << "Number was " << k << endl;
    cout << "Attemtps: " << attempts << endl;
  }
  else {
    cout << "Terminating..." << endl;
  }
  return 0;

}