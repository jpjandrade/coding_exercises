#include <iostream>
#include <iomanip>
using namespace std;

void print_line() {
  for(int i = 0; i < 4; i++) {
    cout << setw(17) << "Hello World!" ;
  }
  cout << endl;
}


int main() {
  char user_decision;
  cout << "Do you want left justified? (Y / any)" << endl;

  cin >> user_decision;
  if(user_decision == 'Y' || user_decision == 'y') {
    cout <<  setiosflags(ios::left);
  }

  for(int i = 0; i < 6; i++) {
    print_line();
  }

  return 0;
}