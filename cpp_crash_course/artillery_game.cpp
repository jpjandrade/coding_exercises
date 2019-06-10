#include <math.h>
#include <ctime>
#include <iostream>

using namespace std;

void introduction(int const cannon_balls) {
  cout << "Welcome to Artillery." << endl;
  cout << "You are in the middle of a war and being charged by thousands of enemies." << endl;
  cout << "You have one cannon, which you can shoot at any angle." << endl;
  cout << "You only have " << cannon_balls << " cannonballs for this target.." << endl;
  cout << "Let's begin..." << endl;
}

int get_random_dist() {
  int k = rand() % 700 + 200;
  return k;
}

double process_shot() {
  double angle;
  cout << "What angle? ";
  while (!(cin >> angle)) {
    cout << "Please type a numeric value" << endl;
    cin.clear();
    cin.ignore(1000, '\n');
  }
  cout << endl;
  return angle;
}

int distance_travelled(double angle_in_degrees) {
  double kPi = 3.14159265358979323846;
  double kVelocity = 200.0;  // initial velocity of 200 ft/sec
  double kGravity = 32.2;    // gravity for distance calculation
  double angle = angle_in_degrees * kPi / 180.0;
  // in_angle is the angle the player has entered, converted to radians.
  double time_in_air = (2.0 * kVelocity * sin(angle)) / kGravity;
  int distance = round((kVelocity * cos(angle) * time_in_air));
  return distance;
}

bool hit(int a, int b) {
  int kExplosionRadius = 5;
  return abs(a - b) < kExplosionRadius;
}

int fire(int total_cannon_balls) {
  int curr_cannon_balls = total_cannon_balls;
  int enemy_distance = get_random_dist();
  int curr_distance = 0;
  double angle;
  cout << "The enemy is " << enemy_distance << " feet away!" << endl;
  while (!hit(curr_distance, enemy_distance) && curr_cannon_balls > 0) {
    angle = process_shot();
    curr_distance = distance_travelled(angle);
    curr_cannon_balls--;
    cout << "Distance fired: " << curr_distance << endl;
    if (curr_distance < enemy_distance) {
      cout << "You under shot by " << enemy_distance - curr_distance << endl;
    } else if (curr_distance > enemy_distance) {
      cout << "You over shot by " << curr_distance - enemy_distance << endl;
    }
  }
  if (hit(curr_distance, enemy_distance)) {
    cout << "****You hit him!!****" << endl;
    cout << "It took you " << total_cannon_balls - curr_cannon_balls << " shots." << endl;
    return 1;
  } else {
    cout << "You are out of cannon balls for this enemy!" << endl;
    return 0;
  }
}

int main() {
  srand(time(NULL));

  int const kCannonBalls = 10;
  introduction(kCannonBalls);  // This displays the introductory script.
  int killed = 0;
  char done = 'Y';

  do {
    killed += fire(kCannonBalls);  // Fire() contains the main loop of each round.
    cout << "I see another one, care to shoot again? (Y/N) " << endl;
    cin >> done;
  } while ((done != 'n') && (done != 'N'));

  cout << "You killed " << killed << " of the enemy!" << endl;
}