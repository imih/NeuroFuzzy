#include "neuro_fuzzy_network.h"
#include "train_data.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

double f(int x, int y) {
  return ((x - 1) * (x - 1) + (y + 2) * (y + 2) - 5 * x * y + 3) * cos(x / 5) *
         cos(x / 5);
}

int main(int argc, char** argv) {
  int m;          /* getopt */
  DescentType dt; /* getopt */
  int opt;
  while ((opt = getopt(argc, argv, "m:t:")) != -1) {
    switch (opt) {
      case 'm':
        m = atoi(optarg);
        break;
      case 't':
        dt = (DescentType)atoi(optarg);
        break;
    }
  }

  TrainData train_data;
  for (int x = -4; x <= 4; ++x)
    for (int y = -4; y <= 4; ++y)
      train_data.push_back({(double)x, (double)y, f(x, y)});

  NeuroFuzzyNetwork n(m);
  n.train(dt, train_data);
  return 0;
}
