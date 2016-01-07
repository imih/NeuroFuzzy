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

TrainData genTrainData() {
  TrainData train_data;
  for (double x = -4; x <= 4; x++)
    for (double y = -4; y <= 4; y++)
      train_data.push_back({(double)x, (double)y, f(x, y)});

  return train_data;
}

void train(int argc, char** argv) {
  int m;          /* getopt */
  DescentType dt; /* getopt */
  int opt;
  double nabla = 0.001;
  while ((opt = getopt(argc, argv, "m:t:s:")) != -1) {
    switch (opt) {
      case 'm':
        m = atoi(optarg);
        break;
      case 't':
        dt = (DescentType)atoi(optarg);
        break;
      case 's':
        nabla = atof(optarg);
    }
  }
  TrainData train_data = genTrainData();
  NeuroFuzzyNetwork n(m, nabla);
  n.train(dt, train_data);
}

void zad6() {
  TrainData train_data = genTrainData();
  NeuroFuzzyNetwork n("params", 0.001);
  for (TrainSample ts : train_data) {
    printf("%lf %lf %lf\n", ts.x, ts.y, n.predict(ts.x, ts.y) - ts.z);
  }
}

int main(int argc, char** argv) {
  train(argc, argv);
  // zad6();

  return 0;
}
