#ifndef TRAIN_DATA_H
#define TRAIN_DATA_H

#include <vector>

struct TrainSample {
  double x;
  double y;
  double z;
};

typedef std::vector<TrainSample> TrainData;

#endif  // TRAIN_DATA_H
