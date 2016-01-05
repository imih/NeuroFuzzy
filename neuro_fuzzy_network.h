#ifndef NEURO_FUZZY_NETWORK_H
#define NEURO_FUZZY_NETWORK_H

#include "train_data.h"

#include <random>
#include <vector>

namespace {
double kMaxParVal = 1;
}

enum DescentType { STOCHASTIC, BATCH };

class NeuroFuzzyNetwork {
 public:
  NeuroFuzzyNetwork(int m);
  void train(DescentType descent_type, const TrainData& train_data);

  double predict(double x, double y) { return o(x, y); }

 private:
  double randDouble() { return distribution_(generator_); }

  double A(int i, double x);
  double B(int i, double x);
  double w(int j, double x, double y);
  double alpha(int j, double x, double y) { return A(j, x) * B(j, y); }
  double o(double x, double y);

  double avg(const TrainData& train_data,
             double (NeuroFuzzyNetwork::*f)(const TrainSample& ts));
  double avg(int i, const TrainData& train_data,
             double (NeuroFuzzyNetwork::*f)(int i, const TrainSample& ts));

  double E(const TrainData& train_data);
  double Ek(const TrainSample& train_sample);

  double dEa(int i, const TrainSample& train_sample);
  double dEa(int i, const TrainData& train_data);

  double dEb(int i, const TrainSample& train_sample);
  double dEb(int i, const TrainData& train_data);

  double dEc(int i, const TrainSample& train_sample);
  double dEc(int i, const TrainData& train_data);

  double dEd(int i, const TrainSample& train_sample);
  double dEd(int i, const TrainData& train_data);

  double dEp(int i, const TrainSample& train_sample);
  double dEp(int i, const TrainData& train_data);

  double dEq(int i, const TrainSample& train_sample);
  double dEq(int i, const TrainData& train_data);

  double dEr(int i, const TrainSample& train_sample);
  double dEr(int i, const TrainData& train_data);

  double pom_ant(int i, const TrainSample& train_sample);
  double pom_konc(int i, const TrainSample& train_sample);

  int m_;
  std::vector<double> a_, b_, c_, d_, p_, q_, r_;

  std::default_random_engine generator_;
  std::uniform_real_distribution<double> distribution_;
};

#endif  // NEURO_FUZZY_NETWORK_H