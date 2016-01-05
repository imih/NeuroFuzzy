#include "neuro_fuzzy_network.h"

#include <memory>
#include <functional>

using std::placeholders::_1;

NeuroFuzzyNetwork::NeuroFuzzyNetwork(int m)
    : m_(m), distribution_(-kMaxParVal, kMaxParVal) {
  for (int i = 0; i < m; ++i) {
    a_.push_back(randDouble());
    b_.push_back(randDouble());
    c_.push_back(randDouble());
    d_.push_back(randDouble());
    p_.push_back(randDouble());
    q_.push_back(randDouble());
    r_.push_back(randDouble());
  }
}

void NeuroFuzzyNetwork::train(DescentType descent_type,
                              const TrainData& train_data) {}  // TODO

double NeuroFuzzyNetwork::A(int i, double x) {
  return 1 / (1 + exp(b_[i] * (x - a_[i])));
}

double NeuroFuzzyNetwork::B(int i, double x) {
  return 1 / (1 + exp(d_[i] * (x - c_[i])));
}

double NeuroFuzzyNetwork::w(int i, double x, double y) {
  return p_[i] * x + q_[i] * y + r_[i];
}

double NeuroFuzzyNetwork::o(double x, double y) {
  double f = 0;
  double g = 0;
  for (int j = 0; j < m_; ++j) {
    double a_j = alpha(j, x, y);
    double w_j = w(j, x, y);
    f += a_j * w_j;
    g += a_j;
  }
  return f / g;
}

double NeuroFuzzyNetwork::avg(
    const TrainData& train_data,
    double (NeuroFuzzyNetwork::*f)(const TrainSample&)) {
  double avg = 0;
  for (const TrainSample& ts : train_data) avg += (this->*f)(ts);
  return avg / (int)train_data.size();
}

double NeuroFuzzyNetwork::avg(
    int i, const TrainData& train_data,
    double (NeuroFuzzyNetwork::*f)(int i, const TrainSample&)) {
  double avg = 0;
  for (const TrainSample& ts : train_data) avg += (this->*f)(i, ts);
  return avg / (int)train_data.size();
}

double NeuroFuzzyNetwork::Ek(const TrainSample& train_sample) {
  double o_k = o(train_sample.x, train_sample.y);
  return (train_sample.z - o_k) * (train_sample.z - o_k) / 2;
}

double NeuroFuzzyNetwork::E(const TrainData& train_data) {
  return avg(train_data, &NeuroFuzzyNetwork::Ek);
}

double NeuroFuzzyNetwork::pom_ant(int i, const TrainSample& train_sample) {
  double x = train_sample.x;
  double y = train_sample.y;
  double z_k = train_sample.z;
  double o_k = o(x, y);
  double w_i = w(i, x, y);
  double f = 0;
  double g = 0;
  for (int j = 0; j < m_; ++j) {
    double a_j = alpha(j, x, y);
    if (i != j) f += (w_i - w(j, x, y)) * a_j;
    g += a_j;
  }
  g = g * g;
  double a_i = alpha(i, x, y);
  return fabs(z_k - o_k) * f * a_i / g;
}

double NeuroFuzzyNetwork::dEa(int i, const TrainSample& train_sample) {
  return -pom_ant(i, train_sample) * (1 - A(i, train_sample.x)) * b_[i];
}

double NeuroFuzzyNetwork::dEa(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEa);
}

double NeuroFuzzyNetwork::dEb(int i, const TrainSample& train_sample) {
  return pom_ant(i, train_sample) * (1 - A(i, train_sample.x)) *
         (train_sample.x - a_[i]);
}

double NeuroFuzzyNetwork::dEb(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEb);
}

double NeuroFuzzyNetwork::dEc(int i, const TrainSample& train_sample) {
  return -pom_ant(i, train_sample) * (1 - B(i, train_sample.y)) * d_[i];
}

double NeuroFuzzyNetwork::dEc(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEc);
}

double NeuroFuzzyNetwork::dEd(int i, const TrainSample& train_sample) {
  return pom_ant(i, train_sample) * (1 - B(i, train_sample.y)) *
         (train_sample.y - c_[i]);
}

double NeuroFuzzyNetwork::dEd(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEd);
}

double NeuroFuzzyNetwork::pom_konc(int i, const TrainSample& train_sample) {
  double x = train_sample.x;
  double y = train_sample.y;
  double z = train_sample.z;
  double o_k = o(x, y);
  double a_i = alpha(i, x, y);
  double g = 0;
  for (int j = 0; j < m_; ++j) g += alpha(j, x, y);
  return -fabs(z - o_k) * a_i / g;
}

double NeuroFuzzyNetwork::dEp(int i, const TrainSample& train_sample) {
  return pom_konc(i, train_sample) * train_sample.x;
}

double NeuroFuzzyNetwork::dEp(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEp);
}

double NeuroFuzzyNetwork::dEq(int i, const TrainSample& train_sample) {
  return pom_konc(i, train_sample) * train_sample.y;
}

double NeuroFuzzyNetwork::dEq(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEq);
}

double NeuroFuzzyNetwork::dEr(int i, const TrainSample& train_sample) {
  return pom_konc(i, train_sample);
}

double NeuroFuzzyNetwork::dEr(int i, const TrainData& train_data) {
  return avg(i, train_data, &NeuroFuzzyNetwork::dEr);
}

