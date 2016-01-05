#include "neuro_fuzzy_network.h"

#include <iostream>
#include <limits>

namespace {
const double eps = 10e-6;
const double nabla = 0.001;
const int kPrintStep = 1;
}

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
                              const TrainData& train_data) {
  int it = 0;
  double E_prev = std::numeric_limits<double>::max();
  while (E(train_data) > eps) {
    double E_cur = E(train_data);
    if (E_cur > E_prev) break;
    E_prev = E_cur;
    it++;
    if (it % kPrintStep == 0) fprintf(stderr, "%d %lf\n", it, E_cur);
    for (int i = 0; i < m_; ++i) {
      double da, db, dc, dd, dp, dq, dr;
      if (descent_type == DescentType::STOCHASTIC) {
        int k = randInt((int)train_data.size());
        const TrainSample& ts = train_data[k];
        da = dEa(i, ts);
        db = dEb(i, ts);
        dc = dEc(i, ts);
        dd = dEd(i, ts);
        dp = dEp(i, ts);
        dq = dEq(i, ts);
        dr = dEr(i, ts);
      } else {
        // DescentType::Batch
        da = dEa(i, train_data);
        db = dEb(i, train_data);
        dc = dEc(i, train_data);
        dd = dEd(i, train_data);
        dp = dEp(i, train_data);
        dq = dEq(i, train_data);
        dr = dEr(i, train_data);
      }

      a_[i] -= nabla * da;
      b_[i] -= nabla * db;
      c_[i] -= nabla * dc;
      d_[i] -= nabla * dd;
      p_[i] -= nabla * dp;
      q_[i] -= nabla * dq;
      r_[i] -= nabla * dr;
    }
    if (it % 1000 == 0) {
      for (int i = 0; i < m_; ++i) {
        printf("%lf %lf %lf %lf %lf %lf %lf\n", a_[i], b_[i], c_[i], d_[i],
               p_[i], q_[i], r_[i]);
        fprintf(stderr, "%lf %lf %lf %lf %lf %lf %lf\n", a_[i], b_[i], c_[i],
                d_[i], p_[i], q_[i], r_[i]);
      }
    }
  }
}

double NeuroFuzzyNetwork::A(int i, double x) {
  return 1.0 / (1 + exp(b_[i] * (x - a_[i])));
}

double NeuroFuzzyNetwork::B(int i, double x) {
  return 1.0 / (1 + exp(d_[i] * (x - c_[i])));
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
  return (z_k - o_k) * f * a_i / g;
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
  return -(z - o_k) * a_i / g;
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

