from __future__ import print_function,division
from builtins import range
import gym
import numpy as np #
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from gym import wrappers
from datetime import datetime
from q_learning_bins import plot_running_avg

class SGDRegressor:
    def __init__(self,D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr=10e-2

    def partial_fit(self,X,Y):
        self.w += self.lr*(Y-X.dot(self.w)).dot(X)

    def predict(self,X):
        return X.dot(self.w)

class FeatureTransformer:
  def __init__(self, env, n_components=1000):
    observation_examples = np.random.random((20000,4))*2-2
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
  def transform(self, observations):
    # print "observations:", observations
    scaled = self.scaler.transform(observations)
    # assert(len(scaled.shape) == 2)
    return self.featurizer.transform(scaled)

# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      model = SGDRegressor(feature_transformer.dimensions)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    result = np.array([m.predict(X)[0] for m in self.models])
    return result

  def update(self, s, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):
    # eps = 0
    # Technically, we don't need to do epsilon-greedy
    # because SGDRegressor predicts 0 for all states
    # until they are updated. This works as the
    # "Optimistic Initial Values" method, since all
    # the rewards for Mountain Car are -1.
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


##Turns list of integers into an int eg.[1,2,3,4,5] to 12345
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def play_one(env,model, eps, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 10000:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    if done:
        reward =-200
    # update the model
    next = model.predict(observation)
    assert(len(next.shape)==1)
    G = reward + gamma*np.max(next)
    model.update(prev_observation, action, G)
    if reward == 1:
        totalreward += reward
    iters += 1

  return totalreward

def main():
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer(env)
    model = Model(env,ft)
    gamma = 0.99
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one(env,model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
if __name__=='__main__':
    main()