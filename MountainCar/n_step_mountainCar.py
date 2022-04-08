from __future__ import print_function,division
from builtins import range
import gym
import numpy as np #
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
from gym import wrappers
from datetime import datetime
import q_learning_mountainCar
from   q_learning_mountainCar import plot_running_avg, FeatureTransformer,Model,plot_cost_to_go

class SGDRegressor:
    def __init__(self,**kwargs):
        self.w = None
        self.lr = 10e-3

    def partial_fit(self,X,Y):
        if self.w is None:
            D=X.shape[1]
            self.w = np.random.randn(D)/np.sqrt(D)
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self,X):
        return X.dot(self.w)

#replace SKLearn Regressor
q_learning_mountainCar.SGDRegressor = SGDRegressor

def play_one(model,eps,gamma,n=5):
    observation = env.reset()
    done = False
    totalreward = 0
    rewards = []
    states =  []
    actions = []
    iters = 0
    #array of [gamma^0,.......,gamma^(n-1)]
    multiplier = np.array([gamma]*n)**np.arange(n)
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        states.append(observation)
        actions.append(action)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if len(rewards)>=n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n],G)
            env.render()

        totalreward+=reward
        iters += 1

    rewards = rewards[-n+1:]
    states = states[-n + 1:]
    actions = actions[-n + 1:]
    if observation[0] >= 0.5:
        # we actually made it to the goal
        # print("made it!")
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        # we did not make it to the goal
        # print("didn't make it...")
        while len(rewards) > 0:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return totalreward
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        # eps = 1.0/(0.1*n+1)
        eps = 0.1 * (0.97 ** n)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)