import gym
import numpy as np #
from tabulate import tabulate

class Agent():
    def __init__(self):
        #create a reward table matrix
        self.reward_table = np.zeros((5,2))
    def play(self,env):
        # Reset th env before play begins
        state = env.reset()
        # Play the game 1000 steps
        end_game = False
        while not end_game:
            # Choose randomly if empty, otherwise choose action with the highest reward
            if self.__reward_table_is_empty(state):
                action = self.__get_action_randomly(env)
            else:
                action = self.__get_action_highest_reward(state)

            # Perform ACTION
            new_state, reward, end_game, _ = env.step(action)
            #Update the reward table
            self.reward_table[state,action] += reward
            # print the latest info terminal and reward table
            print("Started in state {}, Took action {}, entered new state {} and received reward {}.".format(state, action, new_state, reward))
            print(tabulate(self.reward_table,showindex="always",headers=["State","Action 0","Action 1"]))
            #Update state
            state =  new_state
    def __reward_table_is_empty(self,state):
        return np.sum(self.reward_table[state, :]) == 0

    def __get_action_randomly(self,env):
        return env.action_space.sample()

    def __get_action_highest_reward(self,state):
        return np.argmax(self.reward_table[state, :])

#Create the Nchain-v0 environment
env = gym.make('NChain-v0')

#Create an intelligent agent
agent = Agent()

#Play the game
agent.play(env)





