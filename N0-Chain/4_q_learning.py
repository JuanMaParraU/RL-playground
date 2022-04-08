import gym
import numpy as np #
from tabulate import tabulate
import matplotlib.pyplot as plt
import platform

class Agent():
    def __init__(self):
        #create a reward table matrix
        self.q_table = np.zeros((5, 2))
        self.learning_rate = 0.05 # learning rate towards a target Q value
        self.discount_factor = 0.95 # for reward in the future
        self.epsilon = 0.5 #exploration rate
        self.decay_factor = 0.999
        self.average_reward_for_each_episode = []

    def play(self,env, episodes=200):
            for epi in range(episodes):
                print("Episode {}".format(epi))
                # Reset th env before play begins
                state = env.reset()
                total_reward = 0
                #reduce exploration rate
                self.epsilon *= self.decay_factor
                # Play the game 1000 steps
                end_game = False
                while not end_game:
                    # Choose randomly if empty or probability for explore, otherwise choose action with the highest reward
                    if self.__q_table_is_empty(state) or self.__with_probability(self.epsilon):
                        action = self.__get_action_randomly(env)
                    else:
                        action = self.__get_action_highest_reward(state)
                    # Perform ACTION
                    new_state, reward, end_game, _ = env.step(action)
                    total_reward += reward
                    #Update the reward table with Bellman Equation - > Q Learning Eq
                    self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * self._get_expected_reward_in_next_state(new_state) - self.q_table[state,action] )
                    # print the latest info terminal and reward table
                    #print("Started in state {}, Took action {}, entered new state {} and received reward {}.".format(state, action, new_state, reward))
                    #Update state
                    state =  new_state
                #Store de average reward and Qtable
                self.average_reward_for_each_episode.append(total_reward / 1000)
                print(tabulate(self.q_table, showindex="always",
                               headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))

    def __q_table_is_empty(self,state):
        return np.sum(self.q_table[state, :]) == 0

    def __with_probability(self,prob):
        return np.random.random() < prob

    def __get_action_randomly(self,env):
        return env.action_space.sample()

    def __get_action_highest_reward(self,state):
        return np.argmax(self.q_table[state, :])

    def _get_expected_reward_in_next_state(self,next_state): #access to next state assuming next action is optimal based on Q
        return np.max(self.q_table[next_state,:])

def graph(average_reward):
    plt.plot(average_reward)
    plt.title('Performance over time')
    plt.ylabel('Average reward')
    plt.xlabel('Episode')
    plt.show()

#Create the Nchain-v0 environment
env = gym.make('NChain-v0')

#Create an intelligent agent
agent = Agent()

#Play the game
agent.play(env)
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
print(platform.platform())
graph(agent.average_reward_for_each_episode)




