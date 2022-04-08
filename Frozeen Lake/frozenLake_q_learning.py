import gym
import numpy as np #
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
class Agent():
    def __init__(self):
        #create a reward table matrix
        self.q_table = np.zeros((state_space_size, action_space_size))
        print(action_space_size)
        # > Discrete(2)
        print(state_space_size)
        self.learning_rate = 0.05 # learning rate towards a target Q value
        self.discount_factor = 0.95 # for reward in the future
        self.epsilon = 0.5 #exploration rate
        self.decay_factor = 0.999
        self.average_reward_for_each_episode = []

    def play(self,env, episodes=10000, steps = 100 ):
            for epi in range(episodes):
                print("Episode {}".format(epi))
                # Reset th env before play begins
                state = env.reset()
                total_reward = 0
                #reduce exploration rate
                self.epsilon *= self.decay_factor
                # Play the game 100 steps or until it wins or dies (end_game)
                end_game = False
                #time.sleep(1)
                for step in range(steps):
                    clear_output(wait=True)
                    env.render()
                    time.sleep(0.3)
                    # Choose randomly if empty or probability for explore, otherwise choose action with the highest reward
                    if self.__q_table_is_empty(state) or self.__with_probability(self.epsilon):
                        action = self.__get_action_randomly(env)
                    else:
                        action = self.__get_action_highest_reward(state)
                    # Perform ACTION
                    new_state, reward, end_game, _ = env.step(action)
                    total_reward += reward
                    # Update the reward table with Bellman Equation - > Q Learning Eq
                    self.q_table[state, action] += self.learning_rate * (
                                reward + self.discount_factor * self._get_expected_reward_in_next_state(new_state) -
                                self.q_table[state, action])
                    state = new_state
                    if end_game == True:
                        break
                #Store de average reward and Qtable
                self.average_reward_for_each_episode.append(total_reward)
                #print(tabulate(self.q_table, showindex="always",
                 #              headers=["State", "Left", "Down", "Right", "Up"]))
            # Calculate and print the average reward per thousand episodes
            rewards_per_thousand_episodes = np.split(np.array(self.average_reward_for_each_episode), episodes / 1000)
            count = 1000

            print("********Average reward per thousand episodes********\n")
            for r in rewards_per_thousand_episodes:
                print(count, ": ", str(sum(r / 1000)))
                count += 1000

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
env = gym.make('FrozenLake-v0')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
#Create an intelligent agent
agent = Agent()

#Play the game
agent.play(env)
graph(agent.average_reward_for_each_episode)




