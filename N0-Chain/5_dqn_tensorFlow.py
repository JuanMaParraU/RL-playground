import gym
import numpy as np #
from tabulate import tabulate
import matplotlib.pyplot as plt
import tensorflow as tf
import platform
class Agent():
    def __init__(self):
        self.learning_rate = 0.05 # learning rate towards a target Q value
        self.neural_network = NeuralNetwork(self.learning_rate)
        self.discount_factor = 0.95 # for reward in the future
        self.epsilon = 0.5 #exploration rate
        self.decay_factor = 0.999
        self.average_reward_for_each_episode = []

    def play(self,env, episodes=2000 ):
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
                    # Choose randomly probability for explore, otherwise choose action with the highest reward
                    if self.__with_probability(self.epsilon):
                        action = self.__get_action_randomly(env)
                    else:
                        action = self.__get_action_highest_reward(state)
                    # Perform ACTION
                    new_state, reward, end_game, _ = env.step(action)
                    total_reward += reward
                    #Train NN. Two values for 2 outputs for 2 states, 1 at the time.
                    target_output = self.neural_network.predict_expected_rewards_for_each_actions(state)
                    target_output[action] = reward + self.discount_factor * self._get_expected_reward_in_next_state(new_state)
                    #train to reduce the error. Changing it slowly towards its target
                    self.neural_network.train(state,target_output)

                    #Update state
                    state =  new_state
                #Store de average reward
                self.average_reward_for_each_episode.append(total_reward / 1000)

                print(tabulate(self.neural_network.results(), showindex="always",
                               headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))

    def __with_probability(self,prob):
        return np.random.random() < prob

    def __get_action_randomly(self,env):
        return env.action_space.sample()

    def __get_action_highest_reward(self,state):
        return np.argmax(self.neural_network.predict_expected_rewards_for_each_actions(state))

    def _get_expected_reward_in_next_state(self,next_state): #access to next state assuming next action is optimal based on Q
        return np.max(self.neural_network.predict_expected_rewards_for_each_actions(next_state))

class NeuralNetwork(tf.keras.models.Sequential):
    def __init__(self,learning_rate=0.05):
        super().__init__()
        #input layer
        self.add(tf.keras.layers.InputLayer(batch_input_shape=(1,5)))
        #hidden layer
        self.add(tf.keras.layers.Dense(10,activation='sigmoid'))
        #output layer
        self.add(tf.keras.layers.Dense(2,activation='linear'))
        #keras as compile function
        self.compile(loss='mse',optimizer=tf.optimizers.Adam(learning_rate=learning_rate))
    #covertate to input signal and reshape  the target output (desired output)
    def train(self,state,target_output):
        input_signal = self.__convert_state_to_NN(state)
        target_output = target_output.reshape(-1,2)
        #fit ?
        self.fit(input_signal,target_output,epochs=1,verbose=0)

    #take state as input and output predicted Q_values for each actions
    def predict_expected_rewards_for_each_actions(self,state):
        input_signal = self.__convert_state_to_NN(state)
        #predict by keras
        return self.predict(input_signal)[0]

    def results(self):
        results= []
        for state in range(0,5):
            results.append(self.predict_expected_rewards_for_each_actions(state))
        return results

    #Discrite to continuous to set the input to the NN. State to a one hot vector: state 3 -> [[0,0,0,1,0]] extra bracket for 1 batch size
    def __convert_state_to_NN(self,state):
        input_signal = np.zeros((1,5))
        input_signal[0,state] = 1
        return input_signal


def graph(average_reward):
    plt.plot(average_reward)
    plt.title('Performance over time')
    plt.ylabel('Average reward')
    plt.xlabel('Episode')
    plt.show()

#Create the Nchain-v0 environment
#activate tensorENV environment
print('activate tensorENV environment')

env = gym.make('NChain-v0')
#Create an intelligent agent
agent = Agent()
platform.platform()

#Play the game
agent.play(env)
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
graph(agent.average_reward_for_each_episode)




