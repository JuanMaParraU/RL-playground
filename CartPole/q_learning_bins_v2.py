import gym
import numpy as np #
import matplotlib.pyplot as plt

class Agent():
    def __init__(self,env):
        self.action_size = env.action_space.n
        print("Action size: ",self.action_size)
        self.learning_rate = 0.15 # learning rate towards a target Q value
        self.discount_factor = 0.95 # for reward in the future
        self.epsilon = 0.2 #exploration rate
        self.decay_factor = 0.999
        self.average_reward_for_each_episode = []


    def get_action(self,state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action

    def play(self, env, q_table, bins,episodes=5000):
        for epi in range(episodes):
            print("Episode {}".format(epi))
            state = Discrete(env.reset(), bins)
            #print(state)
            total_reward = 0
            # reduce exploration rate
            self.epsilon *= self.decay_factor
            end_game = False
            while not end_game:
                # Choose randomly if empty or probability for explore, otherwise choose action with the highest reward
                if self.__q_table_is_empty(q_table,state) or self.__with_probability(self.epsilon):
                    action = self.__get_action_randomly(env)
                else:
                    action = self.__get_action_highest_reward(q_table,state)
                # performing action
                new_state, reward, end_game, _ = env.step(action)
                total_reward += reward
                next_state = Discrete(new_state,bins)
                # Update the reward table with Bellman Equation - > Q Learning Eq
                q_table[state+(action,)] += self.learning_rate * (
                            reward + self.discount_factor * self._get_expected_reward_in_next_state(next_state) -
                            q_table[state+(action,)])
                state = next_state
                # display the graph
                env.render()
            self.average_reward_for_each_episode.append(total_reward)

    def __q_table_is_empty(self,q_table,state):
        return np.sum(q_table[tuple(map(int, state))]) == 0

    def __with_probability(self,prob):
        return np.random.random() < prob

    def __get_action_randomly(self,env):
        return env.action_space.sample()

    def __get_action_highest_reward(self,q_table,state):
        return np.argmax(q_table[tuple(map(int, state))])

    def _get_expected_reward_in_next_state(self,next_state): #access to next state assuming next action is optimal based on Q
        return np.max(q_table[tuple(map(int, next_state))])

def Qtable(state_space, action_space, bin_size=30):
    bins = [np.linspace(-4.8, 4.8, bin_size),
                np.linspace(-4, 4, bin_size),
                np.linspace(-0.418, 0.418, bin_size),
                np.linspace(-4, 4, bin_size)]

    q_table = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))
    return q_table, bins

def Discrete(state, bins):
    index = []
    for i in range(len(state)): index.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(index)

def graph(average_reward):
    plt.plot(average_reward)
    plt.title('Performance over time')
    plt.ylabel('Average reward')
    plt.xlabel('Episode')
    plt.show()

#Create the CartPole environment
env = gym.make('CartPole-v1')
print("Action space: {}".format(env.action_space))
print("Observation space: {}".format(env.observation_space))

#Create an intelligent agent
agent = Agent(env)
q_table, bins = Qtable(len(env.observation_space.low), env.action_space.n)
#play the game
agent.play(env,q_table,bins)
graph(agent.average_reward_for_each_episode)

w