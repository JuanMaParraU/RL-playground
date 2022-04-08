import gym

class Agent():
    def __init__(self):
        pass
    def play(self,env):
        # Reset th env before play begins
        env.reset()
        # Play the game 1000 steps
        end_game = False
        while not end_game:
            # Choose action 0 or 1 with 50% prob
            action = env.action_space.sample()
            # Perform ACTION
            state, reward, end_game, _ = env.step(action)
            # print the lateste infor terminal
            print("Took action {}, entered state {} and received reward {}.".format(action, state, reward))

#Create the Nchain-v0 environment
env = gym.make('NChain-v0')

#Create an intelligent agent
agent = Agent()

#Play the game
agent.play(env)





