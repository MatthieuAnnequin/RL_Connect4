class YourAgent: 
    """
        An agent that only moves down or right, depending on its position on the grid
    """
    def __init__(self, action_space, observation_space, name = 'Yourname'):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name
        
    def get_action(self,env, obs):
        action = int(input("Enter your value: "))
        return action
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass