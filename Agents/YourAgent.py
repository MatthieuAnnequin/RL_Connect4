class YourAgent: 
    """
        An agent that only moves down or right, depending on its position on the grid
    """
    def _init_(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        
    def get_action(self,env, obs):
        action = int(input("Enter your value: "))
        return action
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass