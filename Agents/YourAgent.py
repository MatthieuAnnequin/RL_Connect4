class YourAgent: 
    """
        An agent that only moves down or right, depending on its position on the grid
    """
    def __init__(self, name = 'Yourname'):
        self.name = name
        
    def get_action(self,env):
        action = int(input("Enter your value: "))
        return action
    
    def update(self, obs, action, reward, terminated, next_obs):
        pass