from Run.run_episode import run_episode
from Run.run_episode import clean_q_values

def run_N_episodes(env, agents, N_episodes=1, clean = True):
    for i in range(N_episodes):
        env.reset()
        if clean:
            env, agents = run_episode(env, agents, clean, display=False)
        print(str(i/N_episodes*100)+" %")
    return env, agents