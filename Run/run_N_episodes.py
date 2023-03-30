from Run.run_episode import run_episode
from tqdm import tqdm

def run_N_episodes(env, agents, N_episodes=1):
    for i in tqdm(range(N_episodes)):
        env.reset()
        env, agents = run_episode(env, agents, display=False)
        # if (i/N_episodes*100 % 1) == 0:
        #     print(str(i/N_episodes*100)+" %")
    return env, agents