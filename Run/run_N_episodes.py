from Run.run_episode import run_episode

def run_N_episodes(env, agents, N_episodes=1):
    for i in range(N_episodes):
        env.reset()
        env, agents = run_episode(env, agents, display=False)
        print(str(i/N_episodes*100)+" %")
    return env, agents