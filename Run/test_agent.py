from Run.run_episode import run_episode

def test_agent(env, agents, N_episodes):
    res = {'player_0':0,'player_1':0,'equal':0}
    for i in range(N_episodes):
        env.reset()
        env, agents = run_episode(env, agents, display=False)
        #print(env.rewards)
        if env.rewards['player_0']==1:
            res['player_0'] += 1
        elif env.rewards['player_1']==1:
            res['player_1'] += 1
        else:
            res['equal'] += 1
        print(str(i/N_episodes*100)+" %")
    print(res)
    return env, agents