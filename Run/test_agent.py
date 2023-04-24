from Run.run_episode import run_episode
from tqdm import tqdm 

def test_agent(env, agents, N_episodes):
    res = {'player_0':0,'player_1':0,'equal':0}
    for _ in tqdm(range(N_episodes)):
        env.reset()
        env, agents = run_episode(env, agents, display=False)
        #print(env.rewards)
        if env.rewards['player_0']==1:
            res['player_0'] += 1
        elif env.rewards['player_1']==1:
            res['player_1'] += 1
        else:
            res['equal'] += 1
    print(res)
    return res, env, agents