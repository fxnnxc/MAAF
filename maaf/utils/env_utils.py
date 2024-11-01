import gym 
import importlib
import supersuit as ss
from smac.env.pettingzoo import StarCraft2PZEnv
from smac.env import StarCraft2Env

def make_env(env_id, num_envs=1, max_cycles=125, render_mode=None, **kwargs):
    
    if env_id == "mpe.tag_v2":
        kwargs['num_adversaries'] = 2 
        env_id = 'mpe.simple_tag_v3'
        
    if env_id == "mpe.tag_v3":
        kwargs['num_adversaries'] = 3 
        env_id = 'mpe.simple_tag_v3'
        
    if env_id == "mpe.tag_v4":
        kwargs['num_adversaries'] = 4
        env_id = 'mpe.simple_tag_v3'
        
    if env_id == "mpe.tag_v15":
        kwargs['num_adversaries'] = 15
        kwargs['num_obstacles'] = 1
        kwargs['num_good'] = 2
        env_id = 'mpe.simple_tag_v3'
        
    # make env 
    if 'mpe' in env_id:
        raw_env = importlib.import_module(f"pettingzoo.{env_id}",).parallel_env(max_cycles=max_cycles, render_mode=render_mode, **kwargs )    
        
        # envs = gym.wrappers.RecordEpisodeStatistics(raw_env)
        envs = ss.pad_observations_v0(raw_env)
        envs = ss.pad_action_space_v0(envs)
        envs = ss.pettingzoo_env_to_vec_env_v1(envs)
        
        envs = ss.concat_vec_envs_v1(
            envs, num_envs, num_cpus=0, base_class="gymnasium"
        )
    if "smac" in env_id:
        env_id = env_id.split(".")[-1]
        smac_env = StarCraft2Env(map_name=env_id)
        envs = SMACWarpperEnvs(smac_env)
        raw_env = SMACWarpperRaw(smac_env)  
    return envs, raw_env    

import numpy as np 
import gym 
class SMACWarpperRaw():
    def __init__(self, smac_env):
        self.smac_env = smac_env
        self.env_info = smac_env.get_env_info()
        self.num_agents = self.env_info['n_agents']
        self.possible_agents = [f"agent_{i}" for i in range(self.env_info['n_agents'])]
        self.obs_shape = self.env_info['obs_shape']
        self.state_shape = self.env_info['state_shape']
        self.n_actions = self.env_info['n_actions']
        self.obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_shape,))
        self.act_space = gym.spaces.Discrete(self.n_actions)
    
    def step(self, actions):
        # actions = []
        # for agent_id in self.possible_agents:
        #     avail_actions = smac_env.get_avail_agent_actions(agent_id)
        #     avail_actions_ind = np.nonzero(avail_actions)[0]
        #     action = np.random.choice(avail_actions_ind)
        #     actions.append(action)
        reward, terminated, _ = self.smac_env.step(actions)
        next_obs = self.smac_env.get_obs()
        rewards = np.array([reward for _ in range(len(self.possible_agents))])
        term = [True if terminated else False for _ in range(len(self.possible_agents))]
        info = [{} for _ in range(len(self.possible_agents))]
        if terminated:
            self.reset()
        return next_obs, rewards, term, term, info
        
    def reset(self, seed=0):
        self.smac_env.reset()
        next_obs = self.smac_env.get_obs()
        return next_obs, [{} for i in range(len(self.possible_agents))] 
    
    def observation_space(self, agent_id):
        return self.obs_space
    def action_space(self, agent_id):
        return self.act_space

class SMACWarpperEnvs():
    def __init__(self, smac_env):
        self.smac_env = smac_env
        self.env_info = smac_env.get_env_info()
        self.num_agents = self.env_info['n_agents']
        self.possible_agents = [f"agent_{i}" for i in range(self.env_info['n_agents'])]
        self.obs_shape = self.env_info['obs_shape']
        self.state_shape = self.env_info['state_shape']
        self.n_actions = self.env_info['n_actions']
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_shape,))
        self.action_space = gym.spaces.Discrete(self.n_actions)
    
    def step(self, actions):
        # print(actions)
        # actions = []
        # for i in range(len(self.possible_agents)):
        #     avail_actions = self.smac_env.get_avail_agent_actions(i)
        #     avail_actions_ind = np.nonzero(avail_actions)[0]
        #     action = np.random.choice(avail_actions_ind)
        #     actions.append(action)
        reward, terminated, _ = self.smac_env.step(actions)
        next_obs = self.smac_env.get_obs()
        rewards = np.array([reward for _ in range(len(self.possible_agents))])
        term = [True if terminated else False for _ in range(len(self.possible_agents))]
        info = [{} for _ in range(len(self.possible_agents))]
        if terminated:
            self.reset()
        next_obs = np.stack(next_obs)
        return next_obs, rewards, term, term, info
        
    def reset(self, seed=0):
        self.smac_env.reset()
        next_obs = self.smac_env.get_obs()
        next_obs = np.stack(next_obs)
        return next_obs, [{} for i in range(len(self.possible_agents))] 
