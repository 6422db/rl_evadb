import torch
from torch import nn
import gymnasium as gym
from tianshou.data import(
    Batch,
    to_numpy
)
import tianshou as ts
import pandas as pd
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb as conActor
from tianshou.utils.net.discrete import Actor as disActor
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete

class RLNetwork(nn.Module):
    def __init__(self, env_name, model_path=None, algo="PPO"):
        super().__init__()
        env = gym.make(env_name)
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.algo = algo
        if isinstance(env.action_space, (Discrete, MultiDiscrete, MultiBinary)):
            action_type = "discrete"
        else:
            action_type = "continuous"

        if algo == "DQN":
            self.model = Net(self.state_shape, self.action_shape, [128,128,128])
        elif algo == "PPO":
            self.net = Net(self.state_shape, hidden_sizes=[128,128,128])
            self.model = disActor(self.net, self.action_shape) if action_type == "discrete" else conActor(self.net, self.action_shape, unbounded=True)
            self.dist = disDist if action_type == "discrete" else conDist
        else:
            raise NotImplementedError
        
        if model_path is not None:
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)

        self.env = env_name

    def forward(self, obs, state=None, info={}):
        assert(obs['name'][0] == self.env)
        return self.collect(gym.make(self.env))    

    def collect(self, env, render=False, exploration_noise=False):
        data = Batch(
            obs={},
            act={},
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={},
        )

        if self.algo == "DQN":
            policy = ts.policy.DQNPolicy(
                model=self.model,
                optim=None,
                action_space=env.action_space,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320
            )
        else:
            policy = ts.policy.PPOPolicy(self.model, None, None, self.dist, action_space=env.action_space, deterministic_eval=True)

        obs, info = env.reset()
        data.obs = [obs]

        rewards = []
        actions = []
        obs = []
        dones = []

        while True:
            last_state = data.policy.pop("hidden_state", None)

            with torch.no_grad():
                result = policy(data, last_state)

            temp_policy = result.get("policy", Batch())
            assert isinstance(temp_policy, Batch)
            state = result.get("state", None)
            if state is not None:
                temp_policy.hidden_state = state
            act = to_numpy(result.act)[0]
            if exploration_noise:
                act = policy.exploration_noise(act, data)
            data.update(policy=temp_policy, act=act)

            action_remap = policy.map_action(data.act)

            obs_next, rew, terminated, truncated, info = env.step(
                action_remap,
            )
            done = terminated or truncated

            rewards.append(rew)
            dones.append(done)
            obs.append(obs_next)
            actions.append(action_remap)
            data.update(
                obs_next=[obs_next],
                rew=rew,
                done=done,
                info=info,
            )

            if render:
                env.render()

            if done:
                break
            data.obs = data.obs_next

        df = {
            "reward": rewards,
            "done": dones,
            "observation": obs,
            "action": actions
        }
        return pd.DataFrame(df)

def conDist(*logits):
    return torch.distributions.Independent(torch.distributions.Normal(*logits), 1)

def disDist(p):
    return torch.distributions.Categorical(logits=p)