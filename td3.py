# Possui uma função Q Valor para estimar o quão bem são as ações nos estados
# Utiliza de uma Qtarget para avaliar a função Q
# A política usa a função Q para aprender.
# L(Q) = (Q(s, a) - (R + lambda * (1 - done) * Qtarget(s', a')))^2
# L(policy) = -Q(s, policy(s))
# Add noise to exploration (mean-zero Gaussian noise)

# Importação de bibliotecas
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Definindo as configurações
configs = {
   # Gerais
   'env_name': 'HalfCheetah-v5',
   'file_name': 'td3',
   'envs': 1,
   'seed': 3,
   'cuda': True,
   'track': True,
   'save_video': True,
   'torch_deterministic': True,
   
   # DDPG
   'gamma': 0.99,
   'polyak': 0.999,
   'actor_lr': 1e-4,
   'critic_lr': 1e-4,
   'batch_size': 256,
   'buffer_size': 1000000,
   'time_steps': 1000000,
   'learning_start': 25000,
   'update_every': 50,
   'exploration_noise': 0.5,
   'grad_clip': 0.5,
   'policy_delay': 2,
}

# Run name
run_name = f"{configs['file_name']}_{configs['env_name']}_{time.strftime('%Y%m%d-%H%M%S')}"

# Definindo a função de ambiente
def make_env(env_name, seed, run_name, idx, save_video, gamma):
   def thunk():
      if save_video and idx == 0:
         env = gym.make(env_name, render_mode='rgb_array')
         env = gym.wrappers.RecordVideo(env, f'videos/{run_name}')
      else:
         env = gym.make(env_name)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.NormalizeReward(env, gamma=gamma)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
   return thunk

# Criando os agentes
class Critic(nn.Module):
   def __init__(self, state_dim, action_dim):
      super(Critic, self).__init__()
      self.sequential = nn.Sequential(
         nn.Linear(state_dim + action_dim, 256),
         nn.ReLU(),
         nn.Linear(256, 256),
         nn.ReLU(),
         nn.Linear(256, 1)
      )
   def forward(self, state, action):
      state = state.float()
      return self.sequential(torch.cat([state, action], dim=1))

class Actor(nn.Module):
   def __init__(self, state_dim, action_dim, action_min, action_max):
      super(Actor, self).__init__()
      self.mean = nn.Sequential(
         nn.Linear(state_dim, 256),
         nn.ReLU(),
         nn.Linear(256, 256),
         nn.ReLU(),
         nn.Linear(256, action_dim),
         nn.Tanh()
      )
      # action rescaling
      # ensure action_min/max are numpy arrays or scalars
      action_min = np.array(action_min, dtype=np.float32)
      action_max = np.array(action_max, dtype=np.float32)
      # store as buffers (torch tensors)
      self.register_buffer(
         "action_scale", torch.tensor((action_max - action_min) / 2.0, dtype=torch.float32)
      )
      self.register_buffer(
         "action_bias", torch.tensor((action_max + action_min) / 2.0, dtype=torch.float32)
      )
   def forward(self, state):
      state = state.float()
      action = self.mean(state)
      action = action * self.action_scale + self.action_bias
      return action

# Função principal
def main():
   # Criando a pasta de vídeos se necessário
   if configs['save_video'] and not os.path.exists('videos'):
      os.makedirs('videos')
   
   # Adicionando o tracking
   if configs['track']:
      if not os.path.exists('runs'):
         os.makedirs('runs')
      writer = SummaryWriter(f"runs/{run_name}")
      writer.add_hparams(configs, {})
      
   # Criando a pasta de modelos
   if not os.path.exists('models'):
      os.makedirs('models')
   if not os.path.exists(f'models/{run_name}'):
      os.makedirs(f'models/{run_name}')
   # Definindo as melhores losses
   best_qf_loss = float('inf')
   best_pi_loss = float('inf')
   
   # Configurando seeds
   torch.manual_seed(configs['seed'])
   np.random.seed(configs['seed'])
   random.seed(configs['seed'])
   torch.backends.cudnn.deterministic = configs['torch_deterministic']
   
   # Pegando o device
   device = torch.device('cuda' if configs['cuda'] and torch.cuda.is_available() else 'cpu')
   
   # Criando os ambientes
   envs = gym.vector.SyncVectorEnv(
      [make_env(
         configs['env_name'], 
         configs['seed'], 
         run_name, 
         i, 
         configs['save_video'],
         configs["gamma"]
      ) for i in range(configs['envs'])])
   
   # Resetando os ambientes
   obs, _ = envs.reset()
   
   # Pegando as dimensões
   action_dim = envs.single_action_space.shape[0]
   state_dim = envs.single_observation_space.shape[0]
   
   # Pegando as ações maiores e menores
   action_max = envs.single_action_space.high
   action_min = envs.single_action_space.low
   action_range = (action_max - action_min) / 2.0
   
   # Criando as redes
   qf1 = Critic(state_dim, action_dim).to(device)
   qf2 = Critic(state_dim, action_dim).to(device)
   pi = Actor(state_dim, action_dim, action_min, action_max).to(device)
   qf1_target = Critic(state_dim, action_dim).to(device)
   qf2_target = Critic(state_dim, action_dim).to(device)
   pi_target = Actor(state_dim, action_dim, action_min, action_max).to(device)
   qf1_target.load_state_dict(qf1.state_dict())
   qf2_target.load_state_dict(qf2.state_dict())
   pi_target.load_state_dict(pi.state_dict())
   
   # Criando os optimizadores
   qf_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=configs['critic_lr'])
   pi_optimizer = optim.Adam(pi.parameters(), lr=configs['actor_lr'])
   
   # Criando o buffer
   buffer = ReplayBuffer(
      configs['buffer_size'],
      envs.single_observation_space,
      envs.single_action_space,
      n_envs=configs['envs'],
      device=device
   )
   
   # Definindo o progresso
   progress = tqdm(range(configs['time_steps']), desc='Steps')
   
   # Definindo o loop
   for step in progress:
      # Coleta dos dados
      if step < configs['learning_start']:
         action = envs.action_space.sample()
      else:
         with torch.no_grad():   
            action = pi(torch.Tensor(obs).to(device))
            action = action.cpu().numpy()
            noise = np.random.normal(0, configs['exploration_noise'] * action_range, size=action.shape) 
            action = np.clip(action + noise, action_min, action_max)
                  
      next_obs, reward, terminated, truncated, infos = envs.step(action)
      done = np.logical_or(terminated, truncated)
      
      # Registrando os dados
      buffer.add(obs, next_obs, action, reward, done, [infos])
      
      # Registrando possíveis resultados (episódios que terminaram)
      if "episode" in infos and configs['track']:
         for reward, length in zip(infos["episode"]["r"], infos["episode"]["l"]):
            if reward and length:               
               # Log to console
               progress.set_description(f"Global Step: {step}, Episodic Return: {reward:.2f}")
               
               # Log to TensorBoard
               writer.add_scalar("charts/episodic_return", reward, step)
               writer.add_scalar("charts/episodic_length", length, step)
      
      # Atualizando a rede
      if step >= configs['learning_start'] and step % configs['update_every'] == 0:
         for update in range(configs['update_every']):
            batch = buffer.sample(configs['batch_size'])

            # Atualização da rede Q
            with torch.no_grad():
               next_action = pi_target(batch.next_observations)
               noise = torch.normal(
                  mean=0.0,
                  std=configs['exploration_noise'],
                  size=next_action.shape,
                  device=device) * action_range
               next_action = next_action + noise
               next_action = torch.clamp(
                  next_action, 
                  torch.Tensor(action_min).to(device), 
                  torch.Tensor(action_max).to(device))
               
               next_q1 = qf1_target(batch.next_observations, next_action).flatten()
               next_q2 = qf2_target(batch.next_observations, next_action).flatten()
               next_q = torch.min(next_q1, next_q2)
               
               target = batch.rewards.flatten() + configs['gamma'] * (1 - batch.dones.flatten()) * next_q
            
            current_q1 = qf1(batch.observations, batch.actions).flatten()
            current_q2 = qf2(batch.observations, batch.actions).flatten()
            
            qf1_loss = F.mse_loss(current_q1, target)
            qf2_loss = F.mse_loss(current_q2, target)
            qf_loss = qf1_loss + qf2_loss
            
            qf_optimizer.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), configs['grad_clip'])
            qf_optimizer.step()
            
            # Atualização da rede pi
            if update % configs['policy_delay'] == 0:
               pi_loss = -qf1(batch.observations, pi(batch.observations)).flatten().mean()
               pi_optimizer.zero_grad()
               pi_loss.backward()
               torch.nn.utils.clip_grad_norm_(pi.parameters(), configs['grad_clip'])
               pi_optimizer.step()
               
               # Atualizando as redes target
               for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                  target_param.data.copy_(configs['polyak'] * target_param.data + (1 - configs['polyak']) * param.data)
                  
               for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                  target_param.data.copy_(configs['polyak'] * target_param.data + (1 - configs['polyak']) * param.data)
               
               for param, target_param in zip(pi.parameters(), pi_target.parameters()):
                  target_param.data.copy_(configs['polyak'] * target_param.data + (1 - configs['polyak']) * param.data)
            
            # Atualizando o melhor modelo
            if qf_loss.item() < best_qf_loss:
               torch.save(qf1.state_dict(), f"models/{run_name}/qf1.pth")
               torch.save(qf2.state_dict(), f"models/{run_name}/qf2.pth")
               best_qf_loss = qf_loss.item()
            if pi_loss.item() < best_pi_loss:
               torch.save(pi.state_dict(), f"models/{run_name}/pi.pth")
               best_pi_loss = pi_loss.item()

         # Atualizando o writer
         if configs['track']:
            writer.add_scalar("losses/qf_loss", qf_loss.item(), step)
            writer.add_scalar("losses/pi_loss", pi_loss.item(), step)
# Execução
if __name__ == '__main__':
   main()