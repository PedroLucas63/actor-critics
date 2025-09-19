# Importação de bibliotecas
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import time
import os
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Definindo as configurações
configs = {
   # Gerais
   'env_name': 'HalfCheetah-v5',
   'file_name': 'sac',
   'envs': 1,
   'seed': 5,
   'cuda': True,
   'track': True,
   'save_video': True,
   'torch_deterministic': True,
   'test_every': 1000,
   'test_episodes': 10,
   
   # SAC
   'gamma': 0.99,
   'tau': 0.005,
   'actor_lr': 10e-3,
   'critic_lr': 10e-3,
   'batch_size': 100,
   'buffer_size': 100000,
   'time_steps': 100000,
   'learning_start': 10000,
   'update_every': 1,
   'grad_clip': 0.5,
   'alpha': 0.01,
   'alpha_tunning': False,
}

# Run name
run_name = f"{configs['file_name']}_{configs['env_name']}_{time.strftime('%Y%m%d-%H%M%S')}"

# Definindo a função de ambiente
def make_env(env_name, seed, run_name, idx, save_video):
   def thunk():
      if save_video and idx == 0:
         env = gym.make(env_name, render_mode='rgb_array')
         env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger=lambda x: x % 100 == 0)
      else:
         env = gym.make(env_name)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
   return thunk

# Criando os agentes
class Critic(nn.Module):
   def __init__(self, state_dim, action_dim):
      super(Critic, self).__init__()
      self.sequence = nn.Sequential(
         nn.Linear(state_dim + action_dim, 256),
         nn.ReLU(),
         nn.Linear(256, 256),
         nn.ReLU(),
         nn.Linear(256, 1)
      )
   def forward(self, state, action):
      state = state.float()
      return self.sequence(torch.cat([state, action], 1))

class Actor(nn.Module):
   MAX_LOG_STD = 2
   MIN_LOG_STD = -20
   
   def __init__(self, state_dim, action_dim, min_action, max_action):
      super(Actor, self).__init__()
      self.encoder = nn.Sequential(
         nn.Linear(state_dim, 256),
         nn.ReLU(),
         nn.Linear(256, 256),
         nn.ReLU()
      )
      self.mean_network = nn.Linear(256, action_dim)
      self.log_std_network = nn.Linear(256, action_dim)
      
      # Normalização das ações
      min_action = np.array(min_action, dtype=np.float32)
      max_action = np.array(max_action, dtype=np.float32)
      self.register_buffer(
         "action_scale", 
         torch.tensor((max_action - min_action) / 2.0, # Metade das ações
         dtype=torch.float32)
      )
      self.register_buffer(
         "action_bias", 
         torch.tensor((max_action + min_action) / 2.0, # Ponto médio das ações
         dtype=torch.float32)
      )

   def forward(self, state):
      state = state.float()
      encoded = self.encoder(state)
      
      mean = self.mean_network(encoded)
      log_std = self.log_std_network(encoded)
      log_std = torch.clamp(log_std, self.MIN_LOG_STD, self.MAX_LOG_STD)
      
      return mean, log_std

   def get_action(self, state, train=True):
      # Pegando a média e variância
      mean, log_std = self.forward(state)
      std = log_std.exp()
      if train:
         # Criando a distribuição normal
         normal = Normal(mean, std)
         z = normal.rsample() # Amostragem do Z sem perda da diferenciabilidade
         y = torch.tanh(z) # Aplicando a função tanh para garantir que a ação esteja entre -1 e 1
         action = y * self.action_scale + self.action_bias # Reescalando a ação
         
         # Correção da ação
         log_prob = normal.log_prob(z) # Calculando o log_prob
         log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6) # Correção do log_prob
         log_prob = log_prob.sum(1, keepdim=True) # Somando as probabilidades logarítmicas
         
         return action, log_prob
      else:
         y = torch.tanh(mean)
         action = y * self.action_scale + self.action_bias
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
         False,
      ) for i in range(configs['envs'])])
   
   # Ambiente para testes
   test_env = make_env(
      configs['env_name'], 
      configs['seed'] + 100, 
      run_name, 
      0, 
      configs['save_video'],
   )()

   # Resetando os ambientes
   obs, _ = envs.reset()
   
   # Pegando as dimensões
   action_dim = envs.single_action_space.shape[0]
   state_dim = envs.single_observation_space.shape[0]
   
   # Pegando as ações maiores e menores
   action_max = envs.single_action_space.high
   action_min = envs.single_action_space.low
   
   # Criando os agentes
   qf1 = Critic(state_dim, action_dim).to(device)
   qf2 = Critic(state_dim, action_dim).to(device)
   pi = Actor(state_dim, action_dim, action_min, action_max).to(device)
   target_qf1 = Critic(state_dim, action_dim).to(device)
   target_qf2 = Critic(state_dim, action_dim).to(device)
   target_qf1.load_state_dict(qf1.state_dict())
   target_qf2.load_state_dict(qf2.state_dict())
   
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
   
   # Definindo o alpha para tunning
   if configs['alpha_tunning']:
      target_entropy = -np.prod(envs.single_action_space.shape).item()
      log_alpha = torch.zeros(1, requires_grad=True, device=device)
      alpha_optimizer = optim.Adam([log_alpha], lr=configs['actor_lr'])
      alpha = log_alpha.exp().item()
   else:
      alpha = configs['alpha']
   
   # Definindo o progresso
   progress = tqdm(range(configs['time_steps']), desc='Steps')
   # Inicializa com todos os campos
   postfix_data = {
      "EpisodicReturn": None,
      "TestEpisodicReturn": None
   }
   progress.set_postfix(postfix_data)
   
   for step in progress:
      # Definindo que é um treino
      pi.train()
      
      # Pegando a ação
      if step < configs['learning_start']:
         action = envs.action_space.sample()
      else:
         with torch.no_grad():
            action, _ = pi.get_action(torch.tensor(obs, device=device))
            action = action.cpu().numpy()
      
      # Interagindo com o ambiente
      next_obs, reward, terminated, truncated, info = envs.step(action)
      done = np.logical_or(terminated, truncated)
      
      # Adicionando ao buffer
      buffer.add(obs, next_obs, action, reward, done, [info])

      # Registrando possíveis resultados (episódios que terminaram)
      if "episode" in info and configs['track']:
         for reward, length in zip(info["episode"]["r"], info["episode"]["l"]):
            if reward and length:               
               # Log to console
               postfix_data['EpisodicReturn'] = reward
               progress.set_postfix(postfix_data)
               
               # Log to TensorBoard
               if configs['track']:
                  writer.add_scalar("charts/episodic_return", reward, step)
                  writer.add_scalar("charts/episodic_length", length, step)

      # Atualizando o estado
      obs = next_obs
      
      # Atualizando os modelos
      if step % configs['update_every'] == 0 and step > configs['learning_start']:
         for _ in range(configs['update_every']):
            batch = buffer.sample(configs['batch_size'])

            # Calculando os Q target
            with torch.no_grad():
               # Pega a ação e o log probabilidade
               next_action, next_log_prob = pi.get_action(batch.next_observations)
               next_log_prob = next_log_prob.flatten()
               
               # Calcula os Q valores alvos
               target_q1_value = target_qf1(batch.next_observations, next_action).flatten()
               target_q2_value = target_qf2(batch.next_observations, next_action).flatten()
               
               # Pega a menor Q para reduzir o viés de superestimação
               target_q_value = torch.min(target_q1_value, target_q2_value) - alpha * next_log_prob
               
               # Define o alvo do Q
               q_target = batch.rewards.flatten() + (1 - batch.dones.flatten()) * configs['gamma'] * target_q_value
            
            # Pega os valores atuais do Q
            q1_value = qf1(batch.observations, batch.actions).flatten()
            q2_value = qf2(batch.observations, batch.actions).flatten()
            
            # Calcula a loss do Q
            qf1_loss = F.mse_loss(q1_value, q_target)
            qf2_loss = F.mse_loss(q2_value, q_target)
            qf_loss = qf1_loss + qf2_loss
            
            # Atualizando a função Q
            qf_optimizer.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), configs['grad_clip'])
            qf_optimizer.step()
            
            # Pegando a próxima ação e o log probabilidade
            new_action, log_prob = pi.get_action(batch.observations)
            
            # Calculando o Q para a nova ação
            q1_new_action = qf1(batch.observations, new_action).flatten()
            q2_new_action = qf2(batch.observations, new_action).flatten()
            
            # Pega a menor Q para reduzir o viés de superestimação
            q_new_action = torch.min(q1_new_action, q2_new_action)
            
            # Calculando a loss da política
            pi_loss = (alpha * log_prob - q_new_action).mean()
            
            # Atualizando a política
            pi_optimizer.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(pi.parameters(), configs['grad_clip'])
            pi_optimizer.step()
            
            # Atualizando o alpha para tunning
            if configs['alpha_tunning']:
               # Calculando a loss do alpha
               alpha_loss = -(log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
               
               # Atualizando o alpha
               alpha_optimizer.zero_grad()
               alpha_loss.backward()
               alpha_optimizer.step()
               alpha = log_alpha.exp().item()
            
            # Atualizando os modelos target
            for param, target_param in zip(qf1.parameters(), target_qf1.parameters()):
               target_param.data.copy_((1 - configs['tau']) * target_param.data + configs['tau'] * param.data)
               
            for param, target_param in zip(qf2.parameters(), target_qf2.parameters()):
               target_param.data.copy_((1 - configs['tau']) * target_param.data + configs['tau'] * param.data)
            
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
               writer.add_scalar("charts/log_prob", log_prob.mean().item(), step)
               writer.add_scalar("charts/alpha", alpha, step)

      # Testando a política
      if step % configs['test_every'] == 0:
         pi.eval()
         rewards = []
         for _ in range(configs['test_episodes']):
            test_obs, _ = test_env.reset()
            test_done = False
            test_episode_return = 0
            
            while not test_done:
               with torch.no_grad():
                  test_action = pi.get_action(torch.tensor(test_obs, device=device), train=False)
                  test_action = test_action.cpu().numpy()
               test_obs, test_reward, test_terminated, test_truncated, _ = test_env.step(test_action)
               test_done = test_terminated or test_truncated
               test_episode_return += test_reward
            
            # Guarda o retorno do episódio
            rewards.append(test_episode_return)
            
         # Calcula o retorno médio
         avg_reward = np.mean(rewards)
         
         # Atualizando o log e writer
         postfix_data['TestEpisodicReturn'] = avg_reward
         progress.set_postfix(postfix_data)
         if configs['track']:
            writer.add_scalar("charts/test_episodic_return",avg_reward , step)
   # Fechando ambientes
   envs.close()
   test_env.close()

# Rodando a função principal
if __name__ == "__main__":
   main()
