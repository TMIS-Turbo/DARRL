import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
device = torch.device("cpu")

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-10.0, max_log_std=10.0):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CostNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(CostNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.cost_buf[self.ptr] = cost
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(device),
                    done=torch.Tensor(self.done_buf[idxs]).to(device),
                    cost=torch.Tensor(self.cost_buf[idxs]).to(device))

class Agent():
        """
        An implementation of Soft Actor-Critic (SAC), Automatic entropy adjustment SAC (ASAC)
        """

        def __init__(self,
                     state_dim,
                     action_dim,
                     action_limit,
                     hidden_sizes,
                     steps=0,
                     gamma=0.99,
                     alpha=0.1,
                     beta=0.1,
                     entropic_index=1.0,
                     buffer_size=int(1e6),
                     batch_size=128,
                     actor_lr=1e-4,
                     qf_lr=1e-3,
                     alpha_lr=1e-4,
                     target_c_alpha=2.0,
                     target_c_beta=2.0
                     ):
            super(Agent, self).__init__()

            self.obs_dim = state_dim
            self.act_dim = action_dim
            self.act_limit = action_limit
            self.hidden_sizes = hidden_sizes
            self.steps = steps
            self.gamma = gamma
            self.alpha = alpha
            self.beta = beta
            self.q = entropic_index
            self.buffer_size = buffer_size
            self.batch_size = batch_size
            self.actor_lr = actor_lr
            self.qf_lr = qf_lr
            self.alpha_lr = alpha_lr

            # Main network
            self.actor = ActorNet(self.obs_dim, self.act_dim, min_log_std=-5.0, max_log_std=2.0).to(device)
            self.qf1 = CriticNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)
            self.qf2 = CriticNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)
            self.cf1 = CostNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)
            self.cf2 = CostNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)

            # Target network
            self.qf1_target = CriticNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)
            self.qf2_target = CriticNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)
            self.cf1_target = CriticNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)
            self.cf2_target = CriticNet(self.obs_dim, self.act_dim, self.hidden_sizes).to(device)

            # Initialize target parameters to match main parameters
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
            self.cf1_target.load_state_dict(self.cf1.state_dict())
            self.cf2_target.load_state_dict(self.cf2.state_dict())

            # Create optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=self.qf_lr)
            self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=self.qf_lr)
            self.cf1_optimizer = optim.Adam(self.cf1.parameters(), lr=self.qf_lr)
            self.cf2_optimizer = optim.Adam(self.cf2.parameters(), lr=self.qf_lr)

            # Experience buffer
            self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)

            self.target_c_alpha = target_c_alpha
            self.target_c_beta = target_c_beta
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.log_beta = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.beta_optimizer = optim.Adam([self.log_beta], lr=self.alpha_lr)

        def train_model(self):
            batch = self.replay_buffer.sample(self.batch_size)
            obs1 = batch['obs1']
            obs2 = batch['obs2']
            acts = batch['acts']
            rews = batch['rews']
            done = batch['done']
            cost = batch['cost']

            mu, std, pi = self.actor(obs1)
            _, _, next_pi = self.actor(obs2)

            q1 = self.qf1(obs1, acts).squeeze(1)
            q2 = self.qf2(obs1, acts).squeeze(1)
            cf1 = self.cf1(obs1, acts).squeeze(1)
            cf2 = self.cf2(obs1, acts).squeeze(1)

            min_q_next_pi = torch.min(self.qf1_target(obs2, next_pi), self.qf2_target(obs2, next_pi)).squeeze(1).to(device)

            # Targets for Q and V regression
            v_backup = min_q_next_pi
            q_backup = rews + self.gamma * (1 - done) * v_backup
            q_backup.to(device)
            c_backup = cost + 0.5 * self.gamma * done * (
                        self.cf1_target(obs2, next_pi) + self.cf2_target(obs2, next_pi)).squeeze(1)
            c_backup.to(device)

            # Critic and cost losses
            qf1_loss = F.mse_loss(q1, q_backup.detach())
            qf2_loss = F.mse_loss(q2, q_backup.detach())
            cf1_loss = F.mse_loss(cf1, c_backup.detach())
            cf2_loss = F.mse_loss(cf2, c_backup.detach())

            # Update two Q network parameter
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            self.cf1_optimizer.zero_grad()
            cf1_loss.backward()
            self.cf1_optimizer.step()

            self.cf2_optimizer.zero_grad()
            cf2_loss.backward()
            self.cf2_optimizer.step()

            # Actor loss
            min_q_pi = torch.min(self.qf1(obs1, pi), self.qf2(obs1, pi)).squeeze(1).to(device)
            cost_pert, a_cost = self.attacker_constraint(obs1)
            actor_loss = (self.alpha * a_cost + self.beta * cost_pert - min_q_pi).mean()

            # Update actor network parameter
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = (self.log_alpha * (self.target_c_alpha - a_cost).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            beta_loss = (self.log_beta * (self.target_c_beta - cost_pert).detach()).mean()
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()
            self.beta = self.log_beta.exp()

            # Polyak averaging for target parameter
            self.soft_target_update(self.qf1, self.qf1_target)
            self.soft_target_update(self.qf2, self.qf2_target)
            self.soft_target_update(self.cf1, self.cf1_target)
            self.soft_target_update(self.cf2, self.cf2_target)

        def soft_target_update(self, main, target, tau=0.005):
            for main_param, target_param in zip(main.parameters(), target.parameters()):
                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

        def select_a(self, state, mode='train'):
            state = torch.FloatTensor(state).to(device)
            mu, std, pi = self.actor(state)
            action = self.act_limit * pi

            if mode != 'train':
                action = self.act_limit * mu

            return action.item(), pi.item()

        def attacker_constraint(self, state_batch, k=0.05, epsilon=0.05):
            self.state_batch = state_batch
            target_list_grads = []
            perturb_list = []

            pbounds = {'pert1': (0.95, 1.05)}
            optimizer = BayesianOptimization(f=self.cost_pert_black_attack, pbounds=pbounds, random_state=0)
            util = UtilityFunction(kind='ucb', kappa=0.2, xi=0.0)
            for i in range(5):
                probe_para = optimizer.suggest(util)
                target = self.cost_pert_black_attack(**probe_para)
                optimizer.register(probe_para, target.item())

                target_list_grads.append(target)
                perturb_list.append(probe_para)

            optimal_perturb = perturb_list[target_list_grads.index(max(target_list_grads))]
            pert1 = optimal_perturb['pert1']

            pert2 = k * torch.randn(state_batch.shape[0], state_batch.shape[1])
            pert2.requires_grad = True

            mu_pert, std_pert, pi_pert = self.actor(pert1 * state_batch + pert2)
            cost_pert = 0.5 * (self.cf1(pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device) + self.cf2(
                pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device))

            # Calculate gradient
            cost_pert.mean().backward(retain_graph=True)

            # Calculate sign of gradient
            signs = torch.sign(pert2.grad)

            # Add
            pert2 = pert2 + epsilon * signs

            # new loss
            mu_pert, std_pert, pi_pert = self.actor(pert1 * state_batch + pert2)
            cost_pert = 0.5 * (self.cf1(pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device) + self.cf2(
                pert1 * state_batch + pert2, pi_pert).squeeze(1).to(device))

            mu, std, pi = self.actor(state_batch)
            a_cost = F.mse_loss(mu_pert, mu)

            return cost_pert, a_cost

        def cost_pert_black_attack(self, pert1):
            mu_pert, std_pert, pi_pert = self.actor(pert1 * self.state_batch)
            cost_pert = 0.5 * (self.cf1(pert1 * self.state_batch, pi_pert).squeeze(1).to(device) + self.cf2(
                pert1 * self.state_batch, pi_pert).squeeze(1).to(device))

            return cost_pert.mean()

        def train(self, mode: bool = True) -> "ASAC":
            self.actor.train(mode)
            self.qf1.train(mode)
            self.qf2.train(mode)
            self.cf1.train(mode)
            self.cf2.train(mode)
            return self

        def save_model(self, model_name, model_path):
            name = './' + model_path + 'policy%d' % model_name
            torch.save(self.actor, "{}.pkl".format(name))
            torch.save(self.cf1, "{}.pkl".format(name))
            torch.save(self.cf2, "{}.pkl".format(name))



