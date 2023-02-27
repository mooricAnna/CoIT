import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

import utils
from torch.distributions.normal import Normal
        

class RandomCrop(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        y1 = np.random.randint(0, 9)
        y2 = np.random.randint(0, 9)
        x = x[:, :, y1:(y1+84), y2:(y2+84)]
        return x


class AutoShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.ratio = 0.2

    def forward(self, x, mean, var, act=False):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1).to(x.device)

        mean = torch.clamp(
            mean, 1e-6, 2 * (2 * self.pad + 1) / (h + 2 * self.pad))
        var = torch.clamp(var, min=1e-6)
        dist = Normal(mean, var)
        if act:
            shift = dist.mean
            shift = shift.unsqueeze(0)
            shift = torch.clamp(shift, 0, 
            2 * (2 * self.pad +1) / (h + 2 * self.pad))

            grid = base_grid + shift
            return F.grid_sample(x,
                                 grid,
                                 padding_mode='zeros',
                                 align_corners=False)
        else:
            shift = dist.rsample()
            shift = shift.unsqueeze(0).unsqueeze(0)
            shift = shift.expand(n, 1, 1, 2)
            noise = torch.normal(0, 0.0075, 
                                 size=(n, 1, 1, 2),
                                 device=x.device,
                                 dtype=x.dtype)

            shift = torch.clamp(shift, 0, 
            2 * (2 * self.pad +1) / (h + 2 * self.pad))
            # noise = torch.clamp(noise, -0.01, 0.01)

            # noise = torch.randint(0,
            #                       2 * self.pad + 1,
            #                       size=(n, 1, 1, 2),
            #                       device=x.device,
            #                       dtype=x.dtype)
            # noise *= (self.ratio * 2.0) / (h + 2 * self.pad)
            grid = base_grid + shift + noise
            return F.grid_sample(x,
                                 grid,
                                 padding_mode='zeros',
                                 align_corners=False)


class LatentAug(nn.Module):
    
    def __init__(self, num):
        super().__init__()
        self.num = num

    def forward(self, x1, x2):        
        alpha = 1
        y = np.random.beta(alpha, alpha)
        x = x1 * y + x2 * (1-y)
        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        
        self.convnet1 = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2), 
                                     nn.BatchNorm2d(32), nn.ReLU()) 
        self.convnet2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=1),
                                     nn.BatchNorm2d(32), nn.ReLU())
        self.convnet = nn.Sequential(nn.Conv2d(32, 32, 3, stride=1), 
                                     nn.BatchNorm2d(32), nn.ReLU(), 
                                     nn.Conv2d(32, 32, 3, stride=1), 
                                     nn.BatchNorm2d(32), nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h1 = self.convnet1(obs)
        h2 = self.convnet2(h1)
        h = self.convnet(h2)
        # h = h.view(h.shape[0], -1)
        return h
    
    def heat_map(self, obs):
        obs = obs / 255.0 - 0.5
        h1 = self.convnet1(obs)
        h2 = self.convnet2(h1)
        return [h.pow(2).mean(1) for h in (h1, h2)]


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.BatchNorm1d(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim), 
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        obs = obs.view(obs.shape[0], -1)
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.tanh = nn.Tanh()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.BatchNorm1d(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        obs = obs.view(obs.shape[0], -1)
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class MLP(nn.Module):
    def __init__(self, repr_dim, hidden_dim, sim_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(repr_dim, hidden_dim),
                                #  nn.BatchNorm1d(hidden_dim), nn.ReLU(), 
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, sim_dim))
        
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        obs = obs.view(obs.shape[0], -1)
        h = self.mlp(obs)
        return h


class CPC(nn.Module):
    def __init__(self, encoder, encoder_target, 
                 mlp, mlp_target):
        super().__init__()
        
        self.encoder = encoder
        self.encoder_target = encoder_target
        self.mlp = mlp
        self.mlp_target = mlp_target

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out
    
    def projection(self, x, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.mlp_target(x)
        else:
            z_out = self.mlp(x)
        return z_out
    
    def compute_sim(self, z_a, z_pos):
        x = F.normalize(z_a, dim=-1, p=2)
        y = F.normalize(z_pos, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def comput_norm(self, obs, mean_bn, var_bn):
        input = obs.unsqueeze(0)
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        r_feature = torch.norm(var_bn.data - var, 2) + torch.norm(
           mean_bn - mean, 2)
        return r_feature


class Agent:
    def __init__(self, obs_shape, action_shape, device, lr, aug_lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, 
                 encoder_target_tau, alpha, lam):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.encoder_target_tau = encoder_target_tau
        self.alpha = alpha
        self.lam = lam
        self.pad = 4
        self.num = 2

        # models
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.encoder_target = Encoder(obs_shape, feature_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.mlp = MLP(self.encoder.repr_dim, hidden_dim).to(device)
        self.mlp_target = MLP(self.encoder.repr_dim, hidden_dim).to(device)
        self.mlp_target.load_state_dict(self.mlp.state_dict())

        self.cpc = CPC(self.encoder, self.encoder_target,
                       self.mlp, self.mlp_target).to(device)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.mlp_opt = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.sim_encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=aug_lr)

        # data augmentation
        self.init_shift = 2 * self.pad + 1
        self.mean =  torch.tensor(
            self.init_shift /(84 + 2 * self.pad)).unsqueeze(0).to(device)
        self.mean = torch.cat([self.mean, self.mean]).unsqueeze(0)
        self.mean.requires_grad = True
        self.var = torch.tensor(
            self.init_shift / (1.2 * (84 + 2 * self.pad))).unsqueeze(0).to(device)
        self.var = torch.cat([self.var, self.var]).unsqueeze(0)
        self.var.requires_grad = True         
        self.aug = AutoShiftsAug(self.pad).to(device)
        self.crop = RandomCrop(self.pad).to(device)
        self.mix = LatentAug(self.num).to(device)
        self.mean_opt = torch.optim.Adam([self.mean], lr=aug_lr)
        self.var_opt = torch.optim.Adam([self.var], lr=aug_lr)

        self.train()
        self.encoder_target.train()
        self.critic_target.train()
        self.mlp_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.mlp.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.aug(obs.unsqueeze(0).float(), 
                       self.mean, self.var, act=True)
        obs = self.encoder(obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, obs_shift, obs_raw, reward, discount, 
                      next_obs, alpha, lam, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        
        # shift loss
        mean_bn = self.encoder.convnet1[1].running_mean.mean()
        mean_bn += self.encoder.convnet2[1].running_mean.mean()
        mean_bn += self.encoder.convnet[1].running_mean.mean()
        mean_bn += self.encoder.convnet[4].running_mean.mean()
        mean_bn = mean_bn / 4

        var_bn = self.encoder.convnet1[1].running_var.mean()
        var_bn += self.encoder.convnet2[1].running_var.mean() 
        var_bn += self.encoder.convnet[1].running_var.mean() 
        var_bn += self.encoder.convnet[4].running_var.mean()
        var_bn = var_bn / 4

        feature_l = self.cpc.comput_norm(obs_shift, mean_bn, var_bn)
        
        z_a_p = self.cpc.projection(obs)
        z_k_p = self.cpc.projection(obs_raw, ema=True)
        sim_l1 = self.cpc.compute_sim(z_a_p, z_k_p)
        sim_l = sim_l1.mean()
        
        shift_loss = alpha * sim_l + lam * feature_l
        critic_loss += shift_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            # metrics['aug_loss'] = shift_loss.item()

        # optimize encoder and critic
        self.mean_opt.zero_grad(set_to_none=True)
        self.var_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.mlp_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.mlp_opt.step()
        self.encoder_opt.step()
        self.var_opt.step()
        self.mean_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
             
        if step % 10000 == 0:
            print(self.mean.detach(), self.var.detach())
        obs_raw = obs.float().detach()
        obs = obs.float()
                
        # obs_shift = self.aug(obs, self.mean, self.var)
        obs = self.aug(obs, self.mean, self.var)
        obs1 = self.aug(obs, self.mean, self.var)
        obs2 = self.aug(obs, self.mean, self.var)
        # obs1 = self.encoder(obs1)
        # obs2 = self.encoder(obs2)
        obs = self.mix(obs1, obs2)
        obs_shift = obs.clone()
        obs = self.encoder(obs)

        with torch.no_grad():
            obs_raw = self.encoder_target(obs_raw)

            next_obs = next_obs.float()
            next_obs1 = self.aug(next_obs, self.mean, self.var)
            next_obs2 = self.aug(next_obs, self.mean, self.var)
            # next_obs1 = self.encoder(next_obs1)
            # next_obs2 = self.encoder(next_obs2)
            next_obs = self.mix(next_obs1, next_obs2)
            next_obs = self.encoder(next_obs)

        # obs = self.mix(obs1, obs2)
        # next_obs = self.mix(next_obs1, next_obs2)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            
        # update critic
        metrics.update(
            self.update_critic(obs, action, obs_shift, obs_raw, reward,
                               discount, next_obs, self.alpha, 
                               self.lam, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        # update encoder & mlp target
        utils.soft_update_params(self.cpc.encoder, self.cpc.encoder_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.cpc.mlp, self.cpc.mlp_target,
                                 self.encoder_target_tau)

        return metrics
