import torch
import numpy as np
import math
from model.baseline import ReactiveBaseline

class PG(object):
    def __init__(self, config):
        self.config = config
        self.positive_reward = 1.0
        self.negative_reward = 0.0
        self.baseline = ReactiveBaseline(config, config['lambda'])
        self.now_epoch = 0

    def get_reward(self, current_entites, answers):
         positive = torch.ones_like(current_entites, dtype=torch.float32) * self.positive_reward
         negative = torch.ones_like(current_entites, dtype=torch.float32) * self.negative_reward
         reward = torch.where(current_entites == answers, positive, negative)
         return reward

    def calc_cum_discounted_reward(self, rewards, all_critic_values):
        # running_add = torch.zeros([rewards.shape[0]]) 
        running_add = torch.stack(all_critic_values, dim=-1).squeeze(dim=1)[:, -1] * self.config['phi']
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config['path_length']]) # shape: [batchsize, path_length]
        if self.config['cuda']:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config['path_length'] - 1] = rewards
        for t in reversed(range(self.config['path_length'])): 
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t] 
            # running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]  
            # all_critic_values.shape: [batchsize, pathlength + 1]
            cum_disc_reward[:, t] = running_add 
        return cum_disc_reward # shape: [batchsize, pathlength] 


    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))
        return entropy_loss


    def calc_reinforce_loss(self, all_loss, all_logits, all_critic_values, cum_discounted_reward):
        loss = torch.stack(all_loss, dim=1)
        critic_values = torch.stack(all_critic_values, dim=-1).squeeze(dim=1)
        final_reward = cum_discounted_reward - critic_values[:, :-1] * self.config['phi']
        # cum_discounted_reward.shape: [batchsize, pathlength]
        # critic_values.shape: [batchsize, pathlength + 1]

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        entropy_loss = self.config['ita'] * math.pow(self.config['zita'], self.now_epoch) * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss
        return total_loss, final_reward


