import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomNoise:
    def __init__(self, alpha=2, beta=2, epsilon=0.09, n=20):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.n = n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def sample_random_noise(self, state_shape):
        noise = np.random.beta(self.alpha, self.beta, size=state_shape)
        return noise - 0.5
    
    def generate_adversarial_state(self, state, policy, policy_name):
        print("hi")

        action = policy.select_action(state)
        q = 0
        state_adv = state
        action_adv = action

        if policy_name == "TD3":
            s, a = torch.FloatTensor(np.array([state])).to(self.device), torch.FloatTensor(np.array([action])).to(self.device)
            q1, q2 = policy.critic(s, a)
            q = torch.min(q1, q2).cpu().data.numpy()[0]
        else:
            s, a = torch.FloatTensor(np.array([state])).to(self.device), torch.FloatTensor(np.array([action])).to(self.device)
            q = policy.critic(s, a).cpu().data.numpy()[0]
        
        q_adv = q
        for i in range(self.n):
            s_temp = state + self.epsilon * self.sample_random_noise(state.shape)
            action_adv = policy.select_action(s_temp)
            q_temp = q_adv 
            s, a = torch.FloatTensor(np.array([s_temp])).to(self.device), torch.FloatTensor(np.array([action_adv])).to(self.device)
            if policy_name == "TD3":
                q1, q2 = policy.critic(s, a)
                q_temp = torch.min(q1, q2).cpu().data.numpy()[0]
            else:
                q_temp = policy.critic(s, a).cpu().data.numpy()[0]
            
            if q_temp < q_adv:
                q_adv = q_temp
                state_adv = s_temp

        return state_adv