# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
 
        u = t.float() / self.n_steps

        lambda_max_tensor = torch.tensor(self.lambda_max, dtype=torch.float32, device=t.device)
        lambda_min_tensor = torch.tensor(self.lambda_min, dtype=torch.float32, device=t.device)
    
        b = torch.arctan(torch.exp(-lambda_max_tensor / 2))
        a = torch.arctan(torch.exp(-lambda_min_tensor / 2)) - b

        lambda_t = -2.0 * torch.log(torch.tan(a * u + b))

        return lambda_t.view(-1, 1, 1, 1)
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        #PASS
        alpha_lambda_t_squared = torch.sigmoid(lambda_t)
    
        alpha_lambda_t=torch.sqrt(alpha_lambda_t_squared)

        return alpha_lambda_t
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
        #PASS
        simga_lambda = torch.sqrt(torch.sigmoid(-lambda_t))

        return simga_lambda
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        #PASS
        a = self.alpha_lambda(lambda_t)
        s = self.sigma_lambda(lambda_t)
        z_lambda_t = a * x + s * noise

        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        sigma_lambda_sq = self.sigma_lambda(lambda_t)**2
        exp_ratio=self.get_exp_ratio(lambda_t,lambda_t_prim)
        var_q = (1-exp_ratio)*sigma_lambda_sq
        
        
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        sigma_lambda_prime_sq = self.sigma_lambda(lambda_t_prim)**2
        exp_ratio=self.get_exp_ratio(lambda_t,lambda_t_prim)
        var_q_x = (1-exp_ratio)*sigma_lambda_prime_sq
    
        return var_q_x.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
        
        e_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        alpha_current = self.alpha_lambda(lambda_t)
        alpha_target = self.alpha_lambda(lambda_t_prim)
        term1 = e_ratio * (alpha_target / alpha_current) * z_lambda_t
        term2 = (1 - e_ratio) * alpha_target * x
        mu = term1 + term2

    
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
        
        sigma_q_sq=self.sigma_q(lambda_t,lambda_t_prim)**2
        sigma_q_x_sq=self.sigma_q_x(lambda_t,lambda_t_prim)**2

        var=(sigma_q_x_sq)**(1-v) * (sigma_q_sq)**v

        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
        
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)

        noise = torch.randn_like(z_lambda_t)
        
        sample = mu + torch.sqrt(var) * noise
    
        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        #TODO: q_sample z
        #raise NotImplementedError

        lambda_t = self.get_lambda(t)
        z_lambda = self.q_sample(x0, lambda_t, noise)
        pred_noise = self.eps_model(z_lambda, labels)

        #loss = F.mse_loss(pred_noise, noise)
        loss= ((self.eps_model(z_lambda, labels)-noise)**2).sum(dim=dim).mean()

    
        return loss



    