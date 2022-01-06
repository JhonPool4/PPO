# libraries
from policies import MLPActorCritic
from buffer import PPOBuffer
from torch.optim import Adam
import torch.nn as nn
import torch
from copy import copy
import pandas as pd
import numpy as np
import os

class PPO():
    def __init__(self, env, **hyperparameters):
        # update hyperparameters
        self.set_hyperparameters(hyperparameters)
        
        # get information from environment
        self.env=env
        self.env.flag_render=self.flag_render
        self.obs_dim=self.env.observation_space.shape[0]
        self.act_dim=self.env.action_space.shape[0]

        # create neural network model
        self.ac_model=MLPActorCritic(self.obs_dim, self.act_dim, self.hidden, self.activation)

        # optimizer for policy and value function
        self.pi_optimizer=Adam(self.ac_model.pi.parameters(), self.pi_lr)
        self.vf_optimizer=Adam(self.ac_model.vf.parameters(), self.vf_lr)

        # buffer of training data
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # logger to print data
        self.logger={'rew_mean':[], 'rew_std':[]}
        #self.logger={'approx_kl':0, 'entropy':0, 'clip_frac':0, 'loss_pi':0, 'loss_vf':0, \
        #             'delta_loss_pi':0, 'delta_loss_vf':0, 'rew_info':[]}

        # path to save data
        self.data_path = os.path.join('C:\\Users\\USERTEST\\rl\\PPO\\training', 'data')        
        # save training data
        self.column_names=['mean', 'std']
        self.df = pd.DataFrame(columns=self.column_names,dtype=object)
        if self.create_new_training_data:
            self.df.to_csv(self.data_path, mode='w' ,index=False)              
        

    def compute_loss_pi(self, data):
        # get specific training data
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        act_dist, logp = self.ac_model.pi(obs, act) # eval new policy
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        ent = act_dist.entropy().mean().item()
        loss_pi = -(torch.min(ratio * adv, clip_adv) + self.coef_ent*ent).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    def compute_loss_vf(self, data):
        # get specific training data
        obs, ret = data['obs'], data['ret']
        # value function loss
        return ((self.ac_model.vf(obs) - ret)**2).mean()    

    def update(self):
        # get all training data
        data = self.buf.get()

        # old policy loss
        #pi_l_old, pi_info_old = self.compute_loss_pi(data)
        #pi_l_old = pi_l_old.item()
        #vf_l_old = self.compute_loss_vf(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                #logger.log('Early stopping at step %d due to reaching max kl.'%i)
                print(f"Early stooping at step {i} due to max kl")
                break
            loss_pi.backward() # compute grads
            self.pi_optimizer.step() # update parameters
    
        # Value function learning
        for i in range(self.train_vf_iters):
            self.vf_optimizer.zero_grad()
            loss_vf = self.compute_loss_vf(data)
            loss_vf.backward() # compute grads 
            self.vf_optimizer.step() # update parameters

        # Log changes from update
        #kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        #self.logger['approx_kl']=kl
        #self.logger['entropy']=ent
        #self.logger['clip_frac']=cf
        #self.logger['loss_pi']=pi_l_old
        #self.logger['loss_vf']=vf_l_old
        #self.logger['delta_loss_pi']=loss_pi.detach().item() - pi_l_old
        #self.logger['delta_loss_vf']=loss_vf.detach().item() - vf_l_old

    def rollout(self):
        # reset environemnt parameters
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        
        # generate training data
        for t in range(self.steps_per_epoch):
            a, v, logp = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            self.buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = copy(next_o) # should be copy

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t==self.steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                self.buf.finish_path(v)
                #if terminal:
                #    # only save EpRet / EpLen if trajectory finished
                #    logger.store(EpRet=ep_ret, EpLen=ep_len)
                # reset environemnt parameters
                o, ep_ret, ep_len = self.env.reset(), 0, 0
        # logger reward information
        self.logger['rew_mean'].append(np.mean(self.buf.rews).item())
        self.logger['rew_std'].append(np.std(self.buf.rews).item())    
                

    def learn(self):         

        for epoch in range(self.epochs):
            # generate data
            self.rollout()

            # call update
            self.update()

            print("====================")
            print(f"epochs: {epoch+1}")
            #print(f"approx_kl: {self.logger['approx_kl']}")
            #print(f"entropy: {self.logger['entropy']}")
            #print(f"pi_loss: {self.logger['loss_pi']}")
            #print(f"vf_loss: {self.logger['loss_vf']}")
            #print(f"d_pi_loss: {self.logger['delta_loss_pi']}")
            #print(f"d_vf_loss: {self.logger['delta_loss_vf']}")            
            print(f"mean_ret: {np.mean(self.logger['rew_mean'])}")
            print(f"std_ret: {np.mean(self.logger['rew_std'])}")
            print("====================\n")

            # save reward info
            row = np.expand_dims(np.array([self.logger['rew_mean'].item(), self.logger['rew_std'].item()]), axis = 1).tolist()
            df_row = pd.DataFrame.from_dict(dict(zip(self.column_names, row)))
            self.df.append(df_row, sort=False).to_csv(self.data_path, index=False, mode = 'a', header=False)  

            # save model
            if ((epoch+1)%self.save_freq==0):
                torch.save(self.ac_model.state_dict(), './training/ppo_ac_model.pth')
                print("saving model")

            # reset logger
            #self.logger={'approx_kl':0, 'entropy':0, 'clip_frac':0, 'loss_pi':0, 'loss_vf':0, \
            #            'delta_loss_pi':0, 'delta_loss_vf':0, 'rew_info':[]}
            self.logger={'rew_mean':[], 'rew_std':[]}

    def set_hyperparameters(self, hyperparameters):
        self.epochs=1000
        self.steps_per_epoch=2000
        self.max_ep_len=400
        self.gamma=0.99
        self.lam=0.97
        self.clip_ratio=0.2
        self.target_kl=0.01
        self.coef_ent = 0.001

        self.train_pi_iters=50
        self.train_vf_iters=50
        self.pi_lr=3e-4
        self.vf_lr=1e-3

        self.hidden=(64,64)
        self.activation=[nn.Tanh, nn.ReLU]

        self.flag_render=False

        self.save_freq=500
        
        self.create_new_training_data=False        

        # change default hyperparameters
        for param, val in hyperparameters.items():
            exec("self."+param+"="+"val")  

