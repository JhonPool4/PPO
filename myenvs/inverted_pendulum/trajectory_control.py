# libraries
from gym import Env
from gym.spaces import Box
import numpy as np
import random
import pygame
from copy import copy

# custom RL environment: has 250 seconds to control
class InvertedPendulum(Env):
    def __init__(self):
        # system parameters
        self.g = 9.81 # gravity (m/s^2)
        self.m = 1 # bar's mass (kg)
        self.l = 1 # bar's length (m)
        self.I = self.m*np.power(self.l,2)/3 # bar's inertia
        self.tau = 0 # init torque value
        self.max_torque= 100 # (N.m)
        self.max_angle = 2*np.pi # (rad)
        self.max_speed = 2*np.pi # (rad/s)
        self.max_accel = 2*np.pi # (rad/s/S)
        self.dt = 0.01 # sampling time (s)
        # desired position, velocity and acceleration
        #self.th_des = 0.0
        #self.dth_des = 0.0
        #self.ddth_des = 0.0
        # render
        self.flag_render=False
        self.iter=1
        # frames per second
        self.FPS=100
        
        # screen dimenstion
        self.SCREEN_WIDTH=500
        self.SCREEN_HEIGHT=500
        # bar position and dimension
        self.BAR_XPOS=0.5*self.SCREEN_WIDTH
        self.BAR_YPOS=0.5*self.SCREEN_HEIGHT
        self.BAR_LENGTH=200
        self.BAR_WIDTH=10
        # pivot point position and radius
        self.CIRCLE_RADIUS=10
        self.CIRCLE_XPOS=self.BAR_XPOS
        self.CIRCLE_YPOS=self.BAR_YPOS

        # colors
        self.BLACK=(0,0,0)
        self.RED=(255,0,0)
        self.GRAY=(125,125,125)
                
        # action space: torque
        max_act_values = np.array([self.max_torque], dtype=np.float32)
        self.action_space = Box(low=-max_act_values, high=max_act_values)
        # observation space: theta, dtheta, ddtheta 
        max_obs_values= np.array([self.max_angle, self.max_speed, self.max_accel, self.max_angle, self.max_speed, self.max_accel], dtype=np.float32)
        self.observation_space = Box(low=-max_obs_values, high=max_obs_values)      

    def reset(self):
        # simulation time
        self.sim_time = 0
        self.max_sim_time=500
        
        # initial configuration: with normalization
        init_ang = np.pi/2 + np.random.uniform(-1*np.pi/180, 1*np.pi/180)
        init_vel = np.random.uniform(-1*np.pi/180, 1*np.pi/180)
        init_accel = 0.0 
    
        # desired states
        th_des = np.pi/2 + np.pi/4*np.sin(2*np.pi*(self.sim_time*self.dt))
        dth_des = 0.0 + np.pi/4*2*np.pi*np.cos(2*np.pi*(self.sim_time*self.dt))
        ddth_des = 0.0 - np.pi/4*2*np.pi*2*np.pi*np.sin(2*np.pi*(self.sim_time*self.dt))
        
        self.states = np.array([init_ang, init_vel, init_accel, th_des, dth_des, ddth_des], dtype=np.float32)
        
        if self.flag_render:        
            # initializate pygame
            pygame.init()        

            # font type
            self.font = pygame.font.SysFont("Arial", 20)
            # screen dimension
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            # screen background
            self.screen.fill(self.BLACK)
            # set screen title
            pygame.display.set_caption("Inverted pendulum with reinforcement learning")

            # clock
            self.clock = pygame.time.Clock()  
            # bar configuration
            self.bar = pygame.Surface((self.BAR_WIDTH , self.BAR_LENGTH)) # dimension
            self.bar_rect = self.bar.get_rect()   # center and dimension
            self.bar_rect.center = (0.5*self.SCREEN_WIDTH , 0.5*self.SCREEN_HEIGHT)  # set center
            self.bar.set_colorkey(self.BLACK) # transparent w.r.t background
            # bar color
            self.bar.fill(self.RED)   
        
        return self.states

    def step(self, act):        
        # get current angular configuration
        th, dth, ddth, th_des, dth_des, ddth_des = self.states
        # torque limit
        self.tau = np.clip(act[0], -self.max_torque, self.max_torque)
       
        # forward dynamics
        new_ddth = self.tau/self.I - self.m*self.g*self.l*np.sin(th)/self.I # acceleration
        new_dth = np.clip(dth + self.dt*new_ddth, -self.max_speed, self.max_speed)  # velocity
        new_th = (th + self.dt*(new_dth) + 0.5*np.power(self.dt,2)*new_ddth)%(2*np.pi) # position: with normalization

        # desired states
        new_th_des = np.pi/2 + np.pi/4*np.sin(2*np.pi*(self.sim_time*self.dt))
        new_dth_des = 0.0 + np.pi/4*2*np.pi*np.cos(2*np.pi*(self.sim_time*self.dt))
        new_ddth_des = 0.0 - np.pi/4*2*np.pi*2*np.pi*np.sin(2*np.pi*(self.sim_time*self.dt))
        
        #self.des_states = np.array([self.th_des, self.dth_des, self.ddth_des])

        # update states
        self.states = np.array([new_th, new_dth, new_ddth, new_th_des, new_dth_des, new_ddth_des], dtype=np.float32)
        
        # reward system:
        cost_e = -1*np.power(th_des - th, 2) # reduce positin error
        cost_de = -0.1*np.power(dth_des - dth,2) # reduce velocity error
        cost_dde = -0.001*np.power(ddth_des - ddth, 2) # reduce acceleration error
        #cost_tau = -0.001*np.power(self.tau,2) # reduce energy consumption
        # reward
        reward = cost_e + cost_de + cost_dde #+ cost_tau #+ alive_bonus
        
        # reduce time
        self.sim_time += 1
        # terminal condition
        if (self.sim_time>=self.max_sim_time) :
            return self.states, 0, True, {} #dict(cost_e=cost_e, cost_de=cost_de, cost_tau=cost_tau)        
        else:
            return self.states, reward, False, {} #dict(cost_e=cost_e, cost_de=cost_de, cost_tau=cost_tau)
        
    def render(self):
        if self.flag_render:
            # background color
            self.screen.fill(self.BLACK)
            # frame-per-seconds
            self.clock.tick(self.FPS)  
            # rotated bar
            rot_bar = pygame.transform.rotate(self.bar , self.states[0]*180/np.pi)
            rot_bar_rect=rot_bar.get_rect() 
            rot_bar_rect.center=(self.BAR_XPOS+0.5*self.BAR_LENGTH*np.sin(self.states[0]),\
                                 self.BAR_YPOS+0.5*self.BAR_LENGTH*np.cos(self.states[0]))
            # display rotated bar
            self.screen.blit(rot_bar, rot_bar_rect) 
            # draw pivot point
            pygame.draw.circle(self.screen, self.GRAY, (self.CIRCLE_XPOS,self.CIRCLE_YPOS), self.CIRCLE_RADIUS, 0)
            
            # display desired position
            y_pos=50
            x_pos=20
            x_space=30
            #text_d = self.font.render("Des: "+str(np.round(self.th_des*180/np.pi)), True, (0, 0, 255))
            #text_d_rect = text_d.get_rect()
            #text_d_rect.center = (y_pos, x_pos)
            #self.screen.blit(text_d, text_d_rect)    
            # display current position
            #text_m = self.font.render("Med: "+str(np.round(self.states[0]*180/np.pi)), True, (0, 0, 255))
            #text_m_rect = text_m.get_rect()
            #text_m_rect.center = (y_pos, x_pos+x_space)
            #self.screen.blit(text_m, text_m_rect)                
            # display position error
            text_e = self.font.render("Error: "+str(np.round((self.states[3]-self.states[0])*180/np.pi,2)), True, (0,0,255))
            text_e_rect = text_e.get_rect()
            text_e_rect.center=(y_pos, x_pos+1*x_space)
            self.screen.blit(text_e, text_e_rect)
            # display velocity error
            text_de = self.font.render("dError: "+str(np.round((self.states[4]-self.states[1])*180/np.pi,2)), True, (0,0,255))
            text_de_rect = text_de.get_rect()
            text_de_rect.center=(y_pos, x_pos+2*x_space)
            self.screen.blit(text_de, text_de_rect)            
            # display u
            text_u = self.font.render("Tau: "+str(np.round(self.tau,2)), True, (0,0,255))
            text_u_rect = text_u.get_rect()
            text_u_rect.center=(y_pos, x_pos+3*x_space)
            self.screen.blit(text_u, text_u_rect)            
            # display iteration
            text_i = self.font.render("Iter: "+str(self.iter), True, (0,0,255))
            text_i_rect = text_i.get_rect()
            text_i_rect.center=(y_pos, x_pos+4*x_space)
            self.screen.blit(text_i, text_i_rect)             
            
            # update screen
            pygame.display.flip() 
        else:
            pass
    
    def close(self):
        # close screen
        if self.flag_render:
            pygame.display.quit()
            pygame.quit()
        else:
            pass        


