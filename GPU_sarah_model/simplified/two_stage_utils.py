#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021

@author: sarah
"""


import torch

torch.set_num_threads(1)
print("torch threads", torch.get_num_threads())
arr_type = "torch"
if arr_type == "numpy":
    import numpy as ar
    array = ar.array
else:
    import torch as ar
    array = ar.tensor

device = torch.device("cuda")

import perception as prc


import itertools

import os
import glob

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda")
#device = torch.device("cpu")


#torch.autograd.set_detect_anomaly(True)
###################################
###################################

class FittingAgent(object):

    def __init__(self, perception, action_selection, policies,
                 trials = 1, T = 10, number_of_states = 6,
                 number_of_rewards = 2,
                 number_of_policies = 10, nsubs = 1):

        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection

        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        self.nr = number_of_rewards

        self.T = T
        self.trials = trials
        
        self.nsubs = nsubs

        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = ar.eye(self.npi, dtype = int).to(device)

        self.possible_polcies = self.policies.clone().detach()

        self.actions = ar.unique(self.policies).to(device)
        self.na = len(self.actions)


    def reset(self, locs):

        self.set_parameters(locs)
        self.perception.reset()


    def update_beliefs(self, tau, t, observation, reward, prev_response, context):

        if t==0:
            self.possible_policies = ar.ones((self.npi, self.nsubs), dtype=bool)
        else:
            curr_policies = (self.policies[:,t-1][:,None] == prev_response)#[0]
            self.possible_policies = ar.logical_and(self.possible_policies, curr_policies)

        self.perception.update_beliefs(tau, t, observation, reward, prev_response, self.possible_policies, context)

    def generate_response(self, tau, t):

        #get response probability
        posterior_actions = self.perception.posterior_actions[-1]

        controls = self.policies[:, t]#[non_zero]
        actions = ar.unique(controls)
        # posterior_policies = posterior_policies[non_zero]
        # avg_likelihood = avg_likelihood[non_zero]
        # prior = prior[non_zero]

        response = self.action_selection.select_desired_action(tau,
                                        t, posterior_actions[:,0,0], actions, None, None)


        return response

        #return self.control_probs[tau,t]

    def set_parameters(self, locs):

        self.perception.set_parameters(locs)

    def locs_to_pars(self, locs):

        par_dict = self.perception.locs_to_pars(locs)

        return par_dict

class AveragedSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_actions, actions, *args):


        #generate the desired response from action probability
        u = ar.distributions.Categorical(posterior_actions).sample()

        return u

    # def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):
    #     #estimate action probability
    #     control_prob = ar.zeros(self.na)
    #     for a in range(self.na):
    #         control_prob[a] = posterior_policies[actions == a].sum()


    #     self.control_probability[tau, t] = control_prob


class MaxSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = ar.zeros((trials, T, self.na))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        #generate the desired response from maximum policy probability
        indices = ar.where(posterior_policies == ar.amax(posterior_policies))
        u = ar.random.choice(actions[indices])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = ar.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()

        self.control_probability[tau, t] = control_prob


"""
run function
"""
def set_up_Bayesian_agent(agent_par_list, trials, T, ns, na, nr, nb, A, B, nsubs=1, **kwargs):

    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    avg, perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h = agent_par_list
    
    utility = torch.tensor([0.01, 0.99])
    

    """
    create matrices
    """

    C_alphas = torch.zeros((nr, ns)) + 1
    C_alphas[0,:(ns-nb)] = 100
    for i in range(1,nr):
        C_alphas[i,0] = 1


    """
    create policies
    """

    pol = torch.tensor(list(itertools.product(list(range(na)), repeat=T-1)))

    #pol = pol[-2:]
    npi = pol.shape[0]



    """
    set state prior (where agent thinks it starts)
    """

    state_prior = torch.zeros((ns))

    state_prior[0] = 1.

    """
    set action selection method
    """

    if avg:

        sel = 'avg'

        ac_sel = AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        sel = 'max'

        ac_sel = MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)


    """
    set up agent
    """
    
    pol_lambda = perception_args["policy rate"]
    r_lambda = perception_args["reward rate"]
    dec_temp = perception_args["dec temp"]    
    if use_h:
        alpha_0 = 1./perception_args["habitual tendency"]
    else:
        alpha_0 = perception_args["habitual tendency"]
    alphas = torch.zeros((npi)) + alpha_0
    cached_weight = perception_args["cached weight"]
    cached_r_lambda = perception_args["cached rate"]

    # print(use_h)
    # print(alpha_0)

    if learn_rewards:
        infer_decision_temp = True
        infer_reward_rate = True
    else:
        infer_decision_temp = False
        infer_reward_rate = False

    if learn_habit:
        infer_h = True
        infer_policy_rate = True
    else:
        infer_h = False
        infer_policy_rate = False

    if learn_cached:
        infer_cached_weight = True
        infer_cached_rate = True
    else:
        infer_cached_weight = False
        infer_cached_rate = False

    bayes_prc = prc.Group2ContextPerception(A, B, torch.tensor([[1]]),
                                    state_prior, utility, torch.tensor([1]), pol,
                                    alpha_0=alpha_0, dirichlet_rew_params=C_alphas, 
                                    learn_habit = learn_habit, mask=valid, learn_cached_rewards=learn_cached,
                                    learn_rew = learn_rewards, T=T, trials=trials,
                                    pol_lambda=pol_lambda, r_lambda=r_lambda,
                                    non_decaying=(ns-nb), dec_temp=dec_temp, 
                                    cached_weight=cached_weight, cached_r_lambda=cached_r_lambda,
                                    nsubs=nsubs, infer_alpha_0=infer_h, use_h=use_h,
                                    infer_context=False, dirichlet_context_obs_params=torch.tensor([[1]]),
                                    infer_decision_temp=infer_decision_temp, infer_policy_rate=infer_policy_rate, 
                                    infer_reward_rate=infer_reward_rate, infer_cached_weight=infer_cached_weight, 
                                    infer_cached_rate=infer_cached_rate)

    # C_alphas = torch.zeros((nr, ns, 2)) + 1
    # C_alphas[0,:(ns-nb),:] = 100
    # for i in range(1,nr):
    #     C_alphas[i,0,:] = 1
    
    # bayes_prc = prc.Group2ContextPerception(A, B, torch.tensor([[0.99, 0.01], [0.01, 0.99]]),
    #                                 state_prior, utility, torch.tensor([0.99, 0.01]), pol,
    #                                 alpha_0=alpha_0, dirichlet_rew_params=C_alphas, 
    #                                 learn_habit = learn_habit, mask=valid,
    #                                 learn_rew = True, T=T, trials=trials,
    #                                 pol_lambda=pol_lambda, r_lambda=r_lambda,
    #                                 non_decaying=(ns-nb), dec_temp=dec_temp, 
    #                                 nsubs=nsubs, infer_alpha_0=infer_h, use_h=use_h,
    #                                 infer_context=True, dirichlet_context_obs_params=torch.tensor([[1, 1], [1, 1]]),
    #                                 learn_context_obs=True,
    #                                 infer_decision_temp=True, infer_policy_rate=infer_policy_rate, infer_reward_rate=True)
    
    bayes_prc.set_parameters(par_dict=perception_args)
    bayes_prc.reset()

    bayes_pln = FittingAgent(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      number_of_states = ns,
                      number_of_policies = npi,
                      number_of_rewards = nr,
                      nsubs = nsubs)
    
    
    return bayes_pln, bayes_prc


def set_up_Bayesian_inference_agent(n_agents, learn_rewards, learn_habit, learn_cached, base_dir, global_experiment_parameters, valid, remove_old=True, use_h=True):

    # if it does exist, empty previous results, if we want that (remove_old==True)
    if remove_old:

        svgs = glob.glob(os.path.join(base_dir,"*.svg"))
        for file in svgs:
            os.remove(file)

        csvs = glob.glob(os.path.join(base_dir,"*.csv"))
        for file in csvs:
            os.remove(file)

        saves = glob.glob(os.path.join(base_dir,"*.save"))
        for file in saves:
            os.remove(file)

        agents = glob.glob(os.path.join(base_dir,"twostage_agent*"))
        for file in agents:
            os.remove(file)

        outputs = glob.glob(os.path.join(base_dir,"*.json"))
        for file in outputs:
            os.remove(file)

    # perception args for init, will instantly be over-written, but have to be set for initialization
    pol_lambda = torch.tensor([0.5])
    r_lambda = torch.tensor([0.5])
    dec_temp = torch.tensor([2.])   
    alpha_0 = torch.tensor([1.])
    c_weight = torch.tensor([1.])
    c_lambda = torch.tensor([0.5])

    perception_args = {"policy rate": pol_lambda, "reward rate": r_lambda, "dec temp": dec_temp, "habitual tendency": alpha_0}
    perception_args = {"dec temp": dec_temp, "reward rate": r_lambda, 
                       "habitual tendency": alpha_0, "policy rate": pol_lambda, 
                       "cached weight": c_weight, "cached rate": c_lambda}

    avg = True

    agent_par_list = [avg, perception_args, learn_rewards, learn_habit, learn_cached, valid, use_h]
    bayes_agent, bayes_perception = set_up_Bayesian_agent(agent_par_list, **global_experiment_parameters, nsubs=n_agents)

    return bayes_agent
