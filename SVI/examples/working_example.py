#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:31:57 2023

@author: sascha
"""

import torch
import pyro
import pyro.distributions as dist
from tqdm import tqdm
device = torch.device("cpu")

class GeneralGroupInference(object):

    def __init__(self, agent, num_agents, group_data):
        """
        Group inference for original model
        
        agents : list of agents
        group_data : list of data dicts 
        """
        self.agent = agent
        self.trials = agent.trials # length of experiment
        self.num_agents = num_agents # no. of participants
        self.data = group_data # list of dictionaries
        self.n_parameters = len(self.agent.param_names) # number of parameters
        self.loss = []

    def model(self):
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        a = pyro.param('a', torch.ones(self.n_parameters), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(self.n_parameters), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = 1/torch.sqrt(tau) # Gaus sigma

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(self.n_parameters))
        s = pyro.param('s', torch.ones(self.n_parameters), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

        # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
        # you embed what should be done for each subject into the "with pyro.plate" context
        # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
        with pyro.plate('subject', self.num_agents) as ind:

            # draw parameters from Normal and transform (for numeric trick reasons)
            base_dist = dist.Normal(0., 1.).expand_by([self.n_parameters]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))
    
            "locs is either of shape [num_agents, n_parameters] or of shape [n_particles, num_agents, n_parameters]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            self.agent.reset(locs)
            
            n_particles = locs.shape[0]
            print("DOING A ROUND WITH %d PARTICLES"%n_particles)
            t = -1
            for tau in pyro.markov(range(self.trials)):
    
                trial = self.data["Trialsequence"][tau]
                blocktype = self.data["Blocktype"][tau]
                
                if all([self.data["Blockidx"][tau][i] <= 5 for i in range(self.num_agents)]):
                    day = 1
                    
                elif all([self.data["Blockidx"][tau][i] > 5 for i in range(self.num_agents)]):
                    day = 2
                    
                else:
                    raise Exception("Da isch a Fehla!")
                
                if all([trial[i] == -1 for i in range(self.num_agents)]):
                    "Beginning of new block"
                    self.agent.update(torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]), day=day, trialstimulus=trial)
                    
                else:
                    current_choice = self.data["Choices"][tau]
                    outcome = self.data["Outcomes"][tau]
                
                if all([trial[i] > 10 for i in range(self.num_agents)]):
                    "Dual-Target Trial"
                    t+=1
                    option1, option2 = self.agent.find_resp_options(trial)
                    
                    probs = self.agent.compute_probs(trial, day)
                    
                    choices = torch.tensor([0 if current_choice[idx] == option1[idx] else 1 for idx in range(len(current_choice))])
                    obs_mask = torch.tensor([0 if cc == -10 else 1 for cc in current_choice ]).type(torch.bool)

                if all([trial[i] != -1 for i in range(self.num_agents)]):
                    "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                    self.agent.update(current_choice, outcome, blocktype, day=day, trialstimulus=trial)

                "Sample if dual-target trial and no error was performed"
                if all([trial[i] > 10 for i in range(self.num_agents)]):
                    pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), \
                                obs = choices.broadcast_to(n_particles, self.num_agents), \
                                obs_mask = obs_mask.broadcast_to(n_particles, self.num_agents))

    def guide(self):
        trns = torch.distributions.biject_to(dist.constraints.positive)

        # define mean vector and covariance matrix of multivariate normal
        m_hyp = pyro.param('m_hyp', torch.zeros(2*self.n_parameters))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*self.n_parameters),
                       constraint=dist.constraints.lower_cholesky)

        # set hyperprior to be multivariate normal
        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :self.n_parameters]
        unc_tau = hyp[..., self.n_parameters:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        # some numerics tricks
        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

        m_locs = pyro.param('m_locs', torch.zeros(self.num_agents, self.n_parameters))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(self.n_parameters).repeat(self.num_agents, 1, 1),
                        constraint=dist.constraints.lower_cholesky)

        with pyro.plate('subject', self.num_agents):
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs}

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        print("Starting inference steps")
        for step in pbar:#range(iter_steps):
            "Runs through the model twice the first time"
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss += [l.cpu() for l in loss] # = -ELBO