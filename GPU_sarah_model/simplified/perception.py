
arr_type = "torch"
if arr_type == "numpy":
    import numpy as ar
    array = ar.array
else:
    import torch as ar
    array = ar.tensor


#device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
#device = ar.device("cuda")
#device = ar.device("cpu")

try:
    from inference import device
except:
    device = ar.device("cpu")


class Group2Perception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 prior_states,
                 prior_rewards,
                 policies,
                 alpha_0 = ar.tensor([1]),
                 dirichlet_rew_params = None,
                 learn_habit = False,
                 learn_rew = False,
                 mask=None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=1., npart=1, nsubs=1, infer_alpha_0=False, use_h=True):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.prior_rewards = prior_rewards
        self.nr = prior_rewards.shape[0]
        self.prior_states = prior_states
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.learn_habit = learn_habit
        self.learn_rew = learn_rew
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.policies = policies
        self.npi = policies.shape[0]
        self.actions = ar.unique(policies)
        self.na = len(self.actions)
        self.npart = npart
        self.nsubs = nsubs
        # infer_alpha_0 says whether to infer alpha_0 at all
        self.infer_alpha_0 = infer_alpha_0
        # use_h says whether to use h or alpha_0 for inference
        self.use_h = use_h
        self.alpha_0 = alpha_0
        
        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()
        else:
            self.mask = mask.long()[:,None,:]

        if self.infer_alpha_0:
            self.npars = 4
        else:
            self.npars = 3
        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.npart, self.nsubs)).to(device) + self.alpha_0[None,...]#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        #self.prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        #self.generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        #print(self.big_trans_matrix.shape)

        # self.reset()

    def locs_to_pars(self, locs):

        if self.infer_alpha_0:
            if self.use_h:
                par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
                            "reward rate": ar.sigmoid(locs[...,1]),
                            "dec temp": 10*ar.sigmoid(locs[...,2]),
                            "habitual tendency": ar.sigmoid(locs[...,3])}
            else:
                par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
                            "reward rate": ar.sigmoid(locs[...,1]),
                            "dec temp": 10*ar.sigmoid(locs[...,2]),
                            "habitual tendency": ar.exp(locs[...,3])}
        else:
            par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
                        "reward rate": ar.sigmoid(locs[...,1]),
                        "dec temp": 10*ar.sigmoid(locs[...,2])}

        return par_dict

    def set_parameters(self, locs=None, par_dict=None):

        if locs is not None:
            par_dict = self.locs_to_pars(locs)

        if 'policy rate' in par_dict.keys():
            self.pol_lambda = par_dict['policy rate']
        if 'reward rate' in par_dict.keys():
            self.r_lambda = par_dict['reward rate']
        if 'dec temp' in par_dict.keys():
            self.dec_temp = par_dict['dec temp']
        if 'habitual tendency' in par_dict.keys():
            if self.use_h:
                self.alpha_0 = 1./par_dict['habitual tendency']
            else:
                self.alpha_0 = par_dict['habitual tendency']

    def reset(self):
        if len(self.dec_temp.shape) > 1:
            self.npart = self.dec_temp.shape[0]
            self.nsubs = self.dec_temp.shape[1]
        else:
            self.nsubs = self.dec_temp.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            self.pol_lambda = self.pol_lambda[None,:]
            self.r_lambda = self.r_lambda[None,:]
            self.dec_temp = self.dec_temp[None,:]

        # print(self.alpha_0.shape)
        # print(self.npart, self.nsubs)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.npart, self.nsubs)).to(device) + self.alpha_0[None,...].to(device)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []


    def make_current_messages(self, tau, t):

        generative_model_rewards = self.generative_model_rewards[-1].to(device)

        #obs_messages = ar.zeros((self.nh, self.T)) + 1/self.nh

        # rew_messages = ar.zeros((self.nh, self.T))
        # rew_messages[:] = self.prior_rewards.matmul(generative_model_rewards)[:,None]
        observations = ar.stack(self.observations[-t-1:])*self.mask[tau]
        # obs_messages = []
        # for n in range(self.nsubs):
        #     prev_obs = [self.generative_model_observations[o] for o in observations[-t-1:,n]]
        #     obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        #     obs = [ar.stack(obs).T.to(device)]*self.npart
        #     obs_messages.append(ar.stack(obs, dim=-1))
        # old_obs_messages = ar.stack(obs_messages, dim=-1).to(device)
        
        prev_obs = self.generative_model_observations[observations].permute((2,0,1))[:,:,None,:]
        exp_obs = ar.zeros(self.nh, self.T-t-1, 1, self.nsubs).to(device)+1./self.nh
        new_obs_messages = ar.cat((prev_obs, exp_obs), dim=1)
        obs_messages = ar.cat([new_obs_messages]*self.npart, dim=-2)
        # print("obs correct", ar.allclose(obs_messages, old_obs_messages))

        # prev_obs = [self.generative_model_observations[o] for o in self.observations[-t-1:]]
        # obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        # obs = [ar.stack(obs).T.to(device)]*self.npart
        # obs = [ar.stack(obs,dim=-1).to(device)]*self.nsubs
        # obs_messages = ar.stack(obs,dim=-1).to(device)

        # prev_obs = [[self.generative_model_observations[o] for o in obs_vec] for obs_vec in self.observations[-t-1:]]
        # obs = prev_obs + [[ar.zeros((self.nh))+1./self.nh]*(self.T-t-1)]*n
        # obs_messages = ar.stack(obs).T

        # prev_rew = [generative_model_rewards[r] for r in self.rewards[-t-1:]]
        # rew = prev_rew + [self.prior_rewards.matmul(generative_model_rewards)]*(self.T-t-1)
        # rew_messages = ar.stack(rew).T

        # prev_rew = [generative_model_rewards[r] for r in self.rewards[-t-1:]]
        # rew = prev_rew + [self.prior_rewards.matmul(generative_model_rewards)]*(self.T-t-1)
        # rew_messages = ar.stack(rew).T
        rewards = ar.stack(self.rewards[-t-1:])*self.mask[tau]

        # rew_messages = []
        # for n in range(self.nsubs):
        #     rew_messages.append(ar.stack([ar.stack([generative_model_rewards[r,:,i,n].to(device) for r in rewards[-t-1:,n]]  \
        #                                            + [self.prior_rewards.matmul(generative_model_rewards[:,:,i,n].to(device)).to(device)]*(self.T-t-1)).T.to(device) for i in range(self.npart)], dim=-1).to(device))
        # old_rew_messages = ar.stack(rew_messages, dim=-1).to(device)
        

        one_hot_rews = ar.nn.functional.one_hot(rewards, num_classes=self.nr).float()
        prev_rew = ar.einsum('tnr,rspn->tspn', one_hot_rews, generative_model_rewards)
        exp_rew = ar.einsum('r,rspn->spn', self.prior_rewards, generative_model_rewards)
        exp_rews = ar.cat([exp_rew[None,...]]*(self.T), dim=0)
        rew_messages = ar.cat((prev_rew, exp_rews[:self.T-t-1]), dim=0).permute((1,0,2,3))
        # print("rew correct", ar.allclose(rew_messages, old_rew_messages))
        #print(rew.shape)

        # for i in range(t):
        #     tp = -t-1+i
            # observation = self.observations[tp]
            # obs_messages[:,i] = self.generative_model_observations[observation]

            # reward = self.rewards[tp]
            # rew_messages[:,i] = generative_model_rewards[reward]

        self.obs_messages.append(obs_messages)
        self.rew_messages.append(rew_messages)

    def update_messages(self, tau, t, possible_policies):

        # bwd_messages = ar.zeros((self.nh, self.T,self.npi)) #+ 1./self.nh
        # bwd_messages[:,-1,:] = 1./self.nh
        bwd = [ar.zeros((self.nh, self.npi, self.npart, self.nsubs)).to(device)+1./self.nh]
        # fwd_messages = ar.zeros((self.nh, self.T, self.npi))
        # fwd_messages[:,0,:] = self.prior_states[:,None]
        fwd = [ar.zeros((self.nh, self.npi, self.npart, self.nsubs)).to(device)+self.prior_states[:,None,None,None]]
        # fwd_norms = ar.zeros((self.T+1, self.npi))
        # fwd_norms[0,:] = 1.
        fwd_norm = [ar.ones(self.npi, self.npart, self.nsubs).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            tmp = ar.einsum('hpnk,shp,hnk,hnk->spnk',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1]).to(device)
            bwd.append(tmp)
            #bwd_messages[:,i,:] = ar.einsum('hp,shp,h,h->sp',bwd_messages[:,i+1,:],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1])
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
            #                             obs_messages[:,t-i]*\
            #                             rew_messages[:, t-i]
            # bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
            #      .matmul(self.generative_model_states[:,:,u])
            #bwd_messages[:,i,:] = test[-1]
            norm = bwd[-1].sum(axis=0)
            mask = norm > 0
            bwd[-1][:,mask] /= norm[None,mask]
            # norm = bwd_messages[:,i,:].sum(axis=0)
            # mask = norm > 0
            # bwd_messages[:,i,:][:,mask] /= norm[None,mask]

        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1,0,2,3,4).to(device)

        #     norm = bwd_messages[-1].sum(axis=0)
        #     mask = norm > 0
        #     bwd_messages[-1][:,mask] /= norm[None,mask]

        # bwd_messages = ar.stack(bwd_messages).permute((1,0,2))

        for i in range(self.T-1):
            tmp = ar.einsum('spnk,shp,snk,snk->hpnk',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i],rew_messages[:,i]).to(device)
            fwd.append(tmp)
            # fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
            #                              obs_messages[:, i]*\
            #                              rew_messages[:, i]
            # fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
            #                              matmul(fwd_messages[:, 1+i, pi])
            norm = fwd[-1].sum(axis=0)
            mask = norm > 0
            fwd[-1][:,mask] /= norm[None,mask]
            zeros = ar.zeros((self.npi, self.npart, self.nsubs))
            fwd_norm.append(ar.where(possible_policies[:,None,:], norm, zeros))
            # fwd_norm.append(ar.zeros((self.npi,self.npart)).to(device))
            # fwd_norm[-1][possible_policies] = norm[possible_policies]
            # if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
            #     fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

            # else:
            #     fwd_messages[:,:,pi] = 0#1./self.nh

        fwd_messages = ar.stack(fwd).permute(1,0,2,3,4).to(device)

        # for pi, cs in enumerate(self.policies):
        #     if self.prior_policies[-1][pi] > 1e-15 and pi in possible_policies:

        #         for i, u in enumerate(ar.flip(cs[:], [0])):
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-1-i,pi]*\
        #                                         obs_messages[:,t-i]*\
        #                                         rew_messages[:, t-i]
        #             bwd_messages[:,-2-i,pi] = bwd_messages[:,-2-i,pi]\
        #                  .matmul(self.generative_model_states[:,:,u])

        #             norm = bwd_messages[:,-2-i,pi].sum()
        #             if norm > 0:
        #                 bwd_messages[:,-2-i, pi] /= norm

        #         for i, u in enumerate(cs[:]):
        #             fwd_messages[:, 1+i, pi] = fwd_messages[:,i, pi]*\
        #                                          obs_messages[:, i]*\
        #                                          rew_messages[:, i]
        #             fwd_messages[:, 1+i, pi] = self.generative_model_states[:,:,u].\
        #                                          matmul(fwd_messages[:, 1+i, pi])
        #             fwd_norms[1+i,pi] = fwd_messages[:,1+i,pi].sum()
        #             if fwd_norms[1+i, pi] > 0: #???? Shouldn't this not happen?
        #                 fwd_messages[:,1+i, pi] /= fwd_messages[:,1+i,pi].sum()

        #     else:
        #         fwd_messages[:,:,pi] = 0#1./self.nh

        posterior = fwd_messages*bwd_messages*obs_messages[:,:,None,:]*rew_messages[:,:,None,:]
        norm = posterior.sum(axis = 0)
        #fwd_norms[-1] = norm[-1]
        fwd_norm.append(norm[-1])
        fwd_norms = ar.stack(fwd_norm).to(device)
        # print(tau,t,self.fwd_norms[tau,t])
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)

        return posterior

    def update_beliefs(self, tau, t, observation, reward, prev_response, possible_policies):

        self.update_beliefs_states(tau, t, observation, reward, possible_policies)

        #update beliefs about policies
        self.update_beliefs_policies(tau, t) #self.posterior_policies[tau, t], self.likelihood[tau,t]
        # if tau == 0:
        #     prior_context = self.prior_context
        # else: #elif t == 0:
        #     prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
#            else:
#                prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])

        # print(tau,t)
        # print("prior", prior_context)
        # print("post", self.posterior_context[tau, t])

        # if t < self.T-1:
        #     #post_pol = ar.matmul(self.posterior_policies[tau, t], self.posterior_context[tau, t])
        #     self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t)

        if t == self.T-1 and self.learn_habit:
            self.update_beliefs_dirichlet_pol_params(tau, t)

        if False:
            self.posterior_rewards[tau, t-1] = ar.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t])
        #if reward > 0:
        # check later if stuff still works!
        if self.learn_rew and t==self.T-1:
            self.update_beliefs_dirichlet_rew_params(tau, t, reward)

    def update_beliefs_states(self, tau, t, observation, reward, possible_policies):
        #estimate expected state distribution
        # if t == 0:
        #     self.instantiate_messages(policies)
        self.observations.append(observation)
        self.rewards.append(reward)

        self.update_messages(tau, t, possible_policies)

        #return posterior#ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())

        likelihood = (self.fwd_norms[-1]+1e-10).prod(axis=0).to(device)
        norm = likelihood.sum(axis=0).to(device)
        log_like = ar.log(likelihood/norm[None,...]+1e-10).to(device)
        likelihood = ar.exp(self.dec_temp[None,...]*self.mask[tau]*log_like).to(device)
        # print("like", likelihood)
        # Fe = ar.log((self.fwd_norms[-1]+1e-10).prod(axis=0))
        # softplus = ar.nn.Softplus(beta=self.dec_temp)
        # likelihood = softplus(Fe)
        # posterior_policies = likelihood * self.prior_policies[-1] / (likelihood * self.prior_policies[-1]).sum(axis=0)

        # likelihood = ar.pow(likelihood/norm[None,...],self.dec_temp[None,...]).to(device) #* ar.pow(norm,self.dec_temp)

        # print("softmax", likelihood)
        # print("norm1", ar.pow(norm,self.dec_temp))
        posterior_policies = likelihood * self.prior_policies[-1]*self.mask[tau][None,...] / (likelihood * self.prior_policies[-1]).sum(axis=0)
        # print("unnorm", likelihood * self.prior_policies[-1])
        # print("norm", (likelihood * self.prior_policies[-1]).sum(axis=0))
        # print("post", posterior_policies)
        #likelihood /= likelihood.sum(axis=0)[None,:]
        #posterior/= posterior.sum(axis=0)[None,:]
        #posterior = ar.nan_to_num(posterior)
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        self.posterior_policies.append(posterior_policies)

        if t<self.T-1:
            posterior_actions = ar.zeros((self.na,self.npart, self.nsubs)).to(device)
            for a in range(self.na):
                posterior_actions[a] = posterior_policies[self.policies[:,t] == a,...].sum(axis=0)*self.mask[tau]

            posterior_actions = ar.where(self.mask[tau]>0, posterior_actions, 1./self.na)

            self.posterior_actions.append(posterior_actions)

        #return posterior, likelihood


    def update_beliefs_dirichlet_pol_params(self, tau, t):
        assert(t == self.T-1)
        chosen = ar.eye(self.npi)[ar.argmax(self.posterior_policies[-1], axis=0)].to(device)
        chosen_pol = chosen.permute((2,0,1))
        # print(chosen_pol.shape)
        #print(chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        dirichlet_pol_params = (1-self.pol_lambda*self.mask[tau])[None,:,:] * self.dirichlet_pol_params[-1] \
                                + (1 - (1-self.pol_lambda*self.mask[tau]))[None,:,:]*self.dirichlet_pol_params_init \
                                + chosen_pol*self.mask[tau][None,:]#*self.dirichlet_pol_params_init
        #dirichlet_pol_params[(chosen_pol[0],list(range(self.npart)))] += 1#posterior_context

        prior_policies = dirichlet_pol_params / dirichlet_pol_params.sum(axis=0)[None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        #prior_policies /= prior_policies.sum(axis=0)[None,:]

        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        #return dirichlet_pol_params, prior_policies

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward):
        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,t,:,:,:] * posterior_policies[None,:,:,:]).sum(axis=1)
        # c = ar.argmax(posterior_context)
        # self.dirichlet_rew_params[reward,:,c] += states[:,c]

#         self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] +1 - (1-self.r_lambda)
#         self.dirichlet_rew_params[tau,t,reward,:,:] += states * posterior_context[None,:]
#         for c in range(self.nc):
#             for state in range(self.nh):
#                 #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
#                 self.generative_model_rewards[tau,t,:,state,c] = self.dirichlet_rew_params[tau,t,:,state,c]#\
#                 # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
#                 #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
#                 self.generative_model_rewards[tau,t,:,state,c] /= self.generative_model_rewards[tau,t,:,state,c].sum()
#             self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        dirichlet_rew_params = self.dirichlet_rew_params[0].clone().to(device)#.detach()
        # dirichlet_rew_params = ar.ones_like(self.dirichlet_rew_params_init)#self.dirichlet_rew_params_init.clone()
        # dirichlet_rew_params[:,:self.non_decaying] = self.dirichlet_rew_params[-1][:,:self.non_decaying]
        dirichlet_rew_params[:,self.non_decaying:,:,:] = (1-self.r_lambda*self.mask[tau])[None,None,:,:] * self.dirichlet_rew_params[-1][:,self.non_decaying:,:,:] \
                                                            +1 - (1-self.r_lambda*self.mask[tau])[None,None,:,:]
        #dirichlet_rew_params[reward[0],:,:,:] += states #* posterior_context[None,:]

        vec_rewards = ar.eye(self.nr)[:,reward]
        vec_subjects = ar.eye(self.nsubs)
        matrix_index = ar.einsum('rn,nm->rm', vec_rewards, vec_subjects)
        addition = states[None,...]*matrix_index[:,None,None,:]*self.mask[None,None,tau,...]
        new_rew_params = dirichlet_rew_params + addition

        generative_model_rewards = new_rew_params / new_rew_params.sum(axis=0)[None,...]
        self.dirichlet_rew_params.append(new_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))

        #return dirichlet_rew_params

class Group2ContextPerception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_context,
                 policies,
                 alpha_0 = ar.tensor([1]),
                 cached_reward_params = None,
                 dirichlet_rew_params = None,
                 dirichlet_context_obs_params = None,
                 learn_habit = False,
                 learn_rew = False,
                 infer_context = False,
                 learn_context_obs = False,
                 learn_cached_rewards = False,
                 mask=None,
                 hidden_state_mapping = False,
                 state_mapping = None,
                 infer_policy_rate = True,
                 infer_reward_rate = True,
                 infer_decision_temp = True,
                 infer_alpha_0=True,
                 infer_cached_weight = False,
                 infer_cached_rate = False,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, cached_r_lambda=0, non_decaying=0,
                 dec_temp=1., cached_weight=0., npart=1, nsubs=1, use_h=False,
                 store_internal_variables=False):
        
        ### if another generative model is supposed to be given, i.e. generative model rewards, simply give it as dirichlet params

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.prior_rewards = prior_rewards
        self.nr = prior_rewards.shape[0]
        self.prior_states = prior_states
        self.prior_context = prior_context
        self.nc = prior_context.shape[0]
        self.noc = dirichlet_context_obs_params.shape[0]
        self.transition_matrix_context = transition_matrix_context
        self.T = T
        self.trials = trials
        self.learn_habit = learn_habit
        self.learn_rew = learn_rew
        self.infer_context = infer_context
        self.learn_context_gen = learn_context_obs
        self.learn_cached_rewards = learn_cached_rewards
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.cached_r_lambda = cached_r_lambda
        self.cached_weight = cached_weight
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.policies = policies
        self.npi = policies.shape[0]
        self.na = len(ar.unique(policies))
        self.npart = npart
        self.nsubs = nsubs
        # infer_alpha_0 says whether to infer alpha_0 at all
        self.infer_alpha_0 = infer_alpha_0
        # use_h says whether to use h or alpha_0 for inference
        self.use_h = use_h
        self.infer_policy_rate = infer_policy_rate
        self.infer_reward_rate = infer_reward_rate
        self.infer_decision_temp = infer_decision_temp
        self.infer_cached_weight = infer_cached_weight
        self.infer_cached_rate = infer_cached_rate
        self.alpha_0 = ar.tensor([1.])#alpha_0/self.npi
        self.hidden_state_mapping = hidden_state_mapping
        self.store_internal_variables = store_internal_variables

        if self.use_h:
            self.h = alpha_0
        else:
            self.hab_bias = alpha_0

        if hidden_state_mapping:
            self.nm = dirichlet_rew_params.shape[1]
            self.nh = prior_states.shape[0]
            self.state_mapping = state_mapping
            self.state_mapping_one_hot = ar.nn.functional.one_hot(self.state_mapping, num_classes=self.nm).float()
        else:
            self.nh, self.nm = [prior_states.shape[0]]*2

        if cached_reward_params is not None:
                self.cached_reward_params_init = cached_reward_params
        else:
            self.cached_reward_params_init = ar.ones((self.nr, self.npi, self.nc))
        self.cached_reward_params = [ar.stack([ar.stack([self.cached_reward_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.cached_rewards = [self.cached_reward_params[0] / self.cached_reward_params[0].sum(dim=0)[None,...]]

        cached_preference = (self.cached_rewards[0] * self.prior_rewards[:,None,None,None,None]).sum(dim=0)
        cached_policy_val = cached_preference / cached_preference.sum(dim=0)[None,...]
        self.cached_policy_val = [cached_policy_val]
        
        if mask is None:
            self.mask = ar.ones(trials, nsubs).bool()[:,None,:]
        else:
            self.mask = mask.long()[:,None,:]

        self.npars = self.infer_alpha_0+self.infer_decision_temp+self.infer_policy_rate+self.infer_reward_rate+self.infer_cached_weight+self.infer_cached_rate
        print(self.npars)

        self.param_names = list(self.locs_to_pars(ar.zeros(self.npars)).keys())

        if len(dirichlet_rew_params.shape) > 2:
            self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        else:
            self.dirichlet_rew_params_init = dirichlet_rew_params[:,:,None]

        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.nc,self.npart, self.nsubs)).to(device) + self.alpha_0#[None,None,...]

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]
        # self.dirichlet_update_counts = [ar.zeros_like(self.dirichlet_pol_params_init)]

        prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [prior_policies_init]

        generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [generative_model_rewards_init]
        
        self.miniblock_context_prior = [ar.stack([ar.stack([self.prior_context for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.posterior_context = [self.miniblock_context_prior[0]]

        # when not learning, the dir params can simply contain the real probabilities that one wants to use for the gen mod.
        self.dirichlet_context_obs_params_init = dirichlet_context_obs_params
        self.dirichlet_context_obs_params = [ar.stack([ar.stack([self.dirichlet_context_obs_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]

        generative_model_context_obs_init = self.dirichlet_context_obs_params[0] / self.dirichlet_context_obs_params[0].sum(axis=0)[None,...]
        self.generative_model_context_obs = [generative_model_context_obs_init]

        self.observations = []
        self.rewards = []
        self.context_obs = []
        self.actions = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t]] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        #print(self.big_trans_matrix.shape)

        # self.reset()

        if self.store_internal_variables:
            self.rewards_structured = ar.zeros((trials, T)).int()
            self.actions_structured = ar.zeros((trials, T)).int() - 1
            self.generative_model_rewards_mb = []
            self.prior_policies_mb = []
            self.posterior_context_mb = []

            self.outcome_surprise_log = ar.zeros((trials, T, self.nc))
            self.policy_entropy_log = ar.zeros((trials, T, self.nc))
            self.policy_surprise_log = ar.zeros((trials, T, self.nc))
            self.context_obs_surprise_log = ar.zeros((trials, T, self.nc))

    def locs_to_pars(self, locs):

        count = 0
        par_dict = {}

        if self.infer_policy_rate:
            par_dict["policy rate"] = ar.sigmoid(locs[...,count])
            count += 1
        if self.infer_reward_rate:
            par_dict["reward rate"] = ar.sigmoid(locs[...,count])
            count += 1
        if self.infer_decision_temp:
            par_dict["dec temp"] = 10*ar.sigmoid(locs[...,count])
            count += 1
        if self.infer_alpha_0:
            if self.use_h:
                par_dict["habitual tendency"] = ar.sigmoid(locs[...,count])
            else:
                hab_tend = 10*ar.sigmoid(locs[...,count])
                #hab_tend = ar.exp(locs[...,count])
                par_dict["habitual tendency"] = hab_tend
            count += 1
        if self.infer_cached_weight:
            cached_weight = 10*ar.sigmoid(locs[...,count])
            par_dict["cached weight"] = cached_weight
            count += 1
        if self.infer_cached_rate:
            cached_r_lambda = ar.sigmoid(locs[...,count])
            par_dict["cached rate"] = cached_r_lambda

        # print("locs to pars")
        # print(par_dict)

        # if self.infer_alpha_0:
        #     if self.use_h:
        #         par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
        #                     "reward rate": ar.sigmoid(locs[...,1]),
        #                     "dec temp": 10*ar.sigmoid(locs[...,2]),
        #                     "habitual tendency": ar.sigmoid(locs[...,3])}
        #     else:
        #         par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
        #                     "reward rate": ar.sigmoid(locs[...,1]),
        #                     "dec temp": 10*ar.sigmoid(locs[...,2]),
        #                     "habitual tendency": ar.exp(locs[...,3])}
        # else:
        #     par_dict = {"policy rate": ar.sigmoid(locs[...,0]),
        #                 "reward rate": ar.sigmoid(locs[...,1]),
        #                 "dec temp": 10*ar.sigmoid(locs[...,2])}

        return par_dict

    def set_parameters(self, locs=None, par_dict=None):

        if locs is not None:
            par_dict = self.locs_to_pars(locs)

            if len(locs[...,0].shape) > 1:
                self.npart = locs[...,0].shape[0]
                self.nsubs = locs[...,0].shape[1]
            else:
                self.nsubs = locs[...,0].shape[0]
                self.npart = 1
                for key in par_dict.keys():
                    par_dict[key] = par_dict[key][None,...]
                
        if 'policy rate' in par_dict.keys():
            self.pol_lambda = par_dict['policy rate']
        if 'reward rate' in par_dict.keys():
            self.r_lambda = par_dict['reward rate']
        if 'dec temp' in par_dict.keys():
            self.dec_temp = par_dict['dec temp']
        if 'habitual tendency' in par_dict.keys():
            if self.use_h:
                self.h = par_dict['habitual tendency']
                self.alpha_0 = ar.tensor([1.])#(1./(par_dict['habitual tendency']))/self.npi
            else:
                self.alpha_0 = ar.tensor([1.])#1./self.npi#par_dict['habitual tendency']/self.npi
                self.hab_bias = par_dict['habitual tendency']
        if 'cached rate' in par_dict.keys():
            self.cached_r_lambda = par_dict['cached rate']
        if 'cached weight' in par_dict.keys():
            self.cached_weight = par_dict['cached weight']

        # print("alpha_0", self.infer_alpha_0, self.alpha_0.mean(axis=0))
        # print(self.alpha_0)

    def reset(self):
        # if len(self.dec_temp.shape) > 1:
        #     self.npart = self.dec_temp.shape[0]
        #     self.nsubs = self.dec_temp.shape[1]
        # else:
        #     self.nsubs = self.dec_temp.shape[0]
        #     self.npart = 1
        #     #self.alpha_0 = self.alpha_0[None,:]
        #     self.pol_lambda = self.pol_lambda[None,:]
        #     self.r_lambda = self.r_lambda[None,:]
        #     self.dec_temp = self.dec_temp[None,:]

        # print(self.alpha_0.shape)
        # print(self.npart, self.nsubs)
        
        self.dirichlet_pol_params_init = ar.ones((self.npi,self.nc,self.npart, self.nsubs)).to(device) + self.alpha_0[None,None,...]
        # print("init")
        # print(self.npart, self.nsubs)
        # print(self.dirichlet_pol_params_init[...,0,0])
        # print(self.dirichlet_pol_params_init[...,0,:,0])

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]
        # self.dirichlet_update_counts = [ar.zeros_like(self.dirichlet_pol_params_init)]
        # if self.use_h:
        #     self.h = ar.ones((self.npart, self.nsubs))*1./self.alpha_0

        self.cached_reward_params = [ar.stack([ar.stack([self.cached_reward_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]

        prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [prior_policies_init]

        generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [generative_model_rewards_init]

        cached_rewards_init = self.cached_reward_params[0] / self.cached_reward_params[0].sum(dim=0)[None,...]
        self.cached_rewards = [cached_rewards_init]

        cached_preference = (self.cached_rewards[0] * self.prior_rewards[:,None,None,None,None]).sum(dim=0)
        cached_policy_val = cached_preference / cached_preference.sum(dim=0)[None,...]
        self.cached_policy_val = [cached_policy_val]

        self.miniblock_context_prior = [ar.stack([ar.stack([self.prior_context for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]
        self.posterior_context = [self.miniblock_context_prior[0]]

        self.dirichlet_context_obs_params = [ar.stack([ar.stack([self.dirichlet_context_obs_params_init for k in range(self.npart)], dim=-1) for j in range(self.nsubs)], dim=-1)]

        generative_model_context_obs_init = self.dirichlet_context_obs_params[0] / self.dirichlet_context_obs_params[0].sum(axis=0)[None,...]
        self.generative_model_context_obs = [generative_model_context_obs_init]

        if not self.infer_reward_rate:
            self.rew_lambda = ar.zeros((self.npart, self.nsubs))
        if not self.infer_policy_rate:
            self.pol_lambda = ar.zeros((self.npart, self.nsubs))
        if not self.infer_cached_rate:
            self.cached_r_lambda = ar.zeros((self.npart, self.nsubs))
        if not self.infer_decision_temp:
            self.dec_temp = ar.zeros((self.npart, self.nsubs))
        if not self.infer_alpha_0:
            self.hab_bias = ar.zeros((self.npart, self.nsubs))
        if not self.infer_cached_weight:
            self.cached_weight = ar.zeros((self.npart, self.nsubs))

        self.observations = []
        self.rewards = []
        self.context_obs = []
        self.actions = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []

        if self.store_internal_variables:
            self.rewards_structured = ar.zeros((self.trials, self.T)).int()
            self.actions_structured = ar.zeros((self.trials, self.T)).int() - 1
            self.generative_model_rewards_mb = []
            self.prior_policies_mb = []
            self.posterior_context_mb = []


    def make_current_messages(self, tau, t):

        generative_model_rewards = self.generative_model_rewards[-1].to(device)

        observations = ar.stack(self.observations[-t-1:])*self.mask[tau]
        
        prev_obs = self.generative_model_observations[observations].permute((2,0,1))[:,:,None,:]
        exp_obs = ar.zeros(self.nh, self.T-t-1, 1, self.nsubs).to(device)+1./self.nh
        new_obs_messages = ar.cat((prev_obs, exp_obs), dim=1)
        obs_messages = ar.cat([new_obs_messages]*self.npart, dim=-2)

        rewards = ar.stack(self.rewards[-t-1:])*self.mask[tau]        

        one_hot_rews = ar.nn.functional.one_hot(rewards, num_classes=self.nr).float()
        prev_rew = ar.einsum('tnr,rscpn->tscpn', one_hot_rews, generative_model_rewards)
        exp_rew = ar.einsum('r,rscpn->scpn', self.prior_rewards, generative_model_rewards)
        exp_rews = ar.cat([exp_rew[None,...]]*(self.T), dim=0)
        # note to self: make sure permute is in the right order now with contexts
        rew_messages = ar.cat((prev_rew, exp_rews[:self.T-t-1]), dim=0).permute((1,0,2,3,4))

        if self.hidden_state_mapping:
            rew_messages = ar.einsum('hm,mtcnk->htcnk', self.state_mapping_one_hot[tau], rew_messages)

        self.obs_messages.append(obs_messages)
        self.rew_messages.append(rew_messages)

    def update_messages(self, tau, t, possible_policies):

        bwd = [ar.zeros((self.nh, self.npi, self.nc, self.npart, self.nsubs)).to(device)+1./self.nh]
        fwd = [ar.zeros((self.nh, self.npi, self.nc, self.npart, self.nsubs)).to(device)+self.prior_states[:,None,None,None,None]]
        fwd_norm = [ar.ones(self.npi, self.nc, self.npart, self.nsubs).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            # to understand the indices, look at how the messages are defined above!
            tmp = ar.einsum('hpcnk,shp,hnk,hcnk->spcnk',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i+1],rew_messages[:,i+1]).to(device)
            bwd.append(tmp)
            norm = bwd[-1].sum(axis=0)
            mask = norm > 0
            bwd[-1][:,mask] /= norm[None,mask]

        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1,0,2,3,4,5).to(device)

        for i in range(self.T-1):
            tmp = ar.einsum('spcnk,shp,snk,scnk->hpcnk',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i],rew_messages[:,i]).to(device)
            fwd.append(tmp)
            norm = fwd[-1].sum(axis=0)

            mask = norm > 0
            fwd[-1][:,mask] /= norm[None,mask]
            zeros = ar.zeros((self.npi, self.nc, self.npart, self.nsubs))
            fwd_norm.append(ar.where(possible_policies[:,None,None,:], norm, zeros))

        fwd_messages = ar.stack(fwd).permute(1,0,2,3,4,5).to(device)

        posterior = fwd_messages*bwd_messages*obs_messages[:,:,None,None,...]*rew_messages[:,:,None,...]
        norm = posterior.sum(axis = 0)
        fwd_norm.append(norm[-1])
        fwd_norms = ar.stack(fwd_norm).to(device)
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)

        return posterior

    def update_beliefs(self, tau, t, observation, reward, prev_response, possible_policies, context_obs=None):

        if prev_response is not None:
            self.actions.append(prev_response)
            if self.store_internal_variables:
                self.actions_structured[tau,t] = prev_response
        else:
            self.actions.append(ar.tensor([-1]))

        self.update_beliefs_states(tau, t, observation, reward, possible_policies)

        #update beliefs about policies
        self.update_beliefs_policies(tau, t) #self.posterior_policies[tau, t], self.likelihood[tau,t]

        if self.infer_context:
            if tau>0 and t==0:
                self.miniblock_context_prior.append(ar.einsum('cz,znk->cnk', 
                                        self.transition_matrix_context, 
                                        self.posterior_context[-1]))
            self.update_beliefs_context(tau, t, context_obs)

        # print(tau,t)
        # print("prior", prior_context)
        # print("post", self.posterior_context[tau, t])

        # if t < self.T-1:
        #     #post_pol = ar.matmul(self.posterior_policies[tau, t], self.posterior_context[tau, t])
        #     self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t)

        if t == self.T-1 and self.learn_habit:
            self.update_beliefs_dirichlet_pol_params(tau, t)

        if False:
            self.posterior_rewards[tau, t-1] = ar.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t])
        #if reward > 0:
        # check later if stuff still works!
        if self.learn_rew:# and t==self.T-1:
            self.update_beliefs_dirichlet_rew_params(tau, t, reward)

        if t == self.T-1 and self.learn_cached_rewards:
            self.update_beliefs_dirichlet_cached_rew_params(tau, t)

        if context_obs is not None and t==self.T-1 and self.learn_context_gen and self.infer_context:
            self.update_beliefs_dirichlet_context_gen_params(tau, t, context_obs)

    def update_beliefs_states(self, tau, t, observation, reward, possible_policies):

        self.observations.append(observation)
        self.rewards.append(reward)

        self.update_messages(tau, t, possible_policies)


    def update_beliefs_policies(self, tau, t):

        likelihood = (self.fwd_norms[-1]+1e-10).prod(axis=0).to(device)
        norm = likelihood.sum(axis=0).to(device)
        log_like = ar.log(likelihood/norm[None,...]+1e-10).to(device)
        # weighted_log_like = self.dec_temp[None,...]*self.mask[tau][None,...]*log_like
        likelihood = ar.exp(self.dec_temp[None,...]*self.mask[tau][None,...]*log_like).to(device)

        if self.learn_habit and not self.use_h:
            log_prior = ar.log(self.prior_policies[-1])+1e-10
            # weighted_log_prior = self.hab_bias[None,...]*self.mask[tau][None,...]*log_prior
            prior = ar.exp(self.hab_bias[None,...]*self.mask[tau][None,...]*log_prior).to(device)
        else:
            # weighted_log_prior = ar.log(self.prior_policies[-1]+1e-10)
            prior = ar.ones_like(self.prior_policies[-1])

        # log_post = weighted_log_like + weighted_log_prior

        if self.learn_cached_rewards:
            log_cached = ar.log(self.cached_policy_val[-1]+1e-10)
            # weighted_log_cached = self.cached_weight[None,...] * log_cached
            # log_post += weighted_log_cached
            cached = ar.exp(self.cached_weight[None,...]*self.mask[tau][None,...] * log_cached)
        else:
            cached = ar.ones_like(self.cached_policy_val[-1])

        # posterior_policies = ar.nn.functional.softmax(log_post, dim=0)

        posterior_policies = likelihood * prior * cached/ (likelihood * prior * cached).sum(axis=0)[None,...]

        self.posterior_policies.append(posterior_policies)
        avg_posterior_policies = ar.einsum('pc...,c...->p...', posterior_policies, self.posterior_context[-1])

        if t<self.T-1:
            posterior_actions = ar.zeros((self.na,self.npart, self.nsubs)).to(device)
            for a in range(self.na):
                posterior_actions[a] = avg_posterior_policies[self.policies[:,t] == a,...].sum(axis=0)*self.mask[tau]

            posterior_actions = ar.where(self.mask[tau]>0, posterior_actions, 1./self.na)

            self.posterior_actions.append(posterior_actions)

    def update_beliefs_context(self, tau, t, context_obs=None):

        if self.nc == 1:
            posterior_context = ar.ones((1, self.npart, self.nsubs))

        else:
            prior_context = self.miniblock_context_prior[-1]
            posterior_policies = self.posterior_policies[-1]
            
            #post_policies = ar.einsum('cnk,pcnk->pnk', prior_context, posterior_policies)

            if t == self.T-1:
                chosen_pol = ar.argmax(posterior_policies, dim=0)
                one_hot_pols = ar.nn.functional.one_hot(chosen_pol, num_classes=self.npi).permute(3,0,1,2).float()
                #inf_context = ar.argmax(prior_context)
                alpha_prime = self.dirichlet_pol_params[-1] + (one_hot_pols*prior_context[None,:,:])
            else:
                alpha_prime = self.dirichlet_pol_params[-1]

            if t>0:
                outcome_surprise = (posterior_policies * ar.log(self.fwd_norms[-1].prod(dim=0)+1e-10)).sum(dim=0)
                entropy = - (posterior_policies * ar.log(posterior_policies+1e-10)).sum(dim=0)
                policy_surprise = (posterior_policies * ar.digamma(alpha_prime)).sum(dim=0) - ar.digamma(alpha_prime.sum(dim=0))
            else:
                outcome_surprise = ar.zeros((self.nc, self.npart, self.nsubs))
                entropy = ar.zeros((self.nc, self.npart, self.nsubs))
                policy_surprise = ar.zeros((self.nc, self.npart, self.nsubs))
                
            if context_obs is not None:
                self.context_obs.append(context_obs)
                one_hot_context_obs = ar.nn.functional.one_hot(context_obs.long(), num_classes=self.noc).permute(1,0).float()
                log_gen_mod = ar.log(self.generative_model_context_obs[-1]+1e-10)
                context_obs_suprise = ar.einsum('ocnk,ok->cnk', log_gen_mod, one_hot_context_obs)
            else:
                context_obs_suprise = ar.zeros(self.nc, self.npart, self.nsubs)

            # note: check sing on policy entropy!
            log_posterior = outcome_surprise + policy_surprise + context_obs_suprise +ar.log(prior_context+1e-10)# + entropy
            
            posterior_context = ar.nn.functional.softmax(log_posterior, dim=0)

        self.posterior_context.append(posterior_context)

        if self.store_internal_variables:
            if t==0:
                self.posterior_context_mb.append([posterior_context])
            else:
                self.posterior_context_mb[-1].append(posterior_context)
            if t==self.T-1:
                self.posterior_context_mb[-1] = ar.stack(self.posterior_context_mb[-1])

            # self.prior_context[tau,t] = prior_context
            self.outcome_surprise_log[tau,t] = outcome_surprise[...,0,0]
            self.policy_entropy_log[tau,t] = entropy[...,0,0]
            self.policy_surprise_log[tau,t] = policy_surprise [...,0,0]
            self.context_obs_surprise_log[tau,t] = context_obs_suprise[...,0,0]

    def update_beliefs_dirichlet_context_gen_params(self, tau, t, context_obs):

        one_hot_obs = ar.nn.functional.one_hot(context_obs.long(), num_classes=self.noc).permute(1,0).float()

        # use einsum instead of multiplication with lots of None
        dirichlet_context_obs_params_update = ar.einsum('ok,cnk->ocnk', one_hot_obs, self.posterior_context[-1])

        dirichlet_context_obs_params = self.dirichlet_context_obs_params[-1] + dirichlet_context_obs_params_update

        self.dirichlet_context_obs_params.append(dirichlet_context_obs_params)

        generative_model_context_obs = dirichlet_context_obs_params / dirichlet_context_obs_params.sum(dim=0)[None,...]

        self.generative_model_context_obs.append(generative_model_context_obs)


    def update_beliefs_dirichlet_pol_params(self, tau, t):
        
        assert(t == self.T-1)
        chosen = ar.argmax(self.posterior_policies[-1], dim=0)
        chosen_pol = ar.nn.functional.one_hot(chosen, num_classes=self.npi).permute(3,0,1,2).float()
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        curr_forgetting_factor = (self.pol_lambda*self.mask[tau])[None,None,:,:]*self.posterior_context[-1][None,:,:,:]
        pol_update = chosen_pol[:,:,:,:]*self.mask[tau][None,None,:,:]*self.posterior_context[-1][None,:,:,:]
        if True:
            dirichlet_pol_params = (1 - curr_forgetting_factor) * self.dirichlet_pol_params[-1] \
                                          + curr_forgetting_factor * self.dirichlet_pol_params_init \
                                          + pol_update#*self.dirichlet_pol_params_init
        #dirichlet_pol_params[(chosen_pol[0],list(range(self.npart)))] += 1#posterior_context

        # updated_counts = self.dirichlet_update_counts[-1] + pol_update*self.mask[tau]
        # self.dirichlet_update_counts.append(updated_counts)

        # dirichlet_pol_params = (1.-self.posterior_context[-1][None,:,None,:])*self.dirichlet_pol_params[-1]\
        #                         + self.posterior_context[-1][None,:,None,:]*dirichlet_pol_params_curr_context

        # dirichlet_pol_params = self.dirichlet_pol_params_init + updated_counts

        if self.use_h:
            exp_prior_policies = ar.pow(dirichlet_pol_params,self.h[None,None,...]).to(device)
        else:
            exp_prior_policies = dirichlet_pol_params# / dirichlet_pol_params.sum(dim=0)[None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
            #prior_policies /= prior_policies.sum(axis=0)[None,:]

            # exp_prior_policies = normalized_prior#ar.exp(self.hab_bias[None,...]*normalized_prior).to(device)

        prior_policies = exp_prior_policies / exp_prior_policies.sum(dim=0)[None,...]

        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        # print(tau, t)
        # print(dirichlet_pol_params[...,0,0])
        # print(prior_policies[...,0,0])

        if self.store_internal_variables:
            if t==self.T-1:
                self.prior_policies_mb.append(prior_policies)

        #return dirichlet_pol_params, prior_policies

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward):
        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,t,:,:,:,:] * posterior_policies[None,:,:,:,:]).sum(dim=1)
        # c = ar.argmax(posterior_context)
        # self.dirichlet_rew_params[reward,:,c] += states[:,c]

#         self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] = (1-self.r_lambda) * self.dirichlet_rew_params[tau,t,:,self.non_decaying:,:] +1 - (1-self.r_lambda)
#         self.dirichlet_rew_params[tau,t,reward,:,:] += states * posterior_context[None,:]
#         for c in range(self.nc):
#             for state in range(self.nh):
#                 #self.generative_model_rewards[:,state,c] = self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,state,c].sum()
#                 self.generative_model_rewards[tau,t,:,state,c] = self.dirichlet_rew_params[tau,t,:,state,c]#\
#                 # ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
#                 #         -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
#                 self.generative_model_rewards[tau,t,:,state,c] /= self.generative_model_rewards[tau,t,:,state,c].sum()
#             self.rew_messages[tau,t+1:,:,t+1:,c] = self.prior_rewards.matmul(self.generative_model_rewards[tau,t,:,:,c])[None,:,None]

        dirichlet_rew_params = self.dirichlet_rew_params[0].clone().to(device)#.detach()        
        # dirichlet_rew_params = ar.ones_like(self.dirichlet_rew_params_init)#self.dirichlet_rew_params_init.clone()
        # dirichlet_rew_params[:,:self.non_decaying] = self.dirichlet_rew_params[-1][:,:self.non_decaying]

        curr_forgetting_factor = (self.r_lambda*self.mask[tau])[None,None,None,:,:]*self.posterior_context[-1][None,None,:,:,:]
        
        dirichlet_rew_params[:,self.non_decaying:,:,:] = ((1-curr_forgetting_factor) * self.dirichlet_rew_params[-1][:,self.non_decaying:,:,:,:]) \
                                                            +(1 - (1-curr_forgetting_factor))
        #dirichlet_rew_params[reward[0],:,:,:] += states #* posterior_context[None,:]

        vec_rewards = ar.eye(self.nr)[:,reward]
        vec_subjects = ar.eye(self.nsubs)
        matrix_index = ar.einsum('rn,nm->rm', vec_rewards, vec_subjects)
        if self.hidden_state_mapping:
            mapped_states = ar.einsum('hm,hcnk->mcnk', self.state_mapping_one_hot[tau], states)
            addition = mapped_states[None,...]*matrix_index[:,None,None,None,:]*self.mask[None,None,None,tau,...]*self.posterior_context[-1][None,None,:,:,:]
        else:
            addition = states[None,...]*matrix_index[:,None,None,None,:]*self.mask[None,None,None,tau,...]*self.posterior_context[-1][None,None,:,:,:]

        new_rew_params = dirichlet_rew_params + addition

        generative_model_rewards = new_rew_params / new_rew_params.sum(dim=0)[None,...]
        self.dirichlet_rew_params.append(new_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))

        if self.store_internal_variables:
            if t==self.T-1:
                self.generative_model_rewards_mb.append(generative_model_rewards)

        #return dirichlet_rew_params

    def update_beliefs_dirichlet_cached_rew_params(self, tau, t):

        assert(t==self.T-1)

        chosen = ar.argmax(self.posterior_policies[-1], dim=0)
        vec_pol = ar.nn.functional.one_hot(chosen, num_classes=self.npi).permute(3,0,1,2).float()

        curr_forgetting_factor = (self.cached_r_lambda*self.mask[tau])[None,None,:,:]*self.posterior_context[-1][None,None,:,:,:]

        # going only to T-2 leaves out the reward for t=0, which is reasonable since no action has been selected at that point
        for k in range(0,self.T-1):

            tp = -(self.T-2) + k
            reward = self.rewards[tp]

            vec_rewards = ar.nn.functional.one_hot(reward, num_classes=self.nr).permute(1,0).float()

            matrix_index = vec_rewards[:,None,None,None,:] * vec_pol[None,...]

            addition = matrix_index*self.posterior_context[-1][None,None,:,:,:]*self.mask[None,None,tau,...]

            cached_rew_params = (1 - curr_forgetting_factor) * self.cached_reward_params[-1] + addition + curr_forgetting_factor * self.cached_reward_params[0]

            self.cached_reward_params.append(cached_rew_params)

            cached_rewards = cached_rew_params / cached_rew_params.sum(dim=0)[None,...]

            self.cached_rewards.append(cached_rewards)

            cached_preference = (cached_rewards * self.prior_rewards[:,None,None,None,None]).sum(dim=0)

            cached_policy_val = cached_preference / cached_preference.sum(dim=0)[None,...]

            self.cached_policy_val.append(cached_policy_val)

