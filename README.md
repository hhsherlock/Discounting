# Simulation and Inference of discounting process

<details>
  <summary markdown="span">Old python and jupyter notebook files on the Sascha suggested model with normal distribution and discounting rate. MCMC and SVI. (until 23.01.2024) </summary>

I stopped writing jupyter notebook and normal python files at the same time. Too much work... even though I do not like how mess jupyter notebook is... I guess it is just me that's messy. BUT I will go back to python files after everything works. 
## Basic Hyperbolic Model (MCMC)
- **basic_hyperbolic_MCMC_agent.py**: The Agent has a behaviour of estimating the large later (LL) value 
and then generate actions by sampling from the distribution and comparing 
with the small earlier (SS) value.  <br/><br/> _Updates on 1/3/2024_: The estimation behaviour is changed to get the cdf of the normal distribution to get the possibility of choosing the large later one, and then a Bernoulli distribution determines if this action will be choosing the early one or the later one. 

- **basic_hyperbolic_MCMC_sim.py**: Mostly about creating the conditions of the choice, including the early smaller value, large later value and the delay time. And generate artifical choices under the discounting rate of 1/16 and estimation deviation of 4.

- **basic_hyperbolic_MCMC_inference.py**: Created a non-hierarchical model as picture shown below. Discounting rate and the deviation are sampled and it is pretty much the same as the agent behaviour, expect for using the generated data from the simulation for observations. <br/><br/>

- **basic_MCMC.ipynb**: Just a combination of all three files above in jupyter notebook, in case anyone likes to use jupyter notebook. And the results are shown as well.
_Note: The Binomial is used as a Bernoulli. (I think it is the same with only one decision.)_


![picture for the most basic and plain model](/images/plain.png)


## Hyperbolic Model with Prior Estimation (MCMC)
Reference paper: https://doi.org/10.1038/s41467-020-16852-y

- **paper2020_hyperbolic_MCMC_agent.py**: This agent follows a discounting rate of division of estimation variation (noise) and prior variation. Only the estimation function is a bit different and the agent also has another property of mean of the prior but so far I do not know where I should put it in the estimation function. 

- **paper2020_hyperbolic_MCMC_sim.py**: It is the same as the basic hyperbolic only with different import agent file.

- **paper2020_hyperbolic_MCMC_inference.py**: The model and the agent initialise parameters are different. The mean of prior is set to zero for now. The model is changed same as the agent estimation function.

- **paper2020_MCMC.ipynb**:
Same as basic model, just a jupyter notebook of the file combination. 

## Estimation variance changes with time (MCMC)
Idea from Ben and also from the paper https://doi.org/10.1038/s41467-020-16852-y. The estimation variance changes linearly with the delay.

- **basic_estimate_linear_MCMC.ipynb**: Cannot say this inference is behaving nicely. The discounting rate is pretty stable as before but the initial and increase rate do not have proper results. (01.10.2024) I just changed the sample distributions of init_dev and dev_rate from HalfCauchy to both Gamma distributions. They look better now. But it still is not better than the former. 🤨 Question 🤨: I guess the distribution choosing does determine a lot about the posterior distribution. But how do I know what to choose if I know almost nothing about the prior? (11.01.2024)

- **basic_estimate_exponen_MCMC.ipynb**: I am getting bored of this...
- **paper2020_estimate_linear_MCMC.ipynb**: Same as above.

Overall I think it is very important to generate good simulated data otherwise MCMC can not get good confidence. However, how to the parameters to generate good data needs guessing and trying and I am bored so let me just do something else for a while. (11.01.2024)

---
(More complicated model will be added in the future. Guys, I did it! I ran the basic_hyperbolic model and paper2020_hyperblic model. They work!)


## Basic Hyperbolic Model (SVI) (get helped from Sia)
- **basic_hyperbolic_SVI.ipynb**: I can never make svi work, with my own guide the elbo does not even decrease, I have to use auto guide which is called "auto_g" then it goes down but only to a certain extent. It stoppped at around 925. You can see that in this file. (09.01.2024) False alarm everyone it actually works! (10.01.2024)

</details>

<details>
  <summary markdown="span">Tried to use free energy to model. Some simulations (until 01.02.2024) </summary>
  
## New stuffs about generative model and free energy
- **graph_of_dep**: tested the influence of some independently changed variables on the choosing possibility. Results see images:
![delay](/images/delay.png)
![estimation_variance](/images/es_var.png)
![prior_variance](/images/prior_dev.png)
![prior_mean](/images/prior_mean.png)

</details>

## Folder Structures
- **images**: simulated data.
- **MCMC**: All MCMC models. Only multiply_MCMC is Sarah's multiplication of two distributions model. The others are all Sascha's model but with different variations.
- **Python_files**: Non important. Some python files which are the same the jupyter notebook files.
- **simulation**: Simulations of ideas and models.
- **SVI**: All svi models. They have a function file called agent_simulation.py all files use this to simulate agents. There is one test.ipynb, it is non important just to test some dimensional arrays. 

## Old-old Updates (12.02.2024)
- Seems like svi with only normal distributions and learning rate of 0.05 works like magic for multiply_svi. 
- The parameter recovery I try the model like the above. 
- I am also working on the dynamical ploting of Sarah's multiplication model. 

## Old Updates (16.02.2024)
Matching to the old updates.
- Okay I was wrong, it was actually pretty bad, i did not know ELBO is bad if it does not go down much.
- Tried parameter recovery and worked (only the for loops) but I used all combinations which is stupid and time consuming and the ELBOs are not good. 
- The Draghandler file in simulation folder at least works pretty well! 

<span style="color:green"> **Here are some writings about interesting findings I got from this interactive plot.**</span>

- **The means of the prior and likelihood change different parts of the discounting process:** the mean of the prior bounds the lower part of the discounitng curve which is located at the bigger delay times. Which means, the prior influences the discounting at the later times. By constrast the mean of the likelihood changes the earlier part of the discounting, it bounds the beginning of the discounting curve.  The discounting is slow when the difference between prior and posterior is narrow. As long as the difference is not too big and the discounting rate (controlled mostly of the deviations) is not too small, it always reaches almost the prior mean at the later part of the discounting process. 

- **Deviations change the curvature of the discounting:** the two deviations change the curvature in opposite directions. The larger prior deviation the less the discounting. The larger the liklihood deviation the more the discounting. Which means, the more uncertainty the person of the future outcome the more they discount, or the more certain the person about the prior the more they discount as well. It can be related to the personality traits for example one of the five big personalities,  openness, makes people be less certain of the past, anything can happen in the future. However, from the graph, we can know it is not about one side only. It is the balance of two or more elements. If a person with large openness triat also put big uncertainty on the future. The discounting rate will still be similar to the people who has less openness trait but also be more certain about the future likelihood. 

## New Update (22.02.2024)
- **Added data_analysis.ipynb**: It is used for analysing the csv file.
- **multiply_SVI_parameter_recovery_sample.py**: Only use for running on server with nohup.
- **multiply_SVI_real_data.ipynb**: Run with the real data. So far only get one person data. The two sigmas are quite big and put them into the draghandler in simulation folder. The discounting curve is almost a straight line. (will do hierarchical modelling in the future and compare different context)
- **SVI works fine now**: The only model that works okay is the "normal_log_model.py" which is only normal distribution and from log transfer back to real parameters for two sigmas. Mean_u is like normal, normal distribution. And I also set params in model which has the same names as in the guide (every variable ends with a _q). It works the best with the same name in model and guide and let them chase each other. 



