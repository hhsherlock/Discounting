import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats

# function for getting new y data from updated params
def acquire_ydata(params):
    global prior
    global likelihood
    global multiply
    global estimation
    global percentage
    global deviation
    global time_percep
    global prior_decre
    # for the first graph
    a = params[0]
    prior_sigma = params[1]
    ll_value = params[2]
    sigma_rate = params[3]
    delay = params[4]

    prior = stats.norm.pdf(x, a/delay*10, prior_sigma)
    es_sigma = sigma_rate*np.sqrt(delay)
    likelihood = stats.norm.pdf(x, ll_value, np.sqrt(0.1*delay)*es_sigma)
   #  t = 1/(1+b*np.exp(-delay*0.1))

    # also for the other graphs
    mean = (a/delay*10*es_sigma**2 + ll_value*prior_sigma**2)/(
       es_sigma**2 + prior_sigma**2)
    sigma = ((es_sigma**2*prior_sigma**2)/
             (es_sigma**2 + prior_sigma**2))**0.5
    
    # this is for the first graph
    multiply = stats.norm.pdf(x, mean, sigma)
    
    # time perception and prior mean
    time_percep = sigma_rate*np.sqrt(delays)
    prior_decre = a/delays*10

    # other graphs y axis values
    estimation = []
    percentage = []
    for i in delays:
      # time perception for estimated value
      # t_sub = 1/(1+b*np.exp(-i*0.1))
      
      # ll estimated value
      estimation_value = (a/i*10*i*0.1*es_sigma**2 + ll_value*prior_sigma**2)/(
       i*0.1*es_sigma**2 + prior_sigma**2)
      estimation.append(estimation_value)
      # choosing LL percentage
      percentage.append((np.exp(estimation_value)/(np.exp(estimation_value) + np.exp(ss_value))))


# the values for the initial plot
params_names = ['μ$_{prior}$', 'σ$_{prior}$', 'μ$_{likeli}$', 'σ$_{likeli}$', 'delay']
# same sequence as the params_names
params = [1, 3, 50, 0.7, 10]
delays = np.linspace(1, 122, 1000)
ss_value = 20
x = np.linspace(-20, 80, 1000)

# plot the initial params
acquire_ydata(params)


# set the plot structure
fig = plt.figure(constrained_layout=False, figsize=(15,15))
gs = fig.add_gridspec(nrows=4, ncols=5, left=0.05, right=0.95, wspace=0.35, hspace=0.35)
# ax1, ax2 are on the left
ax1 = fig.add_subplot(gs[:2, :2], ylim = [0,0.5], xlabel = 'Reward value', ylabel = 'pdf')
ax2 = fig.add_subplot(gs[2:, :2], ylim = [0,55], xlabel = 'Delay', ylabel = 'Estimated LL value')

# ax3, ax4 are on the right
ax3 = fig.add_subplot(gs[:2, 3:], ylim = [0,1.1], xlabel = 'Delay', ylabel = 'Choose LL percentage (%)')
ax4 = fig.add_subplot(gs[2:, 3:], ylim = [0,25], xlabel = 'Delay', ylabel = 'Uncertainty delay relationship')

plt.show()


def update(val):
    # update curve
    global params
    global prior
    global likelihood
    global multiply
    global estimation
    global percentage
    global deviation
    for i in range(len(params)):
      params[i] = sliders[i].val 
    acquire_ydata(params)
    m1.set_ydata(prior)
    m2.set_ydata(likelihood)
    m3.set_ydata(multiply)
    m4.set_ydata(estimation)
    m5.set_ydata(percentage)
    m6.set_ydata(time_percep)
    # redraw canvas while idle
    fig.canvas.draw_idle()



def reset(event):
    global params
    global prior
    global likelihood
    global multiply
    global estimation
    global percentage
    global prior_decre
    global time_percep
    #reset the values
    for i in np.arange(len(params)):
      sliders[i].reset()
    acquire_ydata(params)
    # redraw canvas while idle
    fig.canvas.draw_idle()


m1, = ax1.plot (x, prior, label = 'prior')
m2, = ax1.plot (x, likelihood, label = 'likelihood')
m3, = ax1.plot (x, multiply, label = 'posterior')
m4, = ax2.plot(delays, estimation)
m5, = ax3.plot(delays, percentage)
m6, = ax4.plot(delays, time_percep)

ax1.axvline(x = ss_value, linewidth = 1, label='ss_value', linestyle='--')
ax2.axhline(y = ss_value, linewidth = 1, label='ss_value', linestyle='--')
ax1.legend(loc=2,prop={'size':10})

sliders = []

for i in range(len(params)):

    axamp = plt.axes([0.44, 0.6-(i*0.05), 0.12, 0.02])
    # Slider
    min = 0
    max = 10
    # sigmas
    if i == 1:
       min = 0.1
    # ll_value
    elif i == 2:
       max = 60
    elif i == 3:
       min = 0.1
       max = 4
    elif i == 4:
       min = 1
       max = 122

    s = Slider(axamp, params_names[i], min, max, valinit=params[i])
    sliders.append(s)

    
for i in range(len(params)):
    #samp.on_changed(update_slider)
    sliders[i].on_changed(update)

axres = plt.axes([0.44, 0.6-((len(params))*0.05), 0.12, 0.02])
bres = Button(axres, 'Reset')
bres.on_clicked(reset)


plt.show()