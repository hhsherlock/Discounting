import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

# Define a simple model
def model(data):
    mean = torch.tensor(0., device='cuda')  # Move mean to GPU
    scale = torch.tensor(1., device='cuda')  # Move scale to GPU
    with pyro.plate("data", len(data)):
        obs = pyro.sample("obs", dist.Normal(mean, scale), obs=data)

# Generate some synthetic data
data = torch.randn(100)

# Move the data to the GPU
data = data.to('cuda')

# Create an MCMC sampler
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=100000, warmup_steps=50000)

# Run inference
mcmc.run(data)

# Get samples
samples = mcmc.get_samples()
