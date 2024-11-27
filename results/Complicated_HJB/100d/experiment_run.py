# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# import the required libraries
from equations.equations import Complicated_HJB
from tests.NormalSphere import NormalSphere
from tests.SimpleUniform import SimpleUniform
from tests.ConvergenceRate import ConvergenceRate
from tests.GPRate import GPRate
from solvers.MLP import MLP
from solvers.ScaSML import ScaSML
from models.GP import GP_Complicated_HJB as GP
import numpy as np
import torch
import wandb
import deepxde as dde
import jax
from jax import config


#fix random seed for dde
dde.config.set_random_seed(1234)
#use jax backend
dde.backend.set_default_backend('jax')
# fix random seed for numpy
np.random.seed(1234)
# fix random seed for jax
jax.random.key = jax.random.PRNGKey(1234)
# fix random seed for torch
torch.manual_seed(1234)
#set default data type
torch.set_default_dtype(torch.float64)
# device configuration for torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    # get GPU name
    gpu_name = torch.cuda.get_device_name()
# device configuration for jax
config.update("jax_enable_x64", True)

#initialize wandb
wandb.init(project="Complicated_HJB", notes="100 d", tags=["Gaussian Process"],mode="disabled") #debug mode
# wandb.init(project="Complicated_HJB", notes="100 d", tags=["Gaussian Process"]) #working mode
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=Complicated_HJB(n_input=3,n_output=1)

#initialize the normal sphere test
solver1=GP(equation=equation) #GP solver
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaSML(equation=equation,GP=solver1) #ScaSML object


#run the test for NormalSphere
test1=NormalSphere(equation,solver1,solver2,solver3)
rhomax=test1.test(r"results/Complicated_HJB/100d")
#run the test for SimpleUniform
test2=SimpleUniform(equation,solver1,solver2,solver3)
test2.test(r"results/Complicated_HJB/100d")
#run the test for ConvergenceRate
test3=ConvergenceRate(equation,solver1,solver2,solver3)
test3.test(r"results/Complicated_HJB/100d")
#run the test for GPRate
test4=GPRate(equation,solver1)
test4.test(r"results/Complicated_HJB/100d")


#finish wandb
wandb.finish()