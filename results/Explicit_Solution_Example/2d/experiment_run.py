# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.
import sys
import os
#add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# import the required libraries
from equations.equations import Explicit_Solution_Example
from tests.NormalSphere import NormalSphere
from tests.SimpleUniform import SimpleUniform
from tests.ConvergenceRate import ConvergenceRate
from tests.GPRate import GPRate
from solvers.MLP import MLP
from solvers.ScaSML import ScaSML
from models.GP import GP_Explicit_Solution_Example as GP
import numpy as np
import torch
import wandb
import deepxde as dde


#fix random seed for dde
dde.config.set_random_seed(1234)
#use pytorch backend
dde.backend.set_default_backend('pytorch')
# fix random seed for numpy
np.random.seed(1234)
#set default data type
torch.set_default_dtype(torch.float64)
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    # get GPU name
    gpu_name = torch.cuda.get_device_name()

#initialize wandb
wandb.init(project="Explicit_Solution_Example", notes="2 d", tags=["Gaussian Process"],mode="disabled") #debug mode
# wandb.init(project="Explicit_Solution_Example", notes="2 d", tags=["Gaussian Process"]) #working mode
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=Explicit_Solution_Example(n_input=3,n_output=1)

#initialize the normal sphere test
solver1=GP(equation=equation) #GP solver
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaSML(equation=equation,GP=solver1) #ScaSML object


# #run the test for NormalSphere
# test1=NormalSphere(equation,solver1,solver2,solver3)
# rhomax=test1.test(r"results/Explicit_Solution_Example/2d")
#run the test for SimpleUniform
test2=SimpleUniform(equation,solver1,solver2,solver3)
test2.test(r"results/Explicit_Solution_Example/2d")
#run the test for ConvergenceRate
test3=ConvergenceRate(equation,solver1,solver2,solver3)
test3.test(r"results/Explicit_Solution_Example/2d")
#run the test for GPRate
test4=GPRate(equation,solver1)
test4.test(r"results/Explicit_Solution_Example/2d")


#finish wandb
wandb.finish()