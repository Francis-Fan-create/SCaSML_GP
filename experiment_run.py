# We will use this file to run the experiment. We will train the model and then test it on the NormalSphere test. We will also profile the test to see the time taken by each solver to solve the equation. We will save the profiler results and upload them to wandb.



# import the required libraries
from equations import GP_Example
from GP import GP
# from MLP import MLP
# from ScaSML import ScaSML
from MLP_full_history import MLP_full_history as MLP
from ScaSML_full_history import ScaSML_full_history as ScaSML
from NormalSphere import NormalSphere
from SimpleUniform import SimpleUniform
from ConvergenceRate import ConvergenceRate
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
wandb.init(project="GP_Example", notes="100 d", tags=["Adam_LBFGS training"],mode="disabled") #debug mode
# wandb.init(project="GP_Example", notes="100 d", tags=["Adam_LBFGS training"]) #working mode
wandb.config.update({"device": device.type}) # record device type

#initialize the equation
equation=GP_Example(n_input=101,n_output=1)

#initialize the normal sphere test
solver1=GP(equation=equation) #GP solver
solver2=MLP(equation=equation) #Multilevel Picard object
solver3=ScaSML(equation=equation,GP=solver1) #ScaSML object


#run the test for NormalSphere
test1=NormalSphere(equation,solver1,solver2,solver3)
rhomax=test1.test(r"results/GP_Example/100d")
#run the test for SimpleUniform
test2=SimpleUniform(equation,solver1,solver2,solver3)
test2.test(r"results/GP_Example/100d")
#run the test for ConvergenceRate
test3=ConvergenceRate(equation,solver1,solver2,solver3)
test3.test(r"results/GP_Example/100d")


#finish wandb
wandb.finish()
