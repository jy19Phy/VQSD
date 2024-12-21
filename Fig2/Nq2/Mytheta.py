import math
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable


def uniform_rand_theta(num_thetas):
	theta_min = 0
	theta_max = 2.*np.pi
	x = torch.rand(num_thetas)*(theta_max-theta_min)+theta_min
	return x 

def normal_rand_theta(num_thetas):
	x = torch.randn(num_thetas)
	return x 
