
# import math
# from os import stat
# import numpy as np
# import torch 
import random
# from torch import nn
# from torch.nn  import functional as F 
# from torch.autograd import Variable
# import pennylane as qml

def wiresIDSet0_ini_fun():
	wiresIDSet0 = [[0,0]]
	return wiresIDSet0

def wiresID_ext_fun(wiresIDSet, Nq):
	wiresID = [random.randint(0, Nq-1), random.randint(0,Nq-1) ]
	# print(wiresID)
	wiresIDSet.append(wiresID)
	# print(wiresIDSet, len(wiresIDSet))
	return wiresIDSet

def wiresIDSet_fun(Depth, Nq):
	wiresIDSet =wiresIDSet0_ini_fun()
	for _ in range(Depth-1):
		wiresIDSet = wiresID_ext_fun(wiresIDSet, Nq)
	return wiresIDSet

def wiresIDSet_singlequbit_fun( Nq):
	wiresIDSet = []
	for q in range(Nq):
		wiresIDSet.append([q,q])
	return wiresIDSet

def wiresIDSet_brickwall_fun(Nq):
	wiresIDSet =[]
	for q in range(Nq):
		wiresIDSet.append([q,q])
	for q in range(Nq-1):
		wiresIDSet.append([q,q+1])
	return wiresIDSet



