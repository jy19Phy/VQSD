# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
from os import stat
import numpy as np
import torch 
import random
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
import pennylane as qml
from torch.optim import Adam

from My_circuit_config import wiresIDSet_fun, wiresIDSet_singlequbit_fun, wiresIDSet_brickwall_fun
from Myrho import generat_random_densitymatrix_fromdia, generate_pure_density_matrix
from My_train import train_fun
from Mymonitor import time_string_fun




if __name__ == '__main__':
	print("=========================================================================================\n")

	statetime= time_string_fun()

	Nq = 4

	rho_matrix = torch.load("./Data/rho_N"+str(Nq)+".pt")



	purity = torch.real(torch.einsum( 'ij,ji', rho_matrix , rho_matrix ))
	print("System purity=",purity)
	

	eigE = torch.linalg.eigvals(rho_matrix)
	eigE_sort , index= torch.sort(torch.real(eigE))
	print('EigE',eigE_sort ,"\n")

	rho_hermi = torch.transpose( torch.conj( rho_matrix), 0,1  )

	diff = torch.sum(torch.square(rho_matrix - rho_hermi) ) 
	print("hermi diff=", diff)



	
	



	