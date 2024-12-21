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
from Myrho import generat_random_densitymatrix_fromdia, generate_pure_density_matrix, generate_random_density_matrix
from My_train import train_fun, convergence_fun
from Mymonitor import time_string_fun




if __name__ == '__main__':
	print("=========================================================================================\n")

	statetime= time_string_fun()

	Nq = 2
	rho_matrix = generat_random_densitymatrix_fromdia(Nq)
	print(rho_matrix.dtype)
	torch.save(rho_matrix, "./Data/rho_N"+str(Nq)+".pt")
	np.savetxt("./Data/rho_N"+str(Nq)+".txt",rho_matrix.detach().numpy())
	rho_matrix = torch.load("./Data/rho_N"+str(Nq)+".pt")

	purity = torch.real(torch.einsum( 'ij,ji', rho_matrix , rho_matrix ))
	print("System purity=",purity)
	np.savetxt("./Data/purity.txt", [purity.item()])

	eigE = torch.linalg.eigvals(rho_matrix)
	eigE_sort , index= torch.sort(torch.real(eigE))
	print('EigE',eigE_sort ,"\n")
	np.savetxt("./Data/eigE.txt", torch.cat( (torch.real(eigE[index]).reshape(-1,1) ,torch.imag(eigE[index]).reshape(-1,1)), dim=-1 ).detach().numpy(), fmt='%f')
	# cost_min = cost_min_fun(Nq, eigE_sort )

	# print('cost_min', cost_min)
	# np.savetxt("./Data/cost_min.txt", cost_min.detach().numpy())


	d_block = 2

	P, rho, loss = convergence_fun(Nq,rho_matrix, d_block)

	print("\nSystem purity=", purity.item(),'\t', torch.sum(torch.square(eigE_sort)))
	print('EigE=\t',eigE_sort )
	print('P_f=\t', torch.sort(P.reshape(-1)).values,"\n")
	
	print(statetime)
	endtime= time_string_fun()
	



	