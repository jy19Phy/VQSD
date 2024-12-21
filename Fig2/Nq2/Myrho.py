# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch
import random
from Mytheta import uniform_rand_theta
from Mygate import random_Nqubit_gate_fun
from Mystate import random_stateSet_fun

def generate_pure_density_matrix(batch, Nq, Nd=2):
	stateSet = random_stateSet_fun(batch= batch,Nq= Nq).reshape(batch, Nd**Nq)
	stateBar = torch.conj(stateSet)
	density_matrix= torch.einsum( 'bi,bj-> bij', stateSet, stateBar)
	return density_matrix.reshape(batch, -1)

def generate_random_density_matrix(matrix_size):
	# 生成随机复数矩阵
	random_matrix = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
	# 使其成为厄米特矩阵
	hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2
	# 归一化矩阵
	density_matrix = hermitian_matrix / np.trace(hermitian_matrix)
	# 检查矩阵是否半正定
	while not np.all(np.linalg.eigvals(density_matrix) >= 0):
		random_matrix = np.random.rand(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
		hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2
		density_matrix = hermitian_matrix / np.trace(hermitian_matrix)
	return torch.tensor(density_matrix, dtype=torch.complex64)

def random_densitymatrix_fun(batch, Nq):
	matrix_size = 2**Nq
	rho_Data = [ generate_random_density_matrix(matrix_size) for _ in range(batch)]
	return torch.stack(rho_Data).reshape(batch, -1)


def geneate_random_eigE(num_occupied, matrix_size):
	# eigE = torch.zeros(matrix_size)
	# random_eigE = torch.rand(num_occupied)
	random_eigE = torch.rand(matrix_size)
	# random_eigE = torch.tensor([0.1,0.2,0.3,0.4])
	Norm = torch.sum(random_eigE)
	# eigE[:num_occupied]=random_eigE/Norm
	Norm_random_eigE = random_eigE/Norm
	print("EigE", Norm_random_eigE)
	return Norm_random_eigE

def generat_random_dig_densitymatrix(Nq, Nd = 2):
	matrix_size = Nd**Nq
	# num_occupied = 1
	num_occupied = matrix_size
	# num_occupied = random.randint(1,matrix_size)
	dia_random_matrix = torch.zeros(matrix_size, matrix_size) + 0.j 
	Norm_random_eigE = geneate_random_eigE(num_occupied, matrix_size)
	for i in range(matrix_size):
		dia_random_matrix[i,i]= Norm_random_eigE[i]
	return dia_random_matrix

def random_dig_densitymatrix_fun(batch, Nq, Nd = 2):
	dia_rho_Data = [ generat_random_dig_densitymatrix(Nq) for _ in range(batch)]
	return torch.stack(dia_rho_Data).reshape(batch, -1)


def generat_random_densitymatrix_fromdia(Nq, Nd = 2):
	rhoDia = generat_random_dig_densitymatrix(Nq)
	Urand = random_Nqubit_gate_fun(theta_para = uniform_rand_theta(4**Nq-1) , Nq = Nq)
	Udagger = torch.transpose(torch.conj ( Urand), 0, 1)
	rho = torch.einsum( 'ij,jl,ln-> in ', Urand, rhoDia, Udagger)
	rho = rho/torch.trace(rho)
	return rho

def random_densitymatrix_fromDia_fun(batch, Nq, Nd = 2):
	rho_Data = [ generat_random_densitymatrix_fromdia(Nq) for _ in range(batch)]
	return torch.stack(rho_Data).reshape(batch, -1)






if __name__ == '__main__':
	torch.set_num_threads(1)

	Nd = 2
	Nq = 1
	batch = 1000
	# rho = generate_pure_density_matrix( batch= batch, Nq =Nq)
	
	# rho = random_densitymatrix_fun(batch=batch,Nq=Nq)
	# rho2 =random_dig_densitymatrix_fun(batch=batch, Nq=Nq)
	rho =random_densitymatrix_fromDia_fun(batch=batch, Nq=Nq)
	print(rho.shape)
	print(torch.einsum('bii->b',rho.reshape(batch,Nd**Nq,Nd**Nq)))
	print(torch.einsum('bij, bji->b',rho.reshape(batch,Nd**Nq,Nd**Nq), rho.reshape(batch,Nd**Nq,Nd**Nq) ) )
	np.save("./Data/rhoD2Nq"+str(Nq)+"batch"+str(batch)+".npy", rho)
	np.savetxt("./Data/rhoD2Nq"+str(Nq)+"batch"+str(batch)+".csv",rho)

	





		

