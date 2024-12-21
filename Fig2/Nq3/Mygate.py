# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch 




def random_onequbit_gate_fun(theta_para , Np=3, Nd=2):
	theta_para= theta_para.reshape(Np)
	theta_x = theta_para[0]
	theta_y = theta_para[1]
	theta_z = theta_para[2]
	sigma_x = torch.tensor([[0.0+0.0j, 1.0+0.0j],  [1.0+0.0j, 0.0+0.0j]])
	sigma_y = torch.tensor([[0.0+0.0j, 0.0-1.0j],  [0.0+1.0j, 0.0+0.0j]])
	sigma_z = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j,-1.0+0.0j]])
	A = theta_x*sigma_x+theta_y*sigma_y+theta_z*sigma_z 
	U = torch.matrix_exp(1.j*A)
	U = U.reshape(Nd,Nd)
	return U


def CNOT_gate( Nd = 2):
	CT_gate = torch.tensor(    [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        						[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        						[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        						[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )
	return CT_gate

def Ux_fun(theta):
	U11 = torch.cos(theta/2.0).reshape(-1,1)
	U12 = -1.j*torch.sin(theta/2.0).reshape(-1,1)
	U21 = U12
	U22 = U11
	Ux = torch.cat( (U11, U12, U21, U22), dim=-1 )
	return Ux.reshape(-1,4)

def Uy_fun(theta):
	U11 = torch.cos(theta/2.0).reshape(-1,1)
	U12 = -1.*torch.sin(theta/2.0).reshape(-1,1)+0.j
	U21 = torch.sin(theta/2.0).reshape(-1,1)
	U22 = U11
	Uy = torch.cat( (U11, U12, U21, U22), dim=-1 )
	return Uy.reshape(-1,4)

def Uz_fun(theta):
	U11 = torch.exp(-1.j*theta/2.0).reshape(-1,1)
	U12 = 0.0*theta.reshape(-1,1)
	U21 = U12
	U22 = torch.exp(1.j*theta/2.0).reshape(-1,1)
	U = torch.cat( (U11, U12, U21, U22), dim=-1 )
	return U.reshape(-1,4)

def haar_gate_fun(theta_para ):
	theta_para= theta_para.reshape(-1,3)
	omega = theta_para[:,0]
	theta = theta_para[:,1]
	phi  = theta_para[:,2]
	U11= torch.exp(-1.j*(phi+omega)*0.5) *torch.cos(theta/2.)
	U22= torch.exp( 1.j*(phi+omega)*0.5) *torch.cos(theta/2.)
	U12= -1.*torch.exp(1.j*(phi-omega)*0.5) *torch.sin(theta/2.)
	U21= torch.exp(-1.j*(phi-omega)*0.5) *torch.sin(theta/2.)
	U = torch.cat( (U11.reshape(-1,1),U12.reshape(-1,1),U21.reshape(-1,1),U22.reshape(-1,1)) , dim= -1)
	U = U.reshape(-1,2,2)
	return U


def random_twoqubit_gate_fun(theta_para , Np=15, Nd=2):
	theta_para= theta_para.reshape(Np)+0.0j
	sigma_I = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j, 1.0+0.0j]]).reshape(1,-1)
	sigma_x = torch.tensor([[0.0+0.0j, 1.0+0.0j],  [1.0+0.0j, 0.0+0.0j]]).reshape(1,-1)
	sigma_y = torch.tensor([[0.0+0.0j, 0.0-1.0j],  [0.0+1.0j, 0.0+0.0j]]).reshape(1,-1)
	sigma_z = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j,-1.0+0.0j]]).reshape(1,-1)
	sigma = torch.cat( ( sigma_x, sigma_y, sigma_z, sigma_I)  , dim=0 ).reshape(4,2,2)
	sigmaM = torch.einsum( "bij,klm-> bkiljm",sigma,sigma).reshape(16,4,4)
	basis = sigmaM[0:-1]
	A = torch.einsum( "i, ik-> k", theta_para, basis.reshape(15,-1)).reshape(4,4)
	U = torch.matrix_exp(1.j*A)
	U = U.reshape(4,4)
	return U

def random_Nqubit_gate_fun(theta_para , Nq, Nd=2):
	theta_para= theta_para.reshape(4**Nq-1)+0.0j
	sigma_I = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j, 1.0+0.0j]]).reshape(1,-1)
	sigma_x = torch.tensor([[0.0+0.0j, 1.0+0.0j],  [1.0+0.0j, 0.0+0.0j]]).reshape(1,-1)
	sigma_y = torch.tensor([[0.0+0.0j, 0.0-1.0j],  [0.0+1.0j, 0.0+0.0j]]).reshape(1,-1)
	sigma_z = torch.tensor([[1.0+0.0j, 0.0+0.0j],  [0.0+0.0j,-1.0+0.0j]]).reshape(1,-1)
	sigma = torch.cat( ( sigma_x, sigma_y, sigma_z, sigma_I)  , dim=0 ).reshape(4,2,2)
	sigmaM = sigma
	for q in range(1,Nq):
		sigmaM = torch.einsum( "bij,klm-> bkiljm",sigmaM,sigma).reshape(4**(q+1),Nd**(q+1),Nd**(q+1))
	basis = sigmaM[0:-1]
	A = torch.einsum( "i, ik-> k", theta_para, basis.reshape(4**Nq-1,-1)).reshape(Nd**Nq,Nd**Nq)
	U = torch.matrix_exp(1.j*A)
	# print(U.shape)
	return U

def CNOT_brick_layer_fun(Nq, Nd=2):
	if Nq<2:
		print('error with Nq', Nq)
	CNOT_set = []
	CNOT = CNOT_gate().reshape(4,4)
	CNOT_brick_layer = CNOT
	for q in range(1,Nq-1):
		id = torch.eye(Nd)
		CNOT_brick_layer = torch.kron(CNOT_brick_layer, id)
		# print(CNOT_brick_layer.size())
		CNOTnext = torch.kron(torch.eye(Nd**(q)), CNOT)
		# print(CNOTnext.size())
		CNOT_brick_layer = torch.matmul(CNOT_brick_layer, CNOTnext)
	return CNOT_brick_layer
	

if __name__ == '__main__':
	# torch.set_num_threads(1)
	torch.set_default_dtype(torch.float64)
	#======================================================

	CNOT_brick_layer = CNOT_brick_layer_fun(Nq=3)
	print(CNOT_brick_layer.shape)
	# U = Gate_ensemble(gateID=4)
	# param = torch.rand(4,3)
	# UhaarSet = haar_gate_fun(param)
	# print(UhaarSet.size())
	# Nq = 3
	# param = torch.rand(4**Nq-1)+0.j
	# U = random_Nqubit_gate_fun(theta_para=param, Nq=Nq )
	U = CNOT_brick_layer
	UUD = torch.einsum( 'ij, kj-> ik', U, torch.conj(U))
	print(UUD)
	
	


	


