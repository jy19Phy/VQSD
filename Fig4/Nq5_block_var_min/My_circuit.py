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
from My_cost import cost_fun


def arbitrary_density_matrix_circuit( Nq, rho_matrix):	
	wires = [i for i in range(Nq)]
	qml.QubitDensityMatrix(rho_matrix, wires)

def my_action_gate_function(wiresID, params ):
	# print("wiresID", wiresID)
	# print("params", params)
	if wiresID[0]==wiresID[1]:
		qml.Rot(params[0],params[1],params[2],wires=wiresID[0])
	else:
		qml.CNOT(wires = wiresID)

def my_quantum_circuit_function(Nq, rho_matrix, wiresIDSet, paramsSet ):
	arbitrary_density_matrix_circuit( Nq, rho_matrix)
	paramsSet  = paramsSet.reshape(-1,3)
	for  l  in range(len(wiresIDSet)):
		wiresID = wiresIDSet[l]
		param = paramsSet[l]
		my_action_gate_function(wiresID, param)
	# print("state", qml.state())
	return  qml.state()
	# return qml.probs(wires= qID )


def measurement_circuit_fun(Nq, rho_matrix, wiresIDSet, paramsSet):
	dev = qml.device("default.mixed", wires=Nq)
	circuit_run = qml.QNode( my_quantum_circuit_function, dev, interface="torch" )
	rho = circuit_run(Nq, rho_matrix, wiresIDSet, paramsSet)	
	P = torch.real(rho.diag() )
	# print('rho', rho)
	# print(qml.draw(circuit_run, decimals=None)(Nq, rho_matrix, wiresIDSet, paramsSet))
	# print(qml.draw(circuit_run)(Nq, rho_matrix, wiresIDSet, paramsSet,qID))
	# print("P", P)
	return  P, rho





if __name__ == '__main__':
	Nq  = 2

	# rho_matrix = generat_random_densitymatrix_fromdia(Nq)
	# rho_system = generat_random_densitymatrix_fromdia(TotalNq)
	# torch.save(rho_system, "./Data/rho_N"+str(TotalNq)+".pt")
	# np.savetxt("./Data/rho_N"+str(TotalNq)+".csv",rho_system.detach().numpy())
	# rho_matrix = torch.load("./Data/rho_N"+str(Nq)+".pt")
	# eigE = torch.linalg.eigvals(rho_matrix)
	# print('EigE',torch.real(eigE) )
	# # rho_matrix = generate_pure_density_matrix(1, Nq, Nd=2).reshape(2**Nq, 2**Nq)
	# # rho_matrix = torch.tensor([[ 0.25+0.0000j,  0.0j,  0.0j, 0.0j],
    # #     [ 0.0j, 0.25+0.0j,0.0j,0.0j],
    # #      [ 0.0j, 0.0j,0.25+0.0j,0.0j],
	# # 	  [ 0.0j,0.0j,0.0j,0.25+0.0j]])
	# # print(rho_matrix)
	# purity = torch.real(torch.einsum( 'ij,ji', rho_matrix, rho_matrix))
	# print("purity",purity)


	# # wiresIDSet = [[0,0]]
	# # wiresIDSet = wiresIDSet_fun(Depth, Nq)
	# # wiresIDSet= wiresIDSet_singlequbit_fun( Nq)
	# wiresIDSet = wiresIDSet_brickwall_fun(Nq)*5
	# # print(wiresIDSet)

	# paramsSet = torch.randn( (len(wiresIDSet),3), requires_grad=True )

	# P, rho = measurement_circuit_fun(Nq, rho_matrix, wiresIDSet, paramsSet)
	# # print( P )

	# optimizer = torch.optim.Adam([paramsSet], lr=0.002)
	# epoch_size =10001

	# for epo in range(epoch_size):
	# 	optimizer.zero_grad()
	# 	P, rho = measurement_circuit_fun(Nq, rho_matrix, wiresIDSet, paramsSet)
	# 	loss = cost_fun(Nq, P )
	# 	loss.backward()
	# 	optimizer.step()
	# 	if epo % 5000 ==0 or epo == epoch_size-1:
	# 		print("epo= ", epo,"/"+str(epoch_size-1)+"\tloss=", loss.item(), '\tP', P.tolist()) 
	# print(P.reshape(-1))
	# print('rho', rho)


	



	