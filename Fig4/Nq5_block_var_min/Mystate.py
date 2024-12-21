# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import torch
# import resource

# def get_max_memory_usage():
# 	# 在你的程序中执行需要监测内存的代码
#     max_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#     # ru_maxrss 返回的单位是kilobytes（KB）
#     return max_memory / 1024 /1024 # 转换为 megabytes (GB)

# Nd = 2: number of degrees of freedom for single qubit
# Nq: number of qubits

def GHZ_state_fun(Nq ,  Nd= 2 ):
	WF = torch.zeros(Nd**Nq)*0.0j
	WF[0] = 1.0/np.sqrt(2.0)
	WF[-1] = 1.0/np.sqrt(2.0)
	index = [Nd]*Nq
	WF = WF.reshape(index)
	# index0 = [0]*Nq
	# print(index0, " ", 	WF[tuple(index0) ] )
	# index1 = [1]*Nq 
	# print(index1, " ", 	WF[tuple(index1) ] )
	return WF

def zero_state_fun( Nq, Nd=2 ):
	WF = torch.zeros(Nd**Nq)*0.0j
	WF[0] =1.0
	index = [Nd]*Nq 
	WF = WF.reshape(index)
	return WF

def one_state_fun( Nq, Nd=2):
	WF = torch.zeros(Nd**Nq)*0.0j
	WF[-1] =1.0
	index = [Nd]*Nq 
	WF = WF.reshape(index)
	return WF

def Computational_stateSet_fun(Nq,Nd=2):
	WFSet = torch.zeros(Nd**Nq, Nd**Nq)*0.0j
	for index in range(Nd**Nq):
		WFSet[index,index] =1.0
	return WFSet

def random_state_fun( Nq, Nd=2):
	WF = torch.rand(Nd**Nq)+torch.rand(Nd**Nq)*1.0j
	Nor= torch.real(torch.sum(torch.conj(WF)*WF))
	WFNor= WF/torch.sqrt(Nor)
	return WFNor

def random_stateSet_fun(batch , Nq, Nd=2):
	WFSet = torch.rand(batch,Nd**Nq)+torch.rand(batch, Nd**Nq)*1.0j
	Nor= torch.sum(torch.square(torch.abs(WFSet)), dim=-1).reshape(-1,1).repeat( 1, Nd**Nq)
	WFSetNor= WFSet/torch.sqrt(Nor)
	return WFSetNor


def singlequbit_random_stateSet_fun(batch , Nq=1, Nd=2):
	# theta and phi are randomly sampling between min and max
	theta = torch.rand(batch)*2.*np.pi 		# theta 	[0,2pi)
	phi = torch.rand(batch)*2.*np.pi 		# phi 		[0,2pi)
	a = torch.cos( theta*0.5)
	b = torch.sin(theta*0.5)*torch.exp(1.0j*phi)
	WFSet = torch.cat( (a.reshape(-1,1), b.reshape(-1,1)), dim=-1 ) 
	return WFSet

def singlequbit_uniform_stateSet_fun(batch , Nq=1, Nd=2):
	# theta and phi  are uniformly sampling between min and max. 
	batch  = np.floor(np.sqrt(batch))
	theta_min = 0
	theta_max = 2*np.pi
	theta = torch.arange(theta_min, theta_max, (theta_max-theta_min)/batch)
	phi_min = 0
	phi_max = 2*np.pi
	phi = torch.arange(phi_min, phi_max, (phi_max-phi_min)/batch)
	thetaBatch, phiBatch = torch.meshgrid(theta,phi, indexing='xy') # type: ignore[attr-defined] 
	a = torch.cos( thetaBatch*0.5)
	b = torch.sin(thetaBatch*0.5)*torch.exp(1.0j*phiBatch)
	WFSet = torch.cat( (a.reshape(-1,1), b.reshape(-1,1)), dim=-1 ) 
	return WFSet



if __name__ == '__main__':
	torch.set_num_threads(1)


	Res= singlequbit_random_stateSet_fun(batch=3 , Nq=1, Nd=2)
	print(torch.square(torch.abs(Res) ) )
	
	Res= singlequbit_uniform_stateSet_fun(batch=3 , Nq=1, Nd=2)
	print(torch.square(torch.abs(Res) ) )


	Nd = 2
	Nq = 2
	batch = 50
	state = random_stateSet_fun(batch=batch,Nq=Nq)
	print(state.shape)
	# print(torch.sum(torch.square(torch.abs(state)), dim = -1))
	np.save("./Datastate/state"+str(Nq)+"qubits"+str(batch)+"batch.npy", state)
	np.savetxt("./Datastate/state"+str(Nq)+"qubits"+str(batch)+"batch.csv",state)


	# state = GHZ_state_fun( Nq , Nd  )
	# state = GHZ_state_fun( Nq=Nq , Nd= Nd  )
	# state = GHZ_state_fun( Nq )	

	# print( state.shape )
	# print("Max Memory Usage:", get_max_memory_usage(), "GB")

	# stateSet = Computational_stateSet_fun(Nq = Nq)

	# for WFID in range( 2**Nq):
	# 	state = stateSet[WFID,:]
	# 	print(state)

	# state = random_state_fun( Nq= Nq )	
	# print("random_state: ", state)



		

