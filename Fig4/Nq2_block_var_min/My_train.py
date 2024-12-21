import torch
import numpy as np
from My_circuit_config import  wiresIDSet_brickwall_fun
from My_circuit import measurement_circuit_fun
from My_cost import cost_fun, cost_sumP0_order1_fun


def convergence_fun(Nq,rho_matrix, d_block):

	loss_i = 10.0
	d_loss = 1.0
	block = 0
	while (abs(d_loss)>1e-3 ):
		block = block+d_block
		P, rho, loss_f , P_qubit= train_fun(Nq,rho_matrix, block, d_block)
		print("block", block, 'loss', loss_f, '\tP_q= '+ str(P_qubit.detach().numpy()))
		d_loss = loss_f - loss_i
		loss_i = loss_f
	return P,rho, d_loss



def train_fun(Nq,rho_matrix, block, d_block):

	wiresIDSet = wiresIDSet_brickwall_fun(Nq)*block
	P_finalSet =[]
	rho_finalSet =[]
	P_qubitSet =[]
	rho_offDia_finalSet =[]
	n_v = block/d_block
	paramsSet = torch.randn( (len(wiresIDSet),3), requires_grad=True )
	if n_v>1 :
		paramsSet_i = torch.load("./Res/paramsv"+str(n_v-1)+".pt")
		paramsSet_f = torch.randn( ( (Nq*2-1)*d_block,3) )
		paramsSet_all = torch.cat( (paramsSet_i, paramsSet_f), dim= 0)
		paramsSet = paramsSet_all.detach().requires_grad_()
		# print(paramsSet_i.shape, paramsSet_f.shape, paramsSet.shape)

	optimizer = torch.optim.Adam([paramsSet], lr=0.002)
	epoch_size = 501
	for epo in range(epoch_size):
		optimizer.zero_grad()
		P, rho = measurement_circuit_fun(Nq, rho_matrix, wiresIDSet, paramsSet)
		loss = cost_fun(Nq, P )
		loss.backward()
		optimizer.step()
		if epo % 25 ==0 or epo == epoch_size-1:
			print("epo= ", epo,"/"+str(epoch_size-1)+"\tloss=", loss.item()) 
			P_qubit= cost_sumP0_order1_fun(Nq, P)
			with open("./Res/lossv"+str(n_v)+".txt", "a+",  buffering=1000000) as file:
				file.write("\nepo=\t"+str( epo)+"\t\tloss= \t "+ str(loss.detach().numpy())+'\t\tP_q= '+ str(P_qubit.detach().numpy() ))
			with open("./Res/lossPb"+str(n_v*d_block)+".txt", "a+",  buffering=1000000) as file:
				if epo != 0 :
					file.write("\n")
				p_square = torch.sum(torch.square( P)  )
				purity = torch.real(torch.einsum( 'ij,ji', rho , rho ))
				rho_offDia = rho.detach().reshape(2**Nq,2**Nq)
				rho_offDia = rho_offDia - torch.diag(  rho_offDia.diagonal())
				offD = torch.mean(torch.abs(rho_offDia).reshape(-1) )
				file.write("epo=\t"+str( epo)+"\tloss= \t "+ str(loss.detach().numpy())+
								'\tP^2_sum= \t'+str(p_square.detach().item() )+
								'\tpurity =\t'+str(purity.detach().item()) +
								'\toffD = \t'+ str(torch.real(offD).detach().item() ) )
								
				
	torch.save(paramsSet,"./Res/paramsv"+str(n_v)+".pt")
	P_finalSet.append(torch.sort(P.reshape(-1)).values)
	rho_finalSet.append(rho.reshape(-1))
	rho_offDia = rho.detach().reshape(2**Nq,2**Nq)
	rho_offDia = rho_offDia - torch.diag(  rho_offDia.diagonal())
	rho_offDia_finalSet.append(torch.mean( torch.abs( rho_offDia).reshape(-1) ))
	# print('v=',n_v,'\tP_f=', torch.sort(P.reshape(-1)).values,"\n")
	P_qubit= cost_sumP0_order1_fun(Nq, P)
	P_qubitSet.append(P_qubit.reshape(-1))
	
	np.savetxt("./Res/P_final.csv",torch.stack(P_finalSet).detach().numpy(),delimiter=',')
	np.savetxt("./Res/rho_final.csv",torch.stack(rho_finalSet).detach().numpy(),delimiter=',')
	np.savetxt("./Res/rho_final_offMean.csv",torch.stack(rho_offDia_finalSet).detach().numpy())
	np.savetxt("./Data/P_final.txt",torch.sort(P.reshape(-1)).values.detach().numpy(),fmt='%f')

	return P, rho, loss, P_qubit