import torch
import numpy as np
from My_circuit_arbitrary import measurement_circuit_fun
from My_cost_dia import cost_fun, cost_sumP0_order1_fun



def train_fun(Nq,rho_matrix,v_total, epoch_size):

	
	P_finalSet =[]
	rho_finalSet =[]
	P_qubitSet =[]
	rho_offDia_finalSet =[]
	for n_v in range(v_total):
		paramsSet = torch.randn( 4**Nq-1, requires_grad=True )
		optimizer = torch.optim.Adam([paramsSet], lr=0.002)
		for epo in range(epoch_size):
			optimizer.zero_grad()
			P, rho = measurement_circuit_fun(Nq, rho_matrix,  paramsSet)
			loss = cost_fun(Nq, P )
			loss.backward()
			optimizer.step()
			if epo % 50 ==0 or epo == epoch_size-1:
				print("epo= ", epo,"/"+str(epoch_size-1)+"\tloss=", loss.item(), '\tP', P) 
				P_qubit= cost_sumP0_order1_fun(Nq, P)
				with open("./Res/lossv"+str(n_v)+".txt", "a+",  buffering=1000000) as file:
					file.write("\nepo=\t"+str( epo)+"\t\tloss= \t "+ str(loss.detach().numpy())+'\t\tP_q= '+ str(P_qubit.detach().numpy() ))
				with open("./Res/lossPv"+str(n_v)+".txt", "a+",  buffering=1000000) as file:
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
		rho_offDia_finalSet.append(torch.mean(torch.abs(rho_offDia).reshape(-1) ) )
		print('v=',n_v,'\tP_f=', torch.sort(P.reshape(-1)).values,"\n")
		P_qubit= cost_sumP0_order1_fun(Nq, P)
		P_qubitSet.append(P_qubit.reshape(-1))
		np.savetxt("./Res/P_final.txt",torch.stack(P_finalSet).detach().numpy(),fmt='%f')
		np.savetxt("./Res/rho_final.txt",torch.stack(rho_finalSet).detach().numpy(),fmt='%f')
		np.savetxt("./Res/P_qubit.txt",torch.stack(P_qubitSet).detach().numpy(),fmt='%f')

	np.savetxt("./Res/P_final.csv",torch.stack(P_finalSet).detach().numpy(),delimiter=',')
	np.savetxt("./Res/rho_final.csv",torch.stack(rho_finalSet).detach().numpy(),delimiter=',')
	np.savetxt("./Res/rho_final_offSum.csv",torch.stack(rho_offDia_finalSet).detach().numpy())
	
	

	np.savetxt("./Data/P_final.txt",torch.sort(P.reshape(-1)).values.detach().numpy(),fmt='%f')


	

	return P, rho