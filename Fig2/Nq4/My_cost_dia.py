import torch


def cost_fun(Nq, P ):
	P = P.reshape( -1 )
	cost = -1.*torch.sum(torch.pow(P,2))
	return cost

def cost_sumP0_order1_fun(Nq, P ):
	P = P.reshape( [2]*Nq )
	PSet = []
	for q in range(Nq):
		Pex = torch.transpose( P, 0,q).reshape(2,-1)
		P0 = torch.sum(Pex[0])
		PSet.append(P0)
	PSet = torch.stack(PSet)
	return PSet