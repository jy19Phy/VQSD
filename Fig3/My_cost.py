import torch

def cost_sumP0_order1_fun(Nq, P ):
	P = P.reshape( [2]*Nq )
	PSet = []
	for q in range(Nq):
		Pex = torch.transpose( P, 0,q).reshape(2,-1)
		P0 = torch.sum(Pex[0])
		PSet.append(P0)
	PSet = torch.stack(PSet)
	return PSet

def cost_sumP0_order2_fun(Nq, P):
	P = P.reshape( [2]*Nq )
	PSet = []
	for q in range(Nq):
		Pex = torch.transpose( P, 0,q).reshape(2,-1)
		P0 = torch.sum(Pex[0])
		PSet.append(P0)
	PSet = torch.square(torch.stack(PSet))
	return PSet

def cost_sumP0_order3_fun(Nq, P):
	P = P.reshape( [2]*Nq )
	PSet = []
	for q in range(Nq):
		Pex = torch.transpose( P, 0,q).reshape(2,-1)
		P0 = torch.sum(Pex[0])
		PSet.append(P0)
	PSet = torch.pow(torch.stack(PSet), 3)
	return PSet

def cost_sumP0_order4_fun(Nq, P):
	P = P.reshape( [2]*Nq )
	PSet = []
	for q in range(Nq):
		Pex = torch.transpose( P, 0,q).reshape(2,-1)
		P0 = torch.sum(Pex[0])
		PSet.append(P0)
	PSet = torch.pow(torch.stack(PSet), 4)
	return PSet

def cost_sumP0_order5_fun(Nq, P):
	P = P.reshape( [2]*Nq )
	PSet = []
	for q in range(Nq):
		Pex = torch.transpose( P, 0,q).reshape(2,-1)
		P0 = torch.sum(Pex[0])
		PSet.append(P0)
	PSet = torch.pow(torch.stack(PSet), 5)
	return PSet


def cost_fun(Nq, P ):
	P1Set = cost_sumP0_order1_fun(Nq, P)
	cost1 = torch.sum(P1Set)
	cost = cost1 
	return cost