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


# def cost_fun(Nq, P ):
# 	P1Set = cost_sumP0_order1_fun(Nq, P)
# 	cost1 = P1Set[0]+P1Set[1]+P1Set[2]
# 	P2Set = cost_sumP0_order2_fun(Nq, P)
# 	cost2 = P2Set[0]+P2Set[1]
# 	P3Set = cost_sumP0_order3_fun(Nq, P)
# 	cost3 = P3Set[0]
# 	# P4Set = cost_sumP0_order4_fun(Nq, P)
# 	# cost4 = torch.sum(P4Set)
# 	cost = cost1 +cost2+cost3
# 	return cost

def cost_fun(Nq, P ):
	P1Set = cost_sumP0_order1_fun(Nq, P)
	costSet = []
	for order in range(1,Nq+1):
		PSet = torch.pow(P1Set, order)
		cost_order = torch.sum(PSet[0:(Nq-order+1)])
		costSet.append(cost_order)
	cost = torch.sum(torch.stack(costSet))
	return cost

		
# def cost_fun(Nq, P ):
# 	P1Set = cost_sumP0_order1_fun(Nq, P)
# 	cost1 = P1Set[0]+P1Set[1]+P1Set[2]
# 	P2Set = cost_sumP0_order2_fun(Nq, P)
# 	cost2 = P2Set[0]+P2Set[1]
# 	P3Set = cost_sumP0_order3_fun(Nq, P)
# 	cost3 = P3Set[0]
# 	# P4Set = cost_sumP0_order4_fun(Nq, P)
# 	# cost4 = torch.sum(P4Set)
# 	cost = cost1 +cost2+cost3
# 	return cost

# def cost_min_fun(Nq, P ):
# 	p_qubit = cost_sumP0_order1_fun(Nq, P )
# 	p_qubit, index = torch.sort(p_qubit.reshape(Nq) )
# 	print('p_qubit', p_qubit)
# 	cost1 = torch.sum(p_qubit)
# 	cost2 = torch.sum( torch.pow(p_qubit, 2)[0:2] )
# 	cost3 = torch.sum( torch.pow(p_qubit, 3)[0:1] )
# 	cost = cost1 +cost2 +cost3
# 	return cost.reshape(-1)

