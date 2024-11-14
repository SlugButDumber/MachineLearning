import numpy as np
import torch

def gradNorm(net, layer, alpha, dataloader, num_epochs, lr1, lr2, lamda_vals, lamda=0, log=False):

    #Ignore

    """
    Args:
        net (nn.Module): a multitask network with task loss
        layer (nn.Module): a layers of the full network where appling GradNorm on the weights
        alpha (float): hyperparameter of restoring force
        dataloader (DataLoader): training dataloader
        num_epochs (int): number of epochs
        lr1（float): learning rate of multitask loss
        lr2（float): learning rate of weights
        log (bool): flag of result log
        lamda (float): Lambda for GradNorm
        lamda_vals (list): list of lambda values for each task
    """
    # init log
    if log:
        log_weights = []
        log_loss = []

    # Optimizer for multitask loss    
    # set optimizer
    optimizer1 = torch.optim.Adam(net.parameters(), lr=lr1)

    # start traning
    iters = 0
    net.train() # set model to train mode

    # Loopp starts 
    for epoch in range(num_epochs):
        # load data
        for data in dataloader: # dataloader is a list of data
            # cuda
            if next(net.parameters()).is_cuda:
                data = [d.cuda() for d in data]

            # Calculating Losses (Li(t))
            # forward pass
            loss, task_weights, last_weights = net(*data) # loss is a list of losses for each task

            # Calculating R overall
            R_overall = (((torch.sum(torch.norm(last_weights, p='fro', dim=1)))**2)/2)*(lamda)

            # Calculating R task specific vals
            R_tasks = []
            for i in range(len(loss)):
                R_tasks.append((((torch.sum(torch.norm(task_weights[i], p='fro', dim=1)))**2)/2)*(lamda_vals[i]))

            R_tasks = torch.nn.Parameter(torch.stack(R_tasks))

            # initial conditions
            if iters == 0:
                # init weights
                weights = torch.ones_like(loss) # weights is a tensor of ones with same shape as loss
                weights = torch.nn.Parameter(weights) # weights is added to the list of parameters for the nn now
                T = weights.sum().detach() # sum of weights (same as number of tasks for initial conditions)
                # set optimizer for weights
                optimizer2 = torch.optim.Adam([weights], lr=lr2) # optimizer for weights of shared layers loss
                # set L(0) initial loss
                l0 = loss.detach()

            # Calculating L(t)
            # compute the weighted loss
            weighted_loss = weights @ (R_tasks + loss) + R_overall# element wise multiplication of weights and loss (of matrices)

            # clear gradients of network
            optimizer1.zero_grad() # zero the gradients of the optimizer as we dont want to accumulate them like in RNNs

            # Calculating Del w L(t)
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True) # to calculate gradients of the weighted loss ??????????????????

            # Calculating Gw(i)(t)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(weights[i]*(loss+R_tasks)[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)

            # Calculating Li(t)/Li(0)
            # compute loss ratio per task
            loss_ratio = (loss+R_tasks).detach() / l0

            # Calculating ri(t)
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()

            # Calculating Gw_avg(t)
            # compute the average gradient norm
            gw_avg = gw.mean().detach()

            # Calculating constant term in Lgrad(t)
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()

            # Calculating Lgrad(t)
            gradnorm_loss = torch.abs(gw - constant).sum()

            # clear gradients of weights
            optimizer2.zero_grad()

            # Calculating Del wi Lgrad
            # backward pass for GradNorm
            gradnorm_loss.backward()


            # log weights and loss
            if log:
                # weight for each task
                log_weights.append(weights.detach().cpu().numpy().copy())
                # task normalized loss
                log_loss.append(loss_ratio.detach().cpu().numpy().copy())

            # update model weights
            optimizer1.step()
            
            # update loss weights
            optimizer2.step()

            # renormalize loss weights for sum to be = T
            weights = (weights / weights.sum() * T).detach()
            weights = torch.nn.Parameter(weights)
            optimizer2 = torch.optim.Adam([weights], lr=lr2)

            # update iters
            iters += 1

    
    # get logs
    if log:
        return np.stack(log_weights), np.stack(log_loss)