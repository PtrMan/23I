import torch
from scipy.stats.distributions import chi2

# compute linear regression

# BUG  :  something is wrong with the order of matrix operations ... we can only work with less n samples than dimensions. But this isn't the case in the book I guess
class BayesianLinearRegression(object):
    def __init__(self):
        self.xDat = []

    # /param yDat python array of y data vector
    def calcExpectation(self, yDat):
        
        #y = torch.tensor([[1.0], [0.8], [0.8]])
        y = torch.tensor([yDat])
        y = torch.transpose(y,0,1)
        
        #X = torch.tensor([[1.0, 1.0], [1.0, 0.4], [1.0, 0.4]])
        X = torch.tensor(self.xDat)
        X = torch.transpose(X,0,1) # we need to transpose
        
        
        n = X.size()[0]
        k = X.size()[1] # k equals rank : see page 356
        
        #print(f'{n} x {k}') # DBG
        
        if n <= k:
            print('underdetermined matrix!')
        
        
        
        Vbeta = torch.transpose(X,0,1)@X
        betaHat = (torch.inverse(Vbeta) @ torch.transpose(X,0,1)) @ y
        
        temp0 = y - X@betaHat
        
        sSquare = ((1.0/(n-k)) * torch.transpose(temp0,0,1) @ temp0 ).item()
        #print(f'sSquare={sSquare}') # DBG
        
        
        # compute pdf of Inverse-chi-square function
        # see https://stackoverflow.com/questions/53019080/chi2inv-in-python
        sigmaSquare = chi2.ppf(sSquare, df=(n-k))
        #print(sigmaSquare) # DBG
        
        centralDistributionParam1 = betaHat
        centralDistributionParam2 = Vbeta*sigmaSquare
        #print(centralDistributionParam1) # DBG
        #print(centralDistributionParam2) # DBG
        
        # sample beta
        distr = torch.distributions.Normal(centralDistributionParam1, centralDistributionParam2)
        beta = distr.sample()
        
        #print(beta) # DBG
        
        
        
        
        
        # * compute expectation
        #print(X) # DBG
        
        expectationSum = 0.0
        for idx0 in range(X.size()[1]):
            temp0 = beta[idx0]@X[idx0]
            expectationSum += temp0.item()
        
        expectation = expectationSum
        del expectationSum
        
        #print(expectation) # DBG

        return expectation
        
        # map with logit to a value which is usable to cast as probabilities
        #temp0 = logitInv(expectation)
        
        #print(temp0)
