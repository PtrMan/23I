import torch
import numpy
from math import sqrt, exp
import random

def assert2(v, msg):
    if not v:
        print(f'assert failed! msg={msg}')
        raise Exception(msg)



# implementation of position encoding

# ChatGPT prompt "implement code to generate position encoding in python!"
# inputMaxLen The maximum length of the input sequence
# dimModel The dimensionality of the model
def calcPositionalEncoding(inputMaxLen, dimModel):
    """
    Calculate the positional encoding matrix for a given input length and model dimensionality.
    """
    position = numpy.arange(inputMaxLen)[:, numpy.newaxis]
    div_term = numpy.exp(numpy.arange(0, dimModel, 2) * -(numpy.log(10000.0) / dimModel))
    sinusoid = numpy.zeros((inputMaxLen, dimModel))
    sinusoid[:, 0::2] = numpy.sin(position * div_term)
    sinusoid[:, 1::2] = numpy.cos(position * div_term)
    t0 = torch.from_numpy(sinusoid)
    t0 = t0.to(torch.float32)
    t0 = torch.transpose(t0,0,1) # transpose to make indexing easier
    return t0



# helper to transpose matrix
def transpose2(v):
    return torch.transpose(v, 0, 1)

# helper to convert from 1d vector to 2d matrix
def matConv1dTo2d(v):
    # https://stackoverflow.com/questions/43328632/pytorch-reshape-tensor-dimension
    return v.unsqueeze(0)


# DEPRECATED
"""
# build Toeplitz kernel matrix by values (take values from vector)
def makeToeplitzKernel(tWindow):
    shapeDim = tWindow.shape[0]
    s1 = torch.zeros((shapeDim,shapeDim,))
    for iIdx in range(shapeDim):
        s1 = s1 + torch.diag( torch.ones(shapeDim-iIdx)*tWindow[iIdx], -iIdx )
    return s1
"""


# build Toeplitz kernel matrix by values (take values from vector)
# the result matrix doesn't necessarily have to be a square matrix!
# /param X is the other dimension
def makeToeplitzKernel2(tWindow, D):
    L = tWindow.shape[0]
    z = torch.zeros((L,D,))
    for iIdx in range(L):
        z0 = torch.diag( torch.ones(L-iIdx)*tWindow[iIdx], -iIdx ) # create diagonal matrix
        z1 = z0[0:L, 0:D] # cut of the part we need for the final result
        z = z + z1
    return z


# module for self-attention
class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, ctxLen, dk):
        super(SelfAttentionLayer, self).__init__()
        self.dk = dk
        
        shape = (ctxLen,dk)
        ## weights for 'query', 'key', 'value'
        self.wq = torch.nn.Parameter(torch.rand(shape)*0.05)
        self.wk = torch.nn.Parameter(torch.rand(shape)*0.05)
        self.wv = torch.nn.Parameter(torch.rand(shape)*0.05)
    
    # returns matrix which has width of self.dk
    def forward(self, x):
        # BEGIN self attention layer
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        
        t0 = (q @ transpose2(k)) * sqrt(self.dk)
        t1 = torch.nn.functional.softmax(t0, dim=0)
        t2 = t1 @ v
        # END    self attention layer

        return t2

# nonlinear
# OUTDATED
"""
class NonlinearA(torch.nn.Module):
    def __init__(self, inputDim):
        super(NonlinearA, self).__init__()

        layer0NonlinearDim = 180 #170 #152 #52 # dimensionality of nonlinear layer

        self.nonlinearLayer0_0 = torch.nn.Parameter(torch.rand((inputDim,layer0NonlinearDim,))*0.05)
        self.nonlinearLayer0_0_bias = torch.nn.Parameter(torch.rand((layer0NonlinearDim))*0.005)
        self.nonlinearLayer0_1 = torch.nn.Parameter(torch.rand((layer0NonlinearDim,embeddingDim,))*0.05)

    def forward(self, x):
        t9 = x @ self.nonlinearLayer0_0
        t6 = torch.nn.functional.relu(t9)
        t7 = t6 + self.nonlinearLayer0_0_bias # add bias
        t10 = t7 @ self.nonlinearLayer0_1
        return t10
"""


# module which applies nonlinearity
class ModuleForwardNonlinear(torch.nn.Module):
    def __init__(self, inDim, hiddenDims, outDim):
        super(ModuleForwardNonlinear, self).__init__()

        # commented because non-modular
        """
        # ordered dict describing layers
        orderedDict = [
          ('conv1', torch.nn.Linear(inDim, nHiddens[0])),
          ('relu1', nn.ReLU()),
          ('conv2', torch.nn.Linear(nHiddens[0], nHiddens[1])),
          ('relu2', nn.ReLU()),
          ('out', torch.nn.Linear(nHiddens[1], outDim))
        ]

        self.linearReluStack = torch.nn.Sequential(orderedDict)
        """


        # hidden + output
        dims = hiddenDims
        dims.append(outDim)

        self.linearReluStack = torch.nn.Sequential()
        self.linearReluStack.append(torch.nn.Linear(inDim, dims[0])) # linear layer to first hidden

        for iIdx in range(len(dims)-1):
            self.linearReluStack.append( torch.nn.ReLU() )
            self.linearReluStack.append( torch.nn.Linear(dims[iIdx], dims[iIdx+1]) )
        
        # TODO< access weights to init >
        #       self.linearReluStack[4].weight

    def forward(self, x):
        return self.linearReluStack(x)






# NN which uses Attention blocks but has a suboptimal architecture which only reaches 11% training accuracy
class Nn0(torch.nn.Module):
    # /param dk dimension of queries and keys
    def __init__(self, dk, nTokens, ctxLen, embeddingDim):
        super(Nn0, self).__init__()
        self.dk = dk
        #self.softmax = torch.nn.Softmax(dim=0)

        #self.selfAttention = SelfAttentionLayer(ctxLen, dk)
        

        self.layers = []
        
        submodulesA = []

        if True:
            #shape = (ctxLen,dk)
            createdLayer = []
            
            for z in range(3):
                z = SelfAttentionLayer(ctxLen, dk)
                submodulesA.append(z)
                createdLayer.append(z)

            self.layers.append(createdLayer)
            #del createdHead
            del createdLayer
        
        if True:
            #shape = (embeddingDim,dk)
            createdLayer = []

            z = SelfAttentionLayer(embeddingDim, dk)
            submodulesA.append(z)
            createdLayer.append(z)

            #createdLayer.append(SelfAttentionLayer(embeddingDim, dk))


            #createdHead = {}
            # weights for 'query', 'key', 'value'
            #createdHead["wq"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            #createdHead["wk"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            #createdHead["wv"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            
            #createdLayer.append(createdHead)

            self.layers.append(createdLayer)
            #del createdHead
            del createdLayer

        self.submodulesA = torch.nn.ModuleList(submodulesA) # register sub-modules


        inputDim = 3*embeddingDim + ctxLen*embeddingDim
        self.submoduleNonlinearAfterLayer0 = ModuleForwardNonlinear(inputDim, [220, 180], embeddingDim)
        del inputDim
        self.submodulesB = torch.nn.ModuleList([self.submoduleNonlinearAfterLayer0])

        self.submoduleNonlinearAfterLayer1 = ModuleForwardNonlinear(embeddingDim+1, [180, 160], embeddingDim)
        self.submodulesC = torch.nn.ModuleList([self.submoduleNonlinearAfterLayer1])


        
        #layer0NonlinearDim = 220 #170 #152 #52 # dimensionality of nonlinear layer
        
        #self.nonlinearLayer0_0 = torch.nn.Parameter(torch.rand((embeddingDim*2,layer0NonlinearDim,))*0.05)

        #print(4*dk + ctxLen*embeddingDim)
        #kofokfokfko

        #                                                       4*embeddingDim + ctxLen*embeddingDim no
        #self.nonlinearLayer0_0 = torch.nn.Parameter(torch.rand((4*embeddingDim + ctxLen*embeddingDim,layer0NonlinearDim,))*0.05)
        #self.nonlinearLayer0_0_bias = torch.nn.Parameter(torch.rand((layer0NonlinearDim))*0.005)
        #self.nonlinearLayer0_1 = torch.nn.Parameter(torch.rand((layer0NonlinearDim,embeddingDim,))*0.05)


        
        
        # weights for context vector to probability vector
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((embeddingDim,nTokens,))*0.05)
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((layer0NonlinearDim,nTokens,))*0.05)
        self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((embeddingDim,nTokens,))*0.05)
        self.contextVecToProbabilityVecBias = torch.nn.Parameter(torch.rand((nTokens))*0.005)
        
        self.inputEmbeddings = torch.nn.Embedding(nTokens, embeddingDim)
        
    def forward(self, x0):
        # LAYER #0

        """
        # BEGIN self attention layer        
        q = torch.matmul(x0, self.wq)
        k = torch.matmul(x0, self.wk)
        v = torch.matmul(x0, self.wv)
        
        t0 = torch.matmul(q, torch.transpose(k, 0, 1)) * sqrt(self.dk)
        t1 = self.softmax(t0)
        t2 = torch.matmul(t1, v)
        # END    self attention layer
        """

        # iterate over heads and add up the results
        res0 = []
        for iHead in self.layers[0]:
            t3 = iHead.forward(x0)
            t3 = t3.sum(dim=1) # compute "context vector"
            res0.append(t3)
        
        t14 = torch.cat( tuple(res0) ) # concatenate from heads

        """
        # head[0]
        t2 = self.layers[0][0].forward(x0)

        #plusSkip0 = x0 + t2

        t3 = t2.sum(dim=1) # compute "context vector"


        # head[1]
        t12 = self.layers[0][1].forward(x0)

        #plusSkip1 = x0 + t12

        t13 = t12.sum(dim=1) # compute "context vector"
        
        #print(f't3={t3}') # DBG
        
        #print(t3.shape)
        #print(t13.shape)

        t14 = torch.cat((t3, t13)) # concatenate from heads
        #t14 = torch.cat((t14, x0))
        """

        #print(x0.shape)
        #fkofokfkofko

        #print(t14.shape)

        #fofdokkodfokdofkdf



        t70 = torch.reshape(x0, (-1,)) # convert to one dimensional matrix
        t71 = torch.cat((t14, t70))
        #print(t71.shape)
        #fjiijfijfijfji

        #t80 = t14
        t80 = t71

        #print(t14.shape) # 200
        #print(t70.shape) # 1500

        #print(t80.shape)
        #fkoofkokfokf


        # apply non-linear layer
        #t9 = t80 @ self.nonlinearLayer0_0
        #t6 = torch.nn.functional.relu(t9)
        #t7 = t6 + self.nonlinearLayer0_0_bias # add bias
        #t10 = t7 @ self.nonlinearLayer0_1
        t10 = self.submoduleNonlinearAfterLayer0.forward(t80)

        
        
        # TODO< concat with ressidual and transform over another matrix >
        
        

        # convert to transformed tokens (somewhat hacky!!!)
        t11 = torch.unsqueeze(t10, 0)
        # t11 = t10 @ self.convVecToTokensLayer0

        #print(t11.shape)
        #fkokofkofko




        # LAYER #1
        res0 = []
        for iHead in self.layers[1]:
            t111 = iHead.forward(t11)
            t111 = t111.sum(dim=1) # compute "context vector"
            res0.append(t111)
        
        t140 = torch.cat( tuple(res0) ) # concatenate from heads
        t70 = torch.reshape(t11, (-1,)) # convert to one dimensional matrix
        t71 = torch.cat((t140, t70))

        #t777 = t11
        t777 = t71

        """
        t8 = self.selfAttention.forward(t11)
        t9 = t8 #t8.sum(dim=1) # compute "context vector"
        




        t50 = t9
        #print(t9.shape)
        t51 = transpose2(torch.cat((transpose2(t11),transpose2(t11))))
        #print(transpose2(torch.cat((torch.transpose(t11, 0, 1),torch.transpose(t11, 0, 1)))).shape)
        t52 = t51 + t50 # HACKY  FIXME< should be done with a matrix mul>
        # t50 = t9



        
        
        # TODO< multiply by matrix to get output of encoder layer! >
        """

        # t7771 = t777
        t7771 = self.submoduleNonlinearAfterLayer1.forward(t777)


        
        
        # OUT propbability head
        
        # multiply with matrix to get probabilities
        t4 = t7771 @ self.contextVecToProbabilityVec
        t5 = t4 + self.contextVecToProbabilityVecBias

        # HACKY
        t5 = torch.squeeze(t5)
        
        #print(t4) # DBG
        
        return t5





# experimental NN based on only ReLU
class Nn1(torch.nn.Module):
    # /param dk dimension of queries and keys
    def __init__(self, nTokens, ctxLen, embeddingDim):
        super(Nn1, self).__init__()

        inputDim = ctxLen*embeddingDim
        self.submoduleNonlinearAfterLayer0 = ModuleForwardNonlinear(inputDim, [220, 180], embeddingDim)
        del inputDim
        self.submodulesB = torch.nn.ModuleList([self.submoduleNonlinearAfterLayer0])

        
        # weights for context vector to probability vector
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((embeddingDim,nTokens,))*0.05)
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((layer0NonlinearDim,nTokens,))*0.05)
        self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((embeddingDim,nTokens,))*0.05)
        self.contextVecToProbabilityVecBias = torch.nn.Parameter(torch.rand((nTokens))*0.005)
        
        self.inputEmbeddings = torch.nn.Embedding(nTokens, embeddingDim)
        
    def forward(self, x0):
        # LAYER #0


        t0 = torch.reshape(x0, (-1,)) # convert to one dimensional matrix
        
        t1 = self.submoduleNonlinearAfterLayer0.forward(t0)

        
        
        
        # OUT propbability head
        
        # multiply with matrix to get probabilities
        t4 = t1 @ self.contextVecToProbabilityVec
        t5 = t4 + self.contextVecToProbabilityVecBias

        # HACKY
        t5 = torch.squeeze(t5)
        
        #print(t4) # DBG
        
        return t5




# experimental NN attempting to implement hyena hierachy
#
class Nn2(torch.nn.Module):
    # /param dk dimension of queries and keys
    def __init__(self, device, nTokens, ctxLen, embeddingDim):
        super(Nn2, self).__init__()
        self.device = device # remember on which device we are

        # TODO< rename to self.L > # sequence length seqLen
        self.dim = 128 # 32 # dimensionality of hyena matrices

        self.D = 6 # dimensionalty of non-linear output vector of hyena hierachy, can be any number
        self.N = 1 # order of the hyena hierachy   =   depth of non-linear calculation

        # random projection matrices
        inputDim = ctxLen*embeddingDim
        #self.proj0 = torch.nn.Parameter(torch.rand((inputDim,self.dim,))*0.05, requires_grad=False) # not learned
        #self.proj1 = torch.nn.Parameter(torch.rand((inputDim,self.dim,))*0.05, requires_grad=False) # not learned
        #self.proj2 = torch.nn.Parameter(torch.rand((inputDim,self.dim,))*0.05, requires_grad=False) # not learned

        self.projForX = [] # projection matrices for computing "x"
        for it in range(2):
            temp0 = torch.nn.Parameter(torch.rand((inputDim,1*self.D,))*0.05, requires_grad=False) # not learned
            self.projForX.append(temp0)
        
        self.projForV = torch.nn.Parameter(torch.rand((inputDim,self.dim*self.D,))*0.05, requires_grad=False) # not learned # projection matrix for computing "z"



        self.sArr = []



        # learned window parameters
        self.a = torch.nn.Parameter(torch.randn(self.dim)).to(self.device)
        self.b = torch.nn.Parameter(torch.randn(self.dim)).to(self.device)


        self.updateA()
        #print(a)

        #self.s1 = torch.tril( torch.ones((32,32,)) )

        # diagonal matrices
        #self.d1 = (torch.diag( torch.randn(self.dim) ) * 0.05) .to(self.device)
        #self.d2 = (torch.diag( torch.randn(self.dim) ) * 0.05) .to(self.device)

        self.dArr = []
        for i in range(2):
            d = (torch.diag( torch.randn(self.dim) ) * 0.05) .to(self.device)
            self.dArr.append(d)

        
        self.convYtoLowerDim = torch.nn.Parameter(torch.rand((self.dim,80,))*0.5*(1.0/(self.dim)))

        # weights for context vector to probability vector
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((embeddingDim,nTokens,))*0.05)
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((layer0NonlinearDim,nTokens,))*0.05)
        
        #self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((80,nTokens,))*0.5*(1.0/80))
        #self.contextVecToProbabilityVecBias = torch.nn.Parameter(torch.rand((nTokens))*0.005)
        
        self.contextVecToProbabilityVec = torch.nn.Parameter(torch.rand((self.D, nTokens,))*0.5*(1.0/80))
        self.contextVecToProbabilityVecBias = torch.nn.Parameter(torch.rand((nTokens))*0.005)
        

        self.inputEmbeddings = torch.nn.Embedding(nTokens, embeddingDim)
        #self.inputEmbeddings = self.inputEmbeddings.to(device) # is this necessary?
    
    # copy values after .backward()
    def updateA(self):
        
        filterWindowDecay = 0.3
        filterWindowBias = 0.0005
        # compute filter
        filter_ = torch.tensor([exp(-z*filterWindowDecay)*(1.0-filterWindowBias)+filterWindowBias for z in range(self.dim)]).to(self.device)
        #print(filter_)


        #tWindow = torch.multiply(self.a, filter_)

        #print(tWindow)


        self.sArr = []


        tWindow = torch.multiply(self.a, filter_)
        #print(tWindow.device)
        #self.s1 = makeToeplitzKernel(tWindow.to('cpu')).to(self.device)

        s = makeToeplitzKernel2(tWindow.to('cpu'), self.D).to(self.device)
        self.sArr.append(s)



        tWindow = torch.multiply(self.b, filter_)
        #self.s2 = makeToeplitzKernel(tWindow.to('cpu')).to(self.device)

        s = makeToeplitzKernel2(tWindow.to('cpu'), self.D).to(self.device)
        self.sArr.append(s)


    def forward(self, xIn):
        t0 = torch.reshape(xIn, (-1,)) # convert to one dimensional matrix
        #print(f'{t0.shape} t0')
        t0 = matConv1dTo2d(t0)
        #print(t0.shape, 't0.shape')


        # * paper algorithm "Algorithm 1: Projection"
        xArr = []
        v = None
        if True: # codeblock

            # FIXME< is hacky and not according to paper! >

            zArr = []
            for it in range(2):
                temp0 = t0 @ self.projForX[it]

                # NOTNECESSARY< it is not necessary for now to reshape matrix temp0 into matrix with right dimensions >

                zArr.append(temp0)
            
            temp0 = t0 @ self.projForV
            temp0 = torch.reshape(temp0, (self.D, self.dim,)) # reshape so it has the right shape
            #print(temp0.shape, 'temp0.shape')
            zArr.append(temp0) # NOTE< just append to array >

            # paper: reshape and split into x_1 ... x_N and z
            xArr = zArr[:-1]
            v = zArr[-1]



        # * paper algorithm "Algorithm 3: Forward pass of Hyena"
        if True: # codeblock

            #xArr = []
            #xArr.append(torch.randn((self.N, self.D, )))
            #xArr.append(torch.randn((self.N, self.D, )))

            hArr = [] # array for hyena matrices
            #h.append(torch.randn((self.D  , self.dim, ))*0.05)
            #h.append(torch.randn((self.D  , self.dim, ))*0.05)
            
            # calculate "h" hyena matrix from diagonal matrix "d"  and  Toeplitz matrix "s"
            for idx in range(len(self.sArr)):
                temp0 = self.dArr[idx] @ self.sArr[idx]
                temp0 = transpose2(temp0) # HACKY correction, this doesn't change the math
                hArr.append(temp0)

            #v = torch.randn((self.D, self.dim, ))*0.05

            # self.D is the model width
            # self.N is the order of the hyena filter

            # see "Algorithm 3" in the Hyena paper
            for n in range(1,self.N+1):
                t = []
                for i_t in range(self.D):
                    #print(f'it i_t={i_t}') # DBG

                    #print(f'{n} {i_t}') # DBG
                    
                    #print(h[n-1][i_t], 'h[]')
                    tempV = transpose2(matConv1dTo2d(v[i_t]))
                    #print(tempV, 'v')
                    #fkokofkofokfokfko # OK
                    
                    #print(hArr[n-1].shape, 'h[n-1].shape') # DBG
                    temp1 = hArr[n-1] @ tempV
                    #print(temp1, 'temp1')
                    #kfkfkfkfk # OK


                    #fkookfkofok # OK

                    tempXarrValue = xArr[n-1]
                    #print(tempXarrValue.shape, 'x[n-1].shape') # DBG
                    
                    #tempX = matConv1dTo2d(tempXarrValue[i_t])
                    tempX = tempXarrValue
                    
                    
                    #print(tempX, 'x_t_n')
                    #print(tempX.shape, 'tempX.shape') # DBG
                    #print(temp1.shape, 'temp1.shape') # DBG
                    
                    temp0 = tempX @ temp1 # dot product
                    #print(temp0, 'temp0')
                    assert2(temp0.shape == (1,1), 'result of dot product must be scalar value!')
                    t.append(temp0[0][0].item())
                
                #fkofkookodkokdokd
                v = transpose2(torch.tensor([t])).to(self.device)

            yHyenaHierachy = v

            #print(yHyenaHierachy) # DBG
            #goodA



        """
        # project input with (frozen) input projections
        v = transpose2(t0 @ self.proj0)
        x1 = transpose2(t0 @ self.proj1)

        x2 = transpose2(t0 @ self.proj2)

        #print(f'{v.shape} v')
        #print(f'{x1.shape} x1')
        #gkogogkogko

        # compute results of hyena hierachy
        z1 = v
        #z1 = transpose2(z1)

        #print(f'{x1.shape} A')



        h1 = self.s1 @ self.d1 # compute hyena matrix
        
        # ASSERTION
        #if h1.shape != (self.dim, self.dim, ):  # must be matrix!
        #    fkfkddokkodfokdf
        
        #print('+++')
        #print(f'{x1.shape} x1')
        #print(f'{h1.shape} h1')
        #print(f'{z1.shape} z1')

        #z2 = x1 @ transpose2(h1 @ z1)
        #z2 = transpose2(h1 @ z1) @ x1
        
        
        t50 = h1 @ z1
        z2 = torch.multiply(x1, t50)
        #print(f'{t50.shape} t50')

        #print('=')

        #print(f'{z2.shape} z2')

        # ASSERTION
        #if z2.shape != (self.dim, 1, ):
        #    fkfkddokkodfokdf
        




        h2 = self.s2 @ self.d2 # compute hyena matrix
        
        # ASSERTION
        #if h2.shape != (self.dim, self.dim, ):  # must be matrix!
        #    fkfkddokkodfokdf

        #print(f'{z2.shape} z2')

        #z3 = x2 @ transpose2(h2 @ z2) 
        t50 = h2 @ z2
        z3 = torch.multiply(x2, t50)
        
        # ASSERTION
        #if z3.shape != (self.dim, 1, ):
        #    fkfkddokkodfokdf



        #y = z2
        y = z3
        y = torch.reshape(y, (-1,)) # convert to one dimensional matrix
        
        """
        
        # do transform from y to y0 to reduce dimensionality and thus parameter count
        #y0 = y @ self.convYtoLowerDim

        
        y0 = transpose2(yHyenaHierachy)


        # OUT propbability head
        
        # multiply with matrix to get probabilities
        t4 = y0 @ self.contextVecToProbabilityVec
        t5 = t4 + self.contextVecToProbabilityVecBias

        # HACKY
        t5 = torch.squeeze(t5)
        
        #print(t4) # DBG
        
        return t5





def readTokens(filepath):
    f = open(filepath, 'r')
    z0 = f.read()
    f.close()
    
    z1 = z0.split(', ')
    z2 = list(map(lambda z: int(z), z1))
    return z2



if __name__ == '__main__':

    import time
    import argparse

    parser = argparse.ArgumentParser(prog='a', description='train ML model')

    parser.add_argument('filename') # positional argument
    parser.add_argument('--epochs')
    parser.add_argument('--device')
    parser.add_argument('--restore', action='store_true', help='restore from checkpoint')


    args = parser.parse_args()

    #device = 'cuda' # 'cpu'
    device = args.device


    ctxLen = 30 #24 #10 # length of the context
    nTokens = 5000 #500 # number of tokens

    embeddingDim = 88 # 50 #36 # 24 # 12 # size of the embedding vector


    #dk = 3
    dk = 170 #150 #92 #72 # 66 # 46 # 16 # 8   # dimension of self-attention matrix


    # tokens to train
    txtTokens = []

    # fill with testdata
    #for z in range(500):
    #    t0 = random.randint(0, nTokens-1)
    #    txtTokens.append(t0)

        
    txtTokens = [0] * (ctxLen-1) # fill up with zero for being able to make sense of the beginning
    #txtTokens = readTokens('./trainTokensPROTO.txt')
    #txtTokens2 = readTokens('./trainTokens0.txt')
    #txtTokens2 = readTokens('./trainTokens1.txt')
    #txtTokens2 = readTokens('./trainTokens2small.txt')
    #txtTokens2 = readTokens('./outTokens0.txt')
    txtTokens2 = readTokens(args.filename)
    txtTokens = txtTokens + txtTokens2
    #print(txtTokens) # DBG
    #r = r + 1

    #nn0 = Nn0(dk=dk, nTokens=nTokens, ctxLen=ctxLen, embeddingDim=embeddingDim)
    #nn0 = Nn1(nTokens=nTokens, ctxLen=ctxLen, embeddingDim=embeddingDim)
    nn0 = Nn2(device=device, nTokens=nTokens, ctxLen=ctxLen, embeddingDim=embeddingDim)
    if args.restore:
        nn0 = torch.load('./models/model-snapshot.pth')
    else:
        pass
    
    nn0 = nn0.to(device)
    
    
    print(list(nn0.parameters()))

    # see https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    pytorchTotalParams = sum(p.numel() for p in nn0.parameters() if p.requires_grad)
    print(f'nParams={pytorchTotalParams}')


    #tokensVal = torch.rand((nTokens, embeddingDim)) # matrix with values of tokens

    positionalEncodings = calcPositionalEncoding(embeddingDim, ctxLen)
    positionalEncodings = positionalEncodings.to(device)






    # 1st dimension is dimensionality of embedding
    # 2nd dimension is number of input tokens
    #shape = (embeddingDim, ctxLen)



    lossFn = torch.nn.MSELoss()


    #optimizer = torch.optim.SGD(nn0.parameters(), lr=lr)
    optimizer = torch.optim.Adam(nn0.parameters(), lr=0.0004*0.8) # 0.001 great results      0.005 way to fast

    #nSamples = 1 # number of samples in one epoch

    avgLoss = None

    correctPredictions = 0
    wrongPredictions = 0

    bestRatio = 0.0 # best ratio - used for deciding when to store checkpoint

    nMicrobatch = 8 # size of microbatch

    
    timeLastReport = time.time()

    for iStep in range(int(len(txtTokens)*float(args.epochs)/nMicrobatch)): # 
        
        optimizer.zero_grad() # reset gradients

        timeStart = time.time()
        
        for ibatchIdx in range(nMicrobatch):

            
            
            selStartIdx = random.randrange(2**20) % (len(txtTokens)-ctxLen)

            slice0 = txtTokens[selStartIdx:selStartIdx+ctxLen] # compute slice of tokens
            
            t0 = nn0.inputEmbeddings(torch.tensor(slice0).to(device)) # lookup embeddings
            
            #print(t0.device)

            #print(x)
            #xxxx
            
            x2 = positionalEncodings*t0
            #print(x2.shape)
            
            """ old crappy calculation
            embeddings = []
            for iIdx in range(positionalEncodings.shape[0]):
                embeddingWithPositionalEncoding = (positionalEncodings[iIdx]*t0[iIdx]) # multiply embedding with positional encoding
                embeddings.append(embeddingWithPositionalEncoding)

            x = torch.stack(embeddings, dim=0) # convert list with vectors to matrix
            
            print(x.shape)
            """
            
            x = x2
            
            x = torch.transpose(x, 0, 1)
            
            
            
            #xxxokkofkofkokfo


            #print(x)
            #print(x.shape)
            
            
            yToken = txtTokens[selStartIdx+ctxLen] # token to be predicted
            #print(yToken) # DBG
            #y = tokensVal[yToken] # embedding of token to be predicted
            #print(y) # DBG
            y = torch.zeros((nTokens))
            y[yToken] = 1.0
            
            y = y.to(device)
            x = x.to(device)
            
            pred = nn0(x)
            #print(f'{pred} <<< pred') # DBG
            
            #print(pred.shape)
            #print(y.shape)
            #gkgkkg = kgtkjgjkf
            
            topkRes = torch.topk(pred, 1)
            yTokenIdx = topkRes.indices[0].item()
            if yToken == yTokenIdx:
                correctPredictions+=1
            else:
                wrongPredictions+=1
            
            loss = lossFn(pred, y) # compute loss
            loss = loss / nMicrobatch # see https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch
            
            # HACKY because we use retain_graph !
            loss.backward(retain_graph=(ibatchIdx!=(nMicrobatch-1)))

            #nn0.copyAfterBackward()
            
        timeEnd = time.time()
        timeDelta = timeEnd - timeStart
        if False: # debug speed?
            print(f'dt={timeDelta}')

        
        optimizer.step()

        nn0.updateA()


        
        lossVal = loss.item()
        
        if avgLoss is None:
            avgLoss = lossVal
        avgLoss = avgLoss * (1.0 - 0.001) + lossVal * 0.001

        if iStep % int(2500/nMicrobatch) == 0:
            lossVal, current = loss.item(), (iStep + 1) * len(x)
            epoch = float(iStep) / len(txtTokens) # compute current epoch
            print(f"loss={lossVal:>7f}  avgLoss={avgLoss:>7f}   [epoch={epoch:>4f}]")
            
            
            currentPredRatio = correctPredictions/(correctPredictions+wrongPredictions)
            
            
            print(f'correctPredictions={correctPredictions} wrongPredictions={wrongPredictions} correctPredRatio={currentPredRatio:>4f}')
            
            
            # timing
            timeThisReport = time.time()
            print(f'dt={timeThisReport-timeLastReport}')
            timeLastReport = timeThisReport
            
            # reset counters
            correctPredictions = 0
            wrongPredictions = 0
            
            # store model together with architecture
            torch.save(nn0, './models/model-snapshot.pth')

            if currentPredRatio > bestRatio:
                bestRatio = currentPredRatio
                
                # store model together with architecture
                torch.save(nn0, './models/model-bestRatio-snapshot.pth')


    print('DONE')


    # store model together with architecture
    torch.save(nn0, './models/model.pth')

# run with
# python attentionExp0.py --device=cpu --epochs=200.0 --restore ./outTokens0.txt


# THIS IS THE LATEST VERSION

# DONE< compute in STAGINGcode   h hyena matrices from "s" and "d" >
# TODO< project correctly! >


