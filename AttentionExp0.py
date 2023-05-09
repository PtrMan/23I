import torch
import numpy
from math import sqrt
import random


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
            shape = (ctxLen,dk)
            createdLayer = []
            z = SelfAttentionLayer(ctxLen, dk)
            submodulesA.append(z)
            createdLayer.append(z)
            z = SelfAttentionLayer(ctxLen, dk)
            submodulesA.append(z)
            createdLayer.append(z)
            """
            createdHead = {}
            # weights for 'query', 'key', 'value'
            createdHead["wq"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            createdHead["wk"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            createdHead["wv"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            
            createdLayer.append(createdHead)

            createdHead = {}
            # weights for 'query', 'key', 'value'
            createdHead["wq"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            createdHead["wk"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            createdHead["wv"] = torch.nn.Parameter(torch.rand(shape)*0.05)
            
            createdLayer.append(createdHead)
            """



            self.layers.append(createdLayer)
            #del createdHead
            del createdLayer
        
        if True:
            #shape = (embeddingDim,dk)
            createdLayer = []

            z = SelfAttentionLayer(ctxLen, dk)
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

        
        layer0NonlinearDim = 152 #52 # dimensionality of nonlinear layer
        
        #self.nonlinearLayer0_0 = torch.nn.Parameter(torch.rand((embeddingDim*2,layer0NonlinearDim,))*0.05)
        self.nonlinearLayer0_0 = torch.nn.Parameter(torch.rand((936,layer0NonlinearDim,))*0.05)
        self.nonlinearLayer0_0_bias = torch.nn.Parameter(torch.rand((layer0NonlinearDim))*0.005)
        self.nonlinearLayer0_1 = torch.nn.Parameter(torch.rand((layer0NonlinearDim,embeddingDim,))*0.05)


        
        
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


        # apply non-linear layer
        t9 = t80 @ self.nonlinearLayer0_0
        t6 = torch.nn.functional.relu(t9)
        t7 = t6 + self.nonlinearLayer0_0_bias # add bias
        t10 = t7 @ self.nonlinearLayer0_1

        
        
        # TODO< concat with ressidual and transform over another matrix >
        
        

        # convert to transformed tokens (somewhat hacky!!!)
        t11 = torch.unsqueeze(t10, 0)
        # t11 = t10 @ self.convVecToTokensLayer0

        #print(t11.shape)
        #fkokofkofko




        # LAYER #1
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
        
        
        # OUT propbability head
        
        # multiply with matrix to get probabilities
        #t4 = torch.matmul(t52, self.contextVecToProbabilityVec)
        t4 = torch.matmul(t11, self.contextVecToProbabilityVec)
        t5 = t4 + self.contextVecToProbabilityVecBias

        # HACKY
        t5 = torch.squeeze(t5)
        
        #print(t4) # DBG
        
        return t5


ctxLen = 24 #10 # length of the context
nTokens = 500 # number of tokens

embeddingDim = 36 # 24 # 12 # size of the embedding vector


#dk = 3
dk = 92 #72 # 66 # 46 # 16 # 8   # dimension of self-attention matrix


# tokens to train
txtTokens = []
#txtTokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 7, 4]

# fill with testdata
#for z in range(500):
#    t0 = random.randint(0, nTokens-1)
#    txtTokens.append(t0)


def readTokens(filepath):
    f = open(filepath, 'r')
    z0 = f.read()
    f.close()
    
    z1 = z0.split(', ')
    z2 = list(map(lambda z: int(z), z1))
    return z2
    
txtTokens = [0] * (ctxLen-1) # fill up with zero for being able to make sense of the beginning
#txtTokens = readTokens('./trainTokensPROTO.txt')
#txtTokens2 = readTokens('./trainTokens0.txt')
#txtTokens2 = readTokens('./trainTokens1.txt')
txtTokens2 = readTokens('./trainTokens2small.txt')
txtTokens = txtTokens + txtTokens2
#print(txtTokens) # DBG
#r = r + 1


nn0 = Nn0(dk=dk, nTokens=nTokens, ctxLen=ctxLen, embeddingDim=embeddingDim)


print(list(nn0.parameters()))

# see https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
pytorchTotalParams = sum(p.numel() for p in nn0.parameters() if p.requires_grad)
print(f'nParams={pytorchTotalParams}')


#tokensVal = torch.rand((nTokens, embeddingDim)) # matrix with values of tokens

positionalEncodings = calcPositionalEncoding(embeddingDim, ctxLen)






# 1st dimension is dimensionality of embedding
# 2nd dimension is number of input tokens
shape = (embeddingDim, ctxLen)



lossFn = torch.nn.MSELoss()


#optimizer = torch.optim.SGD(nn0.parameters(), lr=lr)
optimizer = torch.optim.Adam(nn0.parameters(), lr=0.0004*0.8) # 0.001 great results      0.005 way to fast

#nSamples = 1 # number of samples in one epoch

avgLoss = None

correctPredictions = 0
wrongPredictions = 0

bestRatio = 0.0 # best ratio - used for deciding when to store checkpoint

for iStep in range(int(len(txtTokens)*20000.0)): # 
    selStartIdx = random.randrange(2**20) % (len(txtTokens)-ctxLen)

    slice0 = txtTokens[selStartIdx:selStartIdx+ctxLen] # compute slice of tokens
    
    if True:
        t0 = nn0.inputEmbeddings(torch.tensor(slice0)) # lookup embeddings
    else: # old embedding crap
        t0 = list(map(lambda z : tokensVal[z], slice0)) # look up embeddings by tokens

    #print(x)
    #xxxx
        
    embeddings = []
    for iIdx in range(positionalEncodings.shape[0]):
        embeddingWithPositionalEncoding = positionalEncodings[iIdx]*t0[iIdx] # multiply embedding with positional encoding
        embeddings.append(embeddingWithPositionalEncoding)

    x = torch.stack(embeddings, dim=0) # convert list with vectors to matrix
    x = torch.transpose(x, 0, 1)
        


    #print(x)
    #print(x.shape)
    
    
    yToken = txtTokens[selStartIdx+ctxLen] # token to be predicted
    #print(yToken) # DBG
    #y = tokensVal[yToken] # embedding of token to be predicted
    #print(y) # DBG
    y = torch.zeros((nTokens))
    y[yToken] = 1.0
    
    
    
    
    
    
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

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    lossVal = loss.item()
    
    if avgLoss is None:
        avgLoss = lossVal
    avgLoss = avgLoss * (1.0 - 0.001) + lossVal * 0.001

    if iStep % 2500 == 0:
        lossVal, current = loss.item(), (iStep + 1) * len(x)
        epoch = float(iStep) / len(txtTokens) # compute current epoch
        print(f"loss={lossVal:>7f}  avgLoss={avgLoss:>7f}   [epoch={epoch:>4f}]")
        
        
        currentPredRatio = correctPredictions/(correctPredictions+wrongPredictions)
        
        
        print(f'correctPredictions={correctPredictions} wrongPredictions={wrongPredictions} correctPredRatio={currentPredRatio:>4f}')
        
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

# result: this makes it up to 98.28% after 500 epochs
