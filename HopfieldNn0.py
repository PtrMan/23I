#   t000 = x * w1 # map actual input to vector which encodes the combination of attended features
#   
#   t001 = hopfield(mask(t000), w0) # apply hopfied to it
#   
#   y = sigmoid(t001, w2, w3)

class Model0(torch.nn.Module):
    @staticmethod
    def _hopfieldCalc(phi, xMat):
        
        phi2 = transpose2(phi)
        
        beta = 1.0
        

        t0 = beta*(transpose2(xMat)@phi2)
        t1 = torch.softmax(t0, 0) # compute softmax of hopfield NN
        phi2 = xMat@t1
        
        return transpose2(phi2)
    
    def __init__(self):

        super().__init__()
        
        self.verbosity = 0
        
        torch.manual_seed(443)
        
        xSize = (8)*5 # size of stimulus X of the NN
        
        # weights to convert stimuli to  stimuli fed into hopfield NN
        self.w1 = torch.nn.Parameter( ((0.01--0.01)*torch.rand(xSize, 8)+(-0.01)).requires_grad_() )
        
        
        
        # weights for hopfield NN
        """w0 = torch.tensor([
            [1.00000988, 0.000121, 0.0000111, 0.00007211,   0.00003432, 0.0004532, 0.00004453, 0.000006772],
            [0.00000987, 0.000122, 1.0000112, 0.00007212,   0.00003433, 0.0004533, 0.00004454, 0.000006773],
            [0.00000986, 1.000123, 0.0000113, 0.00007213,   0.00003434, 0.0004534, 0.000044544, 0.000006774],        
        ], requires_grad = True)
        """
        nHopfieldVecs = 40 # how many different vectors does the hopfied NN have? - determines memory capacity of hopfield NN. - is independent on everything else
        w0 = ((0.1--0.1)*torch.rand(8, nHopfieldVecs)+(-0.1)).requires_grad_()
        
        self.w0 = torch.nn.Parameter(w0)
        
        
        nUnitsOutput = 3300 # how many output units does the bottom layer have?
        
        self.w2 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(8, nUnitsOutput)+(-1.0)).requires_grad_() )
        self.w3 = torch.nn.Parameter( torch.rand(1, nUnitsOutput).requires_grad_() ) # bias

        
    def forward(self, x):
        
        t2 = x.unsqueeze(0) # convert to matrix
        #t2 = t2.requires_grad_()
        #print(t2)
        
        
        t1 = torch.matmul(t2, self.w1)
        #print('<')
        #print(self.w1)
        #print(t2)
        #print(self.w0)
        #print('>')
        
        # mask to mask out value from hopfield attention
        mask0 = [1.0]*4 + [0.0]*4
        mask0AsTensor = torch.tensor(mask0)
        
        #print(t1)
        #print(mask0AsTensor)
        
        t5 = torch.mul(t1, mask0AsTensor.unsqueeze(0))
        #t5 = t1
        #print('hehehhe')
        #print(t5)
        
        # input vector
        # first part is the key, second part is free
        t0 = Model0._hopfieldCalc(t5, self.w0)
        
        #print(t0)
        
        t3 = torch.matmul(t0, self.w2)
        t4 = t3
        t4 = torch.sigmoid(t3+self.w3)
        
        #print(str(t4))
        
        #jiijijji()
        
        return t4[0]


    def string(self):
        return f'<NN>'

import random

tokenEmbeddings = []
for i in range(3300):
    tokenEmbeddings.append(torch.rand(1, 5).tolist())


# data generator (used for training)
class DatGen0(object):
    def __init__(self):
        self.seqs = [] # list with all sequences
        
        pass
    
    # returns tuple of (None, list of stimuli tokens, predictedToken)
    # first result is reserved for RNN context vector
    def sample(self):
        selSeqIdx = random.randint(0, len(self.seqs)-1)
        selSeq = self.seqs[selSeqIdx]
                
        sliceLen = 8+1
        
        selStartIdxRangeMin = 0
        selStartIdxRangeMax = len(selSeq)-sliceLen
        #print('len='+str(len(selSeq)))
        #print('endIdx='+str(selStartIdxRangeMax))
        
        selStartIdx = random.randint(selStartIdxRangeMin, selStartIdxRangeMax)
        
        slice_ = selSeq[selStartIdx:selStartIdx+sliceLen]
        #print(slice_)
        
        return (None, slice_[:-1], slice_[-1])
        
datGen = DatGen0()
datGen.seqs.append([1777, 46, 10, 81, 58, 2412, 63, 10, 65, 58, 10, 1777, 46, 10, 10, 2979, 46, 10, 10, 81, 58, 1288, 1308, 63, 10, 65, 58, 10, 2979, 46, 10, 10, 81, 58, 2788, 63, 10, 65, 58, 10, 2239, 392, 2595, 437, 46, 917, 691, 264, 913, 920, 2032, 46, 10, 10, 10, 981, 474, 2230, 46, 10, 68, 111, 308, 2955, 594, 2230, 46, 10, 2178, 1036, 594, 701, 1432, 46, 10, 2097, 1228, 737, 103, 295, 437, 46, 10, 417, 814, 416, 303, 110, 39, 265, 1455, 46, 10, 2371, 46, 10, 81, 58, 1288, 1590, 804, 63, 10, 65, 58, 10, 2371, 46, 10, 81, 58, 2194, 44, 1409, 1324, 46, 10, 65, 58, 10, 2709, 10, 81, 58, 2194, 44])


# Create Tensors to hold input and outputs.
"""
trainingTuples = []
trainingTuples.append(([0, 0, 1, 2], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([0, 1, 2, 3], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([1, 2, 3, 1], [0.001, 0.9, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([2, 3, 0, 3], [0.001, 0.001, 0.9, 0.001,    0.001, 0.001]))
trainingTuples.append(([1, 2, 3, 1], [0.001, 0.9, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([2, 3, 4, 3], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
"""

# Construct our model by instantiating the class defined above
modelA = Model0()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
#optimizer = torch.optim.SGD(modelA.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(modelA.parameters(), lr=0.001)
for it in range(3000000):
    #selIdx = random.randint(0, len(trainingTuples)-1)
    
    tupleRnnCtxVec, tupleStimuliTokens, tuplePredToken = datGen.sample()

    #x = torch.tensor(trainingTuples[selIdx][0])
    
    y = [0.01]*len(tokenEmbeddings)
    y[tuplePredToken] = 0.9
    yTorch = torch.tensor(y)
    
    x2 = map(lambda v : tokenEmbeddings[v], tupleStimuliTokens) # map index of embedding to actual embedding
    
    x3 = []
    for iv in x2:
        for iv2 in iv:
            x3.extend(iv2)
    
    xTorch = torch.tensor(x3)
    
    # Forward pass: Compute predicted y by passing x to the model
    yPred = modelA(xTorch)
    
    if (it % 800) == 0:
        pass
        #print('yPred='+str(yPred))
    

    # Compute and print loss
    printLossEvernN = 200
    
    loss = criterion(yPred, yTorch)
    if (it % printLossEvernN) == (printLossEvernN-1):
        print('it=', it, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
