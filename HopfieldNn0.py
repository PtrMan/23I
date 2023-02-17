#   t000 = x * w1 # map actual input to vector which encodes the combination of attended features
#   
#   t001 = hopfield(mask(t000), w0) # apply hopfied to it
#   
#   y = sigmoid(t001, w2, w3)
#   y = sigmoid(t001+x, w2, w3)

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
        
        xSize = (15)*5 # size of stimulus X of the NN
        
        # weights to convert stimuli to  stimuli fed into hopfield NN
        self.w1 = torch.nn.Parameter( ((0.01--0.01)*torch.rand(xSize, 9*2)+(-0.01)).requires_grad_() )
        
        
        
        # weights for hopfield NN
        """w0 = torch.tensor([
            [1.00000988, 0.000121, 0.0000111, 0.00007211,   0.00003432, 0.0004532, 0.00004453, 0.000006772],
            [0.00000987, 0.000122, 1.0000112, 0.00007212,   0.00003433, 0.0004533, 0.00004454, 0.000006773],
            [0.00000986, 1.000123, 0.0000113, 0.00007213,   0.00003434, 0.0004534, 0.000044544, 0.000006774],        
        ], requires_grad = True)
        """
        nHopfieldVecs = 40 # how many different vectors does the hopfied NN have? - determines memory capacity of hopfield NN. - is independent on everything else
        w0 = ((0.1--0.1)*torch.rand(9*2, nHopfieldVecs)+(-0.1)).requires_grad_()
        
        self.w0 = torch.nn.Parameter(w0)
        
        
        nUnitsOutput = 3300 # how many output units does the bottom layer have?
        
        self.w2 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(9*2 + xSize, nUnitsOutput)+(-1.0)).requires_grad_() )
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
        mask0 = [1.0]*9 + [0.0]*9
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
        
        
        #t5 = t0
        #print(t0)
        #print(t2)
        t5 = torch.cat((t0, t2), 1)
        #print(t5)
        
        
        t3 = torch.matmul(t5, self.w2)
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
                
        sliceLen = 15+1
        
        selStartIdxRangeMin = 0
        selStartIdxRangeMax = len(selSeq)-sliceLen
        #print('len='+str(len(selSeq)))
        #print('endIdx='+str(selStartIdxRangeMax))
        
        selStartIdx = random.randint(selStartIdxRangeMin, selStartIdxRangeMax)
        
        slice_ = selSeq[selStartIdx:selStartIdx+sliceLen]
        #print(slice_)
        
        return (None, slice_[:-1], slice_[-1])
        
datGen = DatGen0()
datGen.seqs.append([1777, 46, 10, 81, 58, 2412, 63, 10, 65, 58, 10, 1777, 46, 10, 10, 2979, 46, 10, 10, 81, 58, 1288, 1308, 63, 10, 65, 58, 10, 2979, 46, 10, 10, 81, 58, 2788, 63, 10, 65, 58, 10, 2239, 392, 2595, 437, 46, 917, 691, 264, 913, 920, 2032, 46, 10, 10, 10, 981, 474, 2230, 46, 10, 68, 111, 308, 2955, 594, 2230, 46, 10, 2178, 1036, 594, 701, 1432, 46, 10, 2097, 1228, 737, 103, 295, 437, 46, 10, 417, 814, 416, 303, 110, 39, 265, 1455, 46, 10, 2371, 46, 10, 81, 58, 1288, 1590, 804, 63, 10, 65, 58, 10, 2371, 46, 10, 81, 58, 2194, 44, 1409, 1324, 46, 10, 65, 58, 10, 2709, 10, 81, 58, 2194, 44, 1409, 2070, 256, 1324, 46, 10, 65, 58, 10, 2709, 10, 81, 58, 795, 2356, 39, 1880, 44, 1409, 1153, 1207, 2070, 101, 46, 10, 65, 58, 10, 881, 290, 2643, 889, 1396, 10, 81, 58, 1606, 559, 610, 44, 2411, 46, 10, 65, 58, 10, 1779, 10, 81, 58, 32, 2597, 648, 63, 10, 65, 58, 10, 71, 1207, 562, 1637, 46, 10, 81, 58, 32, 1779, 44, 2411, 46, 10, 65, 58, 10, 1779, 10, 81, 58, 32, 881, 2941, 889, 2882, 868, 44, 1409, 346, 283, 2450, 46, 10, 65, 58, 10, 34, 1953, 434, 2890, 34, 10, 10, 1767, 562, 804, 46, 10, 2711, 647, 1177, 46, 10, 2820, 487, 2995, 39, 265, 1346, 46, 10, 70, 520, 2275, 669, 926, 46, 10, 87, 2357, 277, 119, 356, 46, 10, 2647, 610, 46, 10, 417, 575, 1333, 3008, 610, 46, 10, 2239, 3032, 46, 10, 981, 922, 1766, 3032, 46, 10, 70, 1215, 2955, 2995, 39, 265, 102, 285, 46, 10, 10, 981, 562, 437, 46, 10, 81, 58, 2412, 63, 10, 65, 58, 10, 1777, 46, 10, 1953, 2266, 46, 10, 2097, 2391, 401, 46, 10, 81, 58, 1875, 551, 2388, 63, 10, 65, 58, 10, 1100, 46, 795, 551, 562, 2391, 339, 315, 736, 477, 2266, 46, 10, 81, 58, 364, 298, 289, 2980, 63, 10, 1100, 46, 10, 10, 10, 81, 58, 2788, 63, 10, 65, 58, 10, 1302, 10, 10, 81, 58, 10, 565, 788, 257, 63, 10, 65, 58, 10, 1767, 1030, 46, 10, 10, 10, 81, 58, 10, 565, 684, 63, 10, 65, 58, 10, 2711, 562, 1177, 46, 10, 10, 10, 81, 58, 10, 73, 1342, 923, 861, 922, 472, 775, 1050, 63, 10, 65, 58, 10, 1100, 10, 10, 81, 58, 546, 1618, 1603, 330, 1408, 339, 1366, 1685, 788, 423, 46, 10, 65, 58, 546, 325, 919, 797, 46, 10, 10, 81, 58, 1186, 2328, 315, 804, 259, 797, 63, 10, 65, 58, 10, 1302, 10, 81, 58, 1186, 804, 656, 2328, 2272, 797, 63, 10, 65, 58, 10, 1302, 10, 81, 58, 364, 1455, 259, 610, 63, 10, 65, 58, 10, 1599, 46, 2079, 46, 10, 81, 58, 32, 1107, 289, 1397, 373, 63, 10, 65, 58, 10, 1767, 434, 106, 1146, 46, 10, 81, 58, 341, 110, 1649, 259, 594, 2255, 1432, 44, 920, 2509, 307, 728, 63, 10, 65, 58, 10, 1599, 46, 2079, 46, 10, 81, 58, 32, 873, 2946, 63, 10, 65, 58, 10, 80, 101, 1077, 256, 392, 564, 602, 46, 10, 81, 58, 1875, 1029, 342, 454, 352, 63, 10, 65, 58, 10, 1599, 46, 1824, 816, 282, 46, 10, 81, 58, 32, 403, 1793, 63, 10, 65, 58, 10, 1793, 747, 374, 1533, 463, 536, 271, 549, 1588, 1282, 46, 10, 81, 58, 1875, 324, 285, 98, 473, 1793, 63, 10, 65, 58, 10, 1302, 46, 10, 10, 81, 58, 1186, 2120, 487, 797, 63, 10, 65, 58, 10, 1100, 46, 10, 10, 81, 58, 478, 111, 2501, 508, 2405, 32, 1637, 63, 10, 65, 58, 10, 1100, 46, 10, 10, 438, 264, 2980, 63, 1384, 44, 750, 508, 653, 39, 1880, 33, 10, 1726, 551, 610, 63, 1384, 44, 750, 2356, 39, 1880, 33, 10, 1726, 1766, 758, 63, 1384, 44, 750, 1766, 3008, 758, 44, 2312, 2954, 2987, 39, 265, 594, 1898, 46, 10, 1726, 663, 758, 63, 1384, 44, 750, 663, 653, 39, 265, 758, 33, 10, 438, 1151, 281, 758, 63, 705, 279, 44, 750, 98, 281, 2226, 1749, 2954, 474, 1898, 46, 10, 10, 69, 2941, 2615, 290, 2968, 716, 46, 2190, 2968, 2822, 2615, 883, 501, 111, 46, 484, 957, 589, 102, 356, 116, 46, 2190, 2643, 589, 1057, 101, 825, 1788, 46, 10, 809, 708, 589, 1179, 1345, 122, 46, 10, 2593, 2298, 46, 10, 2178, 267, 704, 2764, 46, 10, 1468, 968, 559, 2764, 46, 10, 2593, 544, 506, 116, 425, 603, 964, 46, 10, 417, 1600, 920, 3033])


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


print('load model...')
modelA.load_state_dict(torch.load('./lmB-checkpoint.pytorchModel'))
print('...done')

# see https://stackoverflow.com/a/49201237/388614
pytorchTotalParams = sum(p.numel() for p in modelA.parameters() if p.requires_grad)
print(f'#params={pytorchTotalParams}')

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
#optimizer = torch.optim.SGD(modelA.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(modelA.parameters(), lr=0.001)

lossAvg = None

for it in range(30000000):
    if (it % 13000) == 0:
        print(f'store model')
        torch.save(modelA.state_dict(), './lmB-checkpoint.pytorchModel')
    
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
    printLossEvernN = 800
    
    loss = criterion(yPred, yTorch)
    
    if lossAvg is None:
        lossAvg = loss.item()
    
    a = 0.9993
    lossAvg = a*lossAvg + (1.0-a)*loss.item()
    
    if (it % printLossEvernN) == (printLossEvernN-1):
        print('it=', it, loss.item(), 'lossAvg=', lossAvg)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
