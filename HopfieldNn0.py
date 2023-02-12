#   t000 = x * w1 # map actual input to vector which encodes the combination of attended features
#   
#   t001 = hopfield(mask(t000), w0) # apply hopfied to it
#   
#   y = sigmoid(t001, w2, w3)

class Model0(torch.nn.Module):
    @staticmethod
    def _hopfieldCalc(phi, xMat):
        """
        verbosity = 0
        
        #print(str(x))
        #print(str(x[0]))
        
        
        r = []
        for iIdx in range(len(w)): # iterate over indices of vectors in w
            
            r0 = torch.dot(x[0], w[iIdx])
            r.append(r0)
        
        if verbosity >= 1:
            print(str(r0))
            print(str(r1))
        
        # compute softmax of hopfield NN
        t0 = torch.tensor(r)
        t1 = torch.softmax(t0, 0)
        t2 = t1.unsqueeze(0) # convert to matrix

        if verbosity >= 1:
            print(str(t2))
            print(str(transpose2(w)))

        # compute scaled result of hopfield NN
        t4 = t2*transpose2(w)
        
        return transpose2(t4)
        """
    
    
    
        
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
        
        # weights to convert stimuli to  stimuli fed into hopfield NN
        self.w1 = torch.nn.Parameter( ((0.01--0.01)*torch.rand(5, 8)+(-0.01)).requires_grad_() )
        
        
        
        # weights for hopfield NN
        """w0 = torch.tensor([
            [1.00000988, 0.000121, 0.0000111, 0.00007211,   0.00003432, 0.0004532, 0.00004453, 0.000006772],
            [0.00000987, 0.000122, 1.0000112, 0.00007212,   0.00003433, 0.0004533, 0.00004454, 0.000006773],
            [0.00000986, 1.000123, 0.0000113, 0.00007213,   0.00003434, 0.0004534, 0.000044544, 0.000006774],        
        ], requires_grad = True)
        """
        nHopfieldVecs = 9 # how many different vectors does the hopfied NN have? - is independent on everything else
        w0 = ((0.1--0.1)*torch.rand(8, nHopfieldVecs)+(-0.1)).requires_grad_()
        
        self.w0 = torch.nn.Parameter(w0)
        
        
        
        self.w2 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(8, 6)+(-1.0)).requires_grad_() )
        self.w3 = torch.nn.Parameter( torch.rand(1, 6).requires_grad_() ) # bias

        
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

# Create Tensors to hold input and outputs.
trainingTuples = []
trainingTuples.append(([0.1, 0.21, 1.0, 0.32, 0.777], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([0.1, 1.0, 0.01, 0.12, -0.147], [0.0333, 0.9, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([0.1, 0.7, 0.81, 0.72, -0.847], [0.001, 0.9, 0.001, 0.001,    0.001, 0.001]))


# Construct our model by instantiating the class defined above
modelA = Model0()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
#optimizer = torch.optim.SGD(modelA.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(modelA.parameters(), lr=0.001)
for it in range(2800):
    selIdx = random.randint(0, len(trainingTuples)-1)
    x = torch.tensor(trainingTuples[selIdx][0])
    y = torch.tensor(trainingTuples[selIdx][1])

    
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = modelA(x)
    #y_pred = modelB(y_pred)
    
    #print('y pred='+str(y_pred))
    

    # Compute and print loss
    printLossEvernN = 250
    
    loss = criterion(y_pred, y)
    if (it % printLossEvernN) == (printLossEvernN-1):
        print('it=', it, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
