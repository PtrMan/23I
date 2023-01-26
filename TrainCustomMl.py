# version 19.01.2023

# version history
# version 1.11.2022: initial code


import torch
import math
import random

from Db import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu" # force CPU
print(device)


# abstraction for retrival of training data
class TrainingDatSrc(object):
  def __init__(self):
    self.dat = []
  
  # called when outside code needs data
  def getDat(self):
    selIdx=random.randint(0,len(self.dat)-1)
    return self.dat[selIdx]
  
  def retCnt(self):
    return len(self.dat)



class TrainingDatSrcDb(object):
  def __init__(self):
    self.db=Db()
    self.db.open('./a.sqlite',False)
  
  def getDatAt(self, selIdx):
    dbRes=self.db.queryById(selIdx)
    #print(dbRes) #dbg
    return (torch.tensor([dbRes[0]]).to(device), torch.tensor([dbRes[1]]).to(device), dbRes[2])

  def getDat(self):
    selIdx=1+random.randint(0,self.retCnt()-1)
    return self.getDatAt(selfIdx)
  
  def retCnt(self):
    return 6880-1

class CachedTrainingDatSrcDb(object):
  def __init__(self, src):
    self.src = src
    self.dat = None

  def updateCache(self):
    self.dat = []
    for iidx in range(self.src.retCnt()):
      self.dat.append(self.src.getDatAt(1+iidx))
  
  def getDatAt(self, selIdx):
    return self.dat[selIdx]

  def getDat(self):
    selIdx=random.randint(0,self.retCnt()-1)
    return self.getDatAt(selIdx)
  
  def retCnt(self):
    return self.src.retCnt()


datSrc0 = TrainingDatSrcDb()
datSrc = CachedTrainingDatSrcDb(datSrc0)

print(f"update cache...")
datSrc.updateCache()
print(f"...done")
#datSrc.dat.append((torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),torch.tensor([[0.9, 0.1]])))
#datSrc.dat.append((torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),torch.tensor([[0.1, 0.9]])))
#datSrc.dat.append((torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),torch.tensor([[0.1, 0.9]])))
#datSrc.dat.append((torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),torch.tensor([[0.9, 0.1]])))
#datSrc.dat.append((torch.tensor([[-0.2681772299,-0.2932987557,0.2887005716,-0.3606388215,0.01428060986,0.2479873095,-0.262352091,-0.3664039208,0.02515615482,-0.4138291012,-0.3968231446,0.1857696804]]),torch.tensor([[0.9, 0.1]])))

#for iA,iB in dat:
#  datSrc.dat.append((torch.tensor([iA]).to(device),torch.tensor([iB]).to(device)))


def siluFn(x):
    return x * torch.sigmoid(x)
def gcuFn(x):
    return x * torch.cos(x)

class gcu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return gcuFn(input)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential(
    torch.nn.Linear(5*29, 60),
    torch.nn.ReLU(),
    #gcu(),
    torch.nn.Linear(60, 50),
    
    gcu(), #torch.nn.ReLU(),
    torch.nn.Linear(50, 5),
    
    torch.nn.Flatten(0, 1)
)

#model.load_state_dict( torch.load("snapshot.pytorch.model", map_location=torch.device('cpu')) )

model = model.to(device)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 0.000001*6.0*2.0

avgLoss = None


for t in range(1000000000):
    x, y, gradientStrength = datSrc.getDat()

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    
    if avgLoss is None:
        avgLoss = loss
    
    avgLoss = loss*0.0001 + (avgLoss*(1.0-0.0001))
    
    debug_everyT = 900
    
    if t % debug_everyT == debug_everyT-1:
        print(f'epoch={t/datSrc.retCnt()} avgLoss={avgLoss} sample_loss={loss.item()}')
      
    

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad * gradientStrength
    

    if t % (datSrc.retCnt()*330) == 0:
        print(f'store snapshot')
        epoch = int(t/datSrc.retCnt())
        filepath = f'./epoch{epoch}.pytorch.model'
        torch.save(model.state_dict(), filepath)
    
    # stopping criteria
    if avgLoss < 0.0001:
        break
    
    del x
    del y
    del y_pred
    

print(f'store final model')
filepath = f'./final.pytorch.model'
torch.save(model.state_dict(), filepath)
