import torch
import numpy
import random

import sys

# import abstraction to simpilfy code
from mlLib import *





lr = 0.008 # good for ReLU model
lr = 0.010 # for model with "elephant activation function"


dat = []

"""
dat.append(([-0.5242430816983074, 0.36042105935533586, -0.8906447542322077, -0.0724611659871699, 0.5448720155027098, -0.12912171386607074], [-0.8232456081655088]))
dat.append(([0.2616619771633586, 0.6676403990330362, 0.14765478724038927, 0.6114508284309184, 0.814665120940034, 0.2543553287416489], [0.47620407095002193]))
dat.append(([-0.2419995717899872, 0.9534147511950763, 0.8851720196591559, -0.12945657741718763, 0.4181288403543313, 0.36307724442476275], [0.39681637910977763]))
dat.append(([-0.34847618236601174, 0.8409266703952694, -0.6812726341566409, 0.35316378593126485, 0.12556262925526251, -0.5446107934902472], [-0.8325764629210236]))
dat.append(([-0.034658027363316135, -0.4372317816304191, 0.07006327485202957, 0.9097825241786346, -0.3298877856367677, 0.5929072236609214], [0.5014066256123626]))
dat.append(([0.23364290093707063, 0.6110331675089118, 0.41774423486448087, 0.046711295399214814, 0.10279993671122933, -0.025849764713016254], [0.66448464787774]))
dat.append(([-0.6012240478774644, 0.895639130797069, 0.9150548819056967, -0.1471230914570033, -0.11610041424450479, -0.5764219786862014], [-0.8229590504537145]))
"""


# import code which defines trainingset
from dat0 import *

class Model0(object):
    def __init__(self):
        pass
    
    def initWeights(self):

        self.xLength = 112
        #self.yLength = 16
        self.yLength = 16*2

        # 10 hidden
        #self.nNeuronsPerLayer = [32, self.yLength]
        self.nNeuronsPerLayer = [23, 23, self.yLength]



        self.x7 = xavierInitializer(self.xLength, self.nNeuronsPerLayer[0])
        # bias
        self.x8 = torch.rand(1, self.nNeuronsPerLayer[0], requires_grad=True) * 0.05



        self.x60 = xavierInitializer(self.nNeuronsPerLayer[0], self.nNeuronsPerLayer[1])
        # bias
        self.x61 = torch.rand(1, self.nNeuronsPerLayer[1], requires_grad=True) * 0.05


        self.x70 = xavierInitializer(self.nNeuronsPerLayer[1], self.nNeuronsPerLayer[2])
        # bias
        self.x71 = torch.rand(1, self.nNeuronsPerLayer[2], requires_grad=True) * 0.05



# compute forward inference for the model
def nnForward(model, x):

    x9 = x
    
    # layer [0]
    #print(x9.shape)
    if x9.shape != (1, model.xLength):
       raise EXE
    
    x10 = x9 @ model.x7 #x7 @ x9
    #print(x10)
    #print(x10.shape)
    
    if x10.shape != (1, model.nNeuronsPerLayer[0]):
       raise EXA

    x11 = x10 + model.x8
    #print(x11)
    
    x12 = libActReluLeaky(x11)
    #x12 = libActElephant(x11)


    


    # layer [1]
    #print(x12) # DBG
    #print(x12.shape)
    if x12.shape != (1, model.nNeuronsPerLayer[0]):
       raise EXB

    
    x13 = x12 @ model.x60
    #print(x13)
    x14 = x13 + model.x61
    
    #x15 = libActReluLeaky(x14)
    x15 = libActElephant(x14) # not used
    #print(x15) # DBG
    
    if x15.shape != (1, model.nNeuronsPerLayer[1]):
       raise EXC
    
    #print(x15)






    if True:
        # layer [2]
        if x15.shape != (1, model.nNeuronsPerLayer[1]):
            raise EXB

        x16 = x15 @ model.x70
        x17 = x16 + model.x71
        
        x18 = libActReluLeaky(x17)
        ##x15 = libActElephant(x14) # not used
        #print(x15) # DBG
        
        if x18.shape != (1, model.nNeuronsPerLayer[2]):
            raise EXC





    modelOut = x18
    return modelOut


def storeModel(model0):
    print("store model...")

    matrices = {
        "x7": model0.x7,
        "x8": model0.x8,
        "x60": model0.x60,
        "x61": model0.x61,
        "x70": model0.x70,
        "x71": model0.x71,
    }
    torch.save(matrices, "model0.pth")




model0 = Model0()

model0.initWeights()


if sys.argv[0+1] == 'train':

    iterations = 900000 # ok results
    iterations = 1200*1000
    iterations = 2200*1000 # enough for reLU model
    iterations = 3*9200*1000 # for elephant NN
    
    for it in range(iterations):
        #for z in range(3):
        #    print("")

        # clear out the gradients of Variables 
        model0.x7 = model0.x7.detach()
        model0.x7.requires_grad = True
        model0.x8 = model0.x8.detach()
        model0.x8.requires_grad = True
        
        model0.x60 = model0.x60.detach()
        model0.x60.requires_grad = True
        model0.x61 = model0.x61.detach()
        model0.x61.requires_grad = True
        
        model0.x70 = model0.x70.detach()
        model0.x70.requires_grad = True
        model0.x71 = model0.x71.detach()
        model0.x71.requires_grad = True
        
        

        # training data
        selIdx = random.randint(0, len(dat) - 1)
        selDat = dat[selIdx]
        trainingX = torch.tensor([selDat[0]]) # take "x" of training data
        trainingY = torch.tensor([selDat[1]])


        """    
        x9 = trainingX



        # layer [0]
        #print(x9.shape)
        if x9.shape != (1, model0.xLength):
        raise EXE
        
        x10 = x9 @ model0.x7 #x7 @ x9
        #print(x10)
        #print(x10.shape)
        
        if x10.shape != (1, model0.nNeuronsPerLayer[0]):
        raise EXA

        x11 = x10 + model0.x8
        #print(x11)
        #flplpflfplpf
        
        x12 = libActReluLeaky(x11)



        


        # layer [1]
        #print(x12) # DBG
        #print(x12.shape)
        if x12.shape != (1, model0.nNeuronsPerLayer[0]):
        raise EXB

        #
        x13 = x12 @ model0.x60
        #print(x13)
        x14 = x13 + model0.x61
        #
        x15 = libActReluLeaky(x14)
        #print(x15) # DBG
        
        if x15.shape != (1, model0.nNeuronsPerLayer[1]):
        raise EXC
        
        #print(x15)




        #x13 = x13.sum()
        modelOut = x15
        """

        modelOut = nnForward(model0, trainingX)
        
        # compute loss and do backprop
        loss = torch.nn.MSELoss()(trainingY, modelOut)
        if (it % 11000) == int(11000/2):
            print('loss='+str(loss))
        
        if (it % (11000*20)) == int((11000*20) /2):
            print('store checkpoint')
            storeModel(model0)
        
        loss.backward()


        #print("x7 grad" + str(x7.grad.data)) # DBG
        #print(x8.grad.data) # DBG
        
        
        model0.x7 = model0.x7 - model0.x7.grad.data*lr
        model0.x8 = model0.x8 - model0.x8.grad.data*lr
        model0.x60 = model0.x60 - model0.x60.grad.data*lr
        model0.x61 = model0.x61 - model0.x61.grad.data*lr
        model0.x70 = model0.x70 - model0.x70.grad.data*lr
        model0.x71 = model0.x71 - model0.x71.grad.data*lr

        #EXIT()


    # * store model
    storeModel(model0)


# inference

if sys.argv[0+1] == 'inf':
    # load model
    modelPath = sys.argv[1+1]

    
    loaded_matrices = torch.load("model0.pth")
    model0.x7 = loaded_matrices["x7"]
    model0.x8 = loaded_matrices["x8"]
    model0.x60 = loaded_matrices["x60"]
    model0.x61 = loaded_matrices["x61"]
    model0.x70 = loaded_matrices["x70"]
    model0.x71 = loaded_matrices["x71"]

    # Access the command line arguments
    command_line_numbers = sys.argv[2+1]  # First argument after the script name

    # Split the string into a list of numbers
    numbers_list = command_line_numbers.split()

    # Convert each number from string to float and create a NumPy array
    inXArr = numpy.array([float(number) for number in numbers_list])

    # Print the array to verify
    #print(inXArr)




    # * do inference with the model
    inX = torch.tensor(numpy.array([inXArr])).float() # convert to tensor in the right format
    #print(inX) # DBG

    y = nnForward(model0, inX)

    #print(y) # DBG
    

    # * print result to console for reading with process
    y2 = y.detach() # make sure that it doesnt require grad
    string_representation = " ".join([f"{value:.6f}" for value in y2[0].numpy()])
    print(string_representation)



