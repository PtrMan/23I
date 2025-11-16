import torch
from torch import nn
import random

class Logger2(object):
    def __init__(self, outFilepath):
        self.outFilepath = outFilepath
    
    def write(self, msg):
        print(f'{msg}', flush=True)

        f = open(self.outFilepath, 'a')
        f.write(f'{msg}\n')
        f.flush()
        f.close()


class RandomVecGenerator(object):
    def __init__(self, seed=None):
        self.rng = torch.manual_seed(seed)
    def gen(self, size):
        return torch.randn(size, generator=self.rng)






class DataGen_TaskMathAddition(object):
    def __init__(self):
        pass
    
    def genByRng(self):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        return gen(a, b)

    def gen2(self, a, b):
        res = a+b

        resStr = f'{res}'
        if res < 0 or abs(res) >= 8:
            # format result so that NN has time to 'reason' about the complicated correct solution
            resStr2 = ''
            idx=0
            for iResChar in resStr:
                resStr2+=f'____{iResChar}'
                idx+=1
            resStr=resStr2

        z = ''
        z += 'goal: math\n'
        z += f'{a}+{b}={resStr}\n'
        z += 'FIN'
        return z





logger = Logger2('log5893934857.txt')






d0 = {} # dictionary of token by key=char
id0 = 0 # id counter of token

def retLetterByTokenId(tokenId):
    for iChar, iTokenId in d0.items():
        if iTokenId == tokenId:
            return iChar
    raise Exception('tokenId was not found!')



filepathsTestset = []






filepathsTrainingset =  ['untitled5.txt', 'untitled6.txt', 'untitled7_qa.txt', 'untitled8.txt', 'untitled9.txt', 'untitled10.txt', 'untitled11.txt', 'untitled12_agent.txt', 'untitled13_math.txt', 'untitled14_nnArch.txt']
filepathsTrainingset =  ['untitled5.txt', 'untitled6.txt', 'untitled7_qa.txt', 'untitled8.txt', 'untitled9.txt', 'untitled10.txt', 'untitled11.txt', 'untitled12_agent.txt', 'untitled13_math.txt', 'untitled14_nnArch.txt', 'untitled17_coding.txt']





filepathsTrainingset = []
filepathsTrainingset = filepathsTrainingset + ['untitled5.txt', 'untitled6.txt', 'untitled13_math.txt'] # trainingset with math
# task: natural language to narsese
filepathsTrainingset = filepathsTrainingset + ['untitled16_toNarseseGoalB.txt', 'untitled17_toNarseseGoalC.txt'] # previous version had trouble with it (!!!?)
filepathsTestset = ['./dataset_test/test0_math.txt', './dataset_test/test1_math.txt', './dataset_test/test2_math.txt']


filepathsTrainingset = ["/notebooks/trainingdata_text/languageSimpleA/shakespearMidsummer.txt"]

filepathsTrainingset = ["/zfsPoolF/TYPE_mlDatasets/txtC/pln errata -- wiki opencog.txt"]


# for testing: simple task which is easy to solve for FFNN alone
#filepathsTestset = []
#filepathsTrainingset = []
#filepathsTrainingset = filepathsTrainingset + ['untitled5.txt', 'untitled6.txt']



#filepathsTrainingset = ['trainingdataDelayTestA.txt'] # test to test if the fast-NN can react differently to the same stimuli,   should train to loss = 0.0

#filepathsTrainingset = []

tokensOfTrainingFiles = []

for iFilename in filepathsTrainingset:
    f = open(iFilename, 'r') # small
    z0 = f.read()
    f.close()

    for z1 in z0:
        if z1 not in d0:
            d0[z1] = id0
            id0+=1


# generate synthetic training samples
if True:
    syntheticDataGenerator = DataGen_TaskMathAddition()

    for ia in range(0, 9+1):
        for ib in range(0, 9+1):

            trainingStr = syntheticDataGenerator.gen2(ia, ib) # generate synthetic training sample

            print(trainingStr)

            for z1 in trainingStr:
                if z1 not in d0:
                    d0[z1] = id0
                    id0+=1
            
            
            tokens = []
            for z1 in trainingStr:
                tokens.append(d0[z1])

            tokensOfTrainingFiles.append(tokens)




#print(d0) # DBG


# tokenize single files
for iFilename in filepathsTrainingset:
    f = open(iFilename, 'r') # small
    z0 = f.read()
    f.close()

    tokens = []
    for z1 in z0:
        tokens.append(d0[z1])

    tokensOfTrainingFiles.append(tokens)

if False:
    print(tokensOfTrainingFiles)





tokensOfTestFiles = []

if False:
    # tokenize single files
    for iFilename in filepathsTestset:
        f = open(iFilename, 'r') # small
        z0 = f.read()
        f.close()

        tokens = []
        for z1 in z0:
            tokens.append(d0[z1])

        tokensOfTestFiles.append(tokens)




rngVecGen = RandomVecGenerator(seed=43)

vectorsByInputToken = []
for z0 in range(id0):
    z1 = rngVecGen.gen(22)
    vectorsByInputToken.append(z1)


















import math


# experimental implementation of fast weight programmer by Schmidhuber
#
# https://openreview.net/pdf?id=HJ8W1Q-0Z fast weight programmer 2018
# https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf fast weight programmer 1990
# 
#
#  x                                                         x
#  |                                                         |
#  V                 weight update for fast network          V
# RNN  -h>   NN     -wuF>                                 Fast-NN
#                                                            |
#                                                            V
#                                                            y
#
# result: works for extremely simple task


def clampGradients(grad):
    # 0.2 works in the small test with two training files
    clampVal = 0.07 # 0.2 # 7.0 # 0.1 # 0.6 #0.1
    return torch.clamp(grad, min=-clampVal, max=clampVal)


class FastNn(object):
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.fc1_weight = torch.randn(inputSize, hiddenSize)#, requires_grad=True)
        self.fc1_weight = self.fc1_weight.cuda()
        self.fc1_bias = torch.zeros(hiddenSize, requires_grad=True)
        self.fc1_bias = self.fc1_bias.cuda()

        self.fc2_weight = torch.randn(hiddenSize, outputSize, requires_grad=True)
        self.fc2_weight = self.fc2_weight.cuda()
        self.fc2_bias = torch.zeros(outputSize, requires_grad=True)
        self.fc2_bias = self.fc2_bias.cuda()

        # init weights for restoring
        self.fc1_weightInitial = self.fc1_weight.clone()
        self.fc1_weightInitial = self.fc1_weightInitial.cuda()
    
    def forward(self, x):
        x = x @ self.fc1_weight + self.fc1_bias
        x = torch.nn.functional.selu(x) # was relu(x)
        x = x @ self.fc2_weight + self.fc2_bias
        return x

    def reset(self):

        #self.ffnnFastNnWeightMatrixInitial = self.ffnnFastNnWeightMatrixInitial.detach()
        #self.ffnnFastNnWeightMatrixInitial.requires_grad_()
        self.fc1_weightInitial = self.fc1_weightInitial.detach()
        self.fc1_weightInitial.requires_grad_()

        #self.ffnnFastNnBias = self.ffnnFastNnBias.detach()
        #self.ffnnFastNnBias.requires_grad_()
        self.fc1_bias = self.fc1_bias.detach()
        self.fc1_bias.requires_grad_()

        self.fc2_weight = self.fc2_weight.detach()
        self.fc2_weight.requires_grad_()
        self.fc2_bias = self.fc2_bias.detach()
        self.fc2_bias.requires_grad_()



        ####self.ffnnFastNnWeightMatrix = self.ffnnFastNnWeightMatrixInitial
    
    def resetInternalState(self):
        self.fc1_weight = self.fc1_weightInitial
    
    def learn(self, learningRate):
        self.fc1_weightInitial = self.fc1_weightInitial - clampGradients(self.fc1_weightInitial.grad)*learningRate
        self.fc1_bias = self.fc1_bias - clampGradients(self.fc1_bias.grad)*learningRate

        self.fc2_weight = self.fc2_weight - clampGradients(self.fc2_weight.grad)*learningRate
        self.fc2_bias = self.fc2_bias - clampGradients(self.fc2_bias.grad)*learningRate
    
    def updateWeights(self, deltaVector):
        # debug
        #print(f'weight update 2 L2norm={torch.norm(deltaVector, p=2).item()}') # DBG

        # split delta vector of weight update into parts which are added to the weight matrices
        lenA = self.fc1_weight.numel()
        deltaVectorA = deltaVector[:lenA]
        #deltaVectorB = deltaVector[lenA:]

        deltaAMatrix = deltaVectorA.reshape(self.fc1_weight.shape)
        #deltaBMatrix = deltaVectorB.reshape(self.fc2_weight.shape)

        if False: # DBG
            print('weight update A=')
            print(deltaAMatrix)
            #print('weight update B=')
            #print(deltaBMatrix)

        # compute actual update of weights
        self.fc1_weight = self.fc1_weight + deltaAMatrix
        #self.fc2_weight = self.fc2_weight + deltaBMatrix



class Z(object):
    def __init__(self):
        
        self.inputSize = 22*4
        self.outputSize = 0


        # for small unittest
        self.rnnHiddenstateSize = 5
        self.fastNnHiddensize = 15

        # attempt at small dataset
        self.rnnHiddenstateSize = 15
        self.fastNnHiddensize = 30

        # trying somewhat bigger for numeric problem
        self.rnnHiddenstateSize = 23
        self.fastNnHiddensize = 30

        # trying somewhat bigger for "toNarseseGoal"
        self.rnnHiddenstateSize = 80
        self.fastNnHiddensize = 60

        # small big NN
        self.rnnHiddenstateSize = 160
        self.fastNnHiddensize = 200
    
    # builds the NN from the sizes etc.
    def buildNn(self):

        # fast-NN
        self.fastNn = FastNn(self.inputSize+self.rnnHiddenstateSize, self.fastNnHiddensize, self.outputSize)
        
        #print(f'FFNN parameter count = {self.ffnnWeightupdateCalcWeightMatrix.numel()+self.ffnnWeightupdateCalcBias.numel()}')
        #print(f'fast NN parameter count = {self.ffnnFastNnWeightMatrixInitial.numel()+self.ffnnFastNnBias.numel()}')
        


        # RNN

        self.rnnWeightMatrix = torch.randn(self.rnnHiddenstateSize, self.rnnHiddenstateSize+self.inputSize) * 0.02 # not learned!!!
        self.rnnWeightMatrix = self.rnnWeightMatrix.cuda()
        self.rnnBiasVector = torch.randn(self.rnnHiddenstateSize) * 0.02 # not learned!!!
        self.rnnBiasVector = self.rnnBiasVector.cuda()
        self.rnnInitialHiddenstate = torch.randn(self.rnnHiddenstateSize) * 0.0005 # not learned!!! (because the gradients vanish to zero if it would be learned)
        self.rnnInitialHiddenstate = self.rnnInitialHiddenstate.cuda()




        # weight update translation NN
        weightUpdateOutputSize = self.fastNn.fc1_weight.numel()# + self.fastNn.fc2_weight.numel() #self.inputSize*self.outputSize
        print(f'FFNN weight update output_size={weightUpdateOutputSize}')
        self.ffnnWeightupdateCalcWeightMatrix = torch.randn(weightUpdateOutputSize, self.rnnHiddenstateSize) * 0.05 
        self.ffnnWeightupdateCalcWeightMatrix = self.ffnnWeightupdateCalcWeightMatrix.cuda()
        self.ffnnWeightupdateCalcBias = torch.randn(weightUpdateOutputSize) * 0.05
        self.ffnnWeightupdateCalcBias = self.ffnnWeightupdateCalcBias.cuda()
        
        


        # trace for learning of the parameters
        self.ffnnWeightupdateCalcWeightMatrixTrace = []
        self.ffnnWeightupdateCalcBiasTrace = []
    
    # reset internal state of NN
    def resetInternalState(self):
        self.fastNn.resetInternalState()

        self.rnnHiddenstate = self.rnnInitialHiddenstate


    def reset(self):
        self.rnnWeightMatrix = self.rnnWeightMatrix.detach()
        self.rnnBiasVector = self.rnnBiasVector.detach()
        
        self.rnnInitialHiddenstate = self.rnnInitialHiddenstate.detach()
        ####self.rnnInitialHiddenstate.requires_grad_()
        
        self.ffnnWeightupdateCalcWeightMatrix = self.ffnnWeightupdateCalcWeightMatrix.detach()
        self.ffnnWeightupdateCalcWeightMatrix.requires_grad_()
        
        self.ffnnWeightupdateCalcBias = self.ffnnWeightupdateCalcBias.detach()
        self.ffnnWeightupdateCalcBias.requires_grad_()
        

        
        # flush trace
        self.ffnnWeightupdateCalcWeightMatrixTrace = []
        self.ffnnWeightupdateCalcBiasTrace = []

        
        # init network
        #self.rnnHiddenstate = self.rnnInitialHiddenstate

        self.fastNn.reset()



        # debug
        #print(self.rnnWeightMatrix)

    
    def forward(self, xTensor):

        


        #print(rnnInputTensor.shape)

        rnnInputTensor2 = torch.concat((self.rnnHiddenstate*1.0, xTensor)) # combine input with hidden state of RNN
        #print(rnnInputTensor2.shape)

        linearCombinationRnn = rnnInputTensor2 @ self.rnnWeightMatrix.T + self.rnnBiasVector # Compute the linear combination of input and weights
        tanhActivation = torch.tanh(linearCombinationRnn) # Apply the Tanh activation function


        #print(tanh_activation) # DBG level 1

        #input_tensor = torch.concat((tanh_activation, sin_activation))
        self.rnnHiddenstate = tanhActivation

        rnnHiddenstate2 = self.rnnHiddenstate
        if False:
            rnnHiddenstate2 = self.rnnHiddenstate - torch.mean(self.rnnHiddenstate) # subtract mean so that later calculations are easier


        # HACK to check if RNN has a effect
        if False:
            rnnHiddenstate2 = rnnHiddenstate2*0.0





        # add parameters to trace for learning
        self.ffnnWeightupdateCalcWeightMatrixTrace.append(self.ffnnWeightupdateCalcWeightMatrix)
        self.ffnnWeightupdateCalcBiasTrace.append(self.ffnnWeightupdateCalcBias)

        # NN to compute weight update by output of RNN

        linearCombinationWeightupdateCalc = rnnHiddenstate2 @ self.ffnnWeightupdateCalcWeightMatrix.T + self.ffnnWeightupdateCalcBias
        #updateNnActivation = torch.tanh(linearCombinationWeightupdateCalc) # nonlinear version
        updateNnActivation = linearCombinationWeightupdateCalc # linear version
        #print(updateNnActivation) # DBG







        # fastNN
        fastNnActivation = self.fastNn.forward(torch.concat((xTensor,rnnHiddenstate2)))


        # * (4) update weight
        #       we update weight after usage in fast-NN because that way we force the slow (R)NN to not memorize
        
        ####self.ffnnFastNnWeightMatrix = self.ffnnFastNnWeightMatrix + updateNnActivation.reshape(self.ffnnFastNnWeightMatrix.shape)
        if True:
            self.fastNn.updateWeights(updateNnActivation)



        
        
        return fastNnActivation
    
    def learn(self, learningRate):
        
        
        # commented because it is not anymore learned
        ##if self.rnnInitialHiddenstate.grad is not None: # can be None
        ##    #print('update rnnInitialHiddenstate') # DBG
        ##    self.rnnInitialHiddenstate = self.rnnInitialHiddenstate - clampGradients(self.rnnInitialHiddenstate.grad)*learningRate*0.08

        for idx in range(len(self.ffnnWeightupdateCalcWeightMatrixTrace)):
            # compute learning rate of each trace item
            # we spread the learning-rate over the trace items to make it more fair
            lr2 = ( learningRate/len(self.ffnnWeightupdateCalcWeightMatrixTrace) ) * 1.0

            if self.ffnnWeightupdateCalcWeightMatrixTrace[idx].grad is not None: # check because it can be None
                #print('update weightUpdate') # DBG
                self.ffnnWeightupdateCalcWeightMatrix = self.ffnnWeightupdateCalcWeightMatrix - clampGradients(self.ffnnWeightupdateCalcWeightMatrixTrace[idx].grad)*lr2
                self.ffnnWeightupdateCalcBias = self.ffnnWeightupdateCalcBias - clampGradients(self.ffnnWeightupdateCalcBiasTrace[idx].grad)*lr2


        self.fastNn.learn(learningRate)
    

    # returns magnitude of weights for weight decay
    def calcWeightMag(self):
        ##res = torch.sum(torch.pow(self.ffnnWeightupdateCalcWeightMatrix, 2.0)) # not taken into account for "weight decay" because it should be beneficial when this NN has large values for the weights
        res = torch.sum(torch.pow(self.fastNn.fc1_weightInitial, 2.0))
        res = res + torch.sum(torch.pow(self.fastNn.fc2_weight, 2.0))
        return res

    
    def saveToDisk(self, filepath):
        # create a dictionary to store the content
        modelDataDict = {
            #'integer_value': integer_value,

            # fast FFNN
            'fastFfNn_fc1_weightInitial': self.fastNn.fc1_weightInitial,
            'fastFfNn_fc1_bias': self.fastNn.fc1_bias,
            'fastFfNn_fc2_weight': self.fastNn.fc2_weight,
            'fastFfNn_fc2_bias': self.fastNn.fc2_bias,

            # RNN
            'rnn_weightMatrix': self.rnnWeightMatrix,
            'rnn_bias': self.rnnBiasVector,
            'rnn_InitialHiddenstate': self.rnnInitialHiddenstate,

            # weight update translation FFNN
            'weightUpdateTranslator_weight': self.ffnnWeightupdateCalcWeightMatrix,
            'weightUpdateTranslator_bias': self.ffnnWeightupdateCalcBias,
        }
        torch.save(modelDataDict, filepath)


import random

model = Z()

model.outputSize = len(d0) # output size of the NN is the number of symbols

model.buildNn() # build the NN


xStimulusArr = [torch.randn(model.inputSize), torch.randn(model.inputSize), torch.randn(model.inputSize)]
xStimulusArr = [torch.randn(10)]

###yTarget = torch.randn(10)

learningRate = 0.000005
learningRate = 0.0000015
#learningRate = 0.000002 # works well enough down to error=~15.0 which isn't great

# for unittest
learningRate = 0.00015 # worked very well for unittest
#learningRate = 0.0006

learningRate = 0.00005 / len(tokensOfTrainingFiles) # worked well with training set size 10
learningRate = 0.00005 # works fine for smallish sized dataset
learningRate = 0.00009 # experimental, seems to be to fast according to failure of convergence on simple arithmetic problem
learningRate = 0.00005

#learningRate = 0.00001 # experimenting for NARSESE translation task, result: learns to a low loss


learningRate = 0.0002

errorByDataIdx = {}

import time

wallclockStart = time.time()

timeLastSaved = time.time() # time of the last saving of the model to disk

for it in range(500000):
    model.reset()
    model.resetInternalState()
    
    delta_E = torch.tensor([[0.0]])
    delta_E = delta_E.cuda()
    
    dataIdx = None
    
    if False: # simple task
        for learnIt in range(6):

            xTensor = xStimulusArr[it % len(xStimulusArr)]
            yTensor = model.forward(xTensor)

            #print('y='+str(yTensor)) # DBG

            delta_E = delta_E + (yTensor - yTarget).pow(2).sum()
    elif True:
        
        
        
        dataIdx = random.randint(0, len(tokensOfTrainingFiles)-1) # index in training data selection
        tokens = tokensOfTrainingFiles[dataIdx] # select tokens of random input file
        
        # iterate over token and feed stimulus into NN
        for iTokenIdx in range(len(tokens)-4):
            
            currentTokenArr = []
            for idx in range(4):
                currentTokenArr.append( tokens[iTokenIdx+idx] )
            predictedToken = tokens[iTokenIdx+4]
            
            xTensor0 = vectorsByInputToken[currentTokenArr[0]] * 0.1
            xTensor1 = vectorsByInputToken[currentTokenArr[1]] * 0.1
            xTensor2 = vectorsByInputToken[currentTokenArr[2]] * 0.1
            xTensor3 = vectorsByInputToken[currentTokenArr[3]] * 0.1
            
            xTensor = torch.concat((xTensor0,xTensor1,xTensor2,xTensor3))
            xTensor = xTensor.cuda()

            ###
            ###yTargetTensor = vectorsByInputToken[predictedToken] * 0.1
            

            yTensor = model.forward(xTensor)

            #print('y='+str(yTensor)) # DBG

            #delta_E = delta_E + (yTargetTensor - yTensor).pow(2).sum()

            yTargetTensor = torch.ones(model.outputSize)*1e-5
            yTargetTensor[predictedToken] = 1.0
            yTargetTensor = yTargetTensor.cuda()

            # target = torch.randint(5, (3,), dtype=torch.int64)
            target = torch.tensor([predictedToken], dtype=torch.int64).cuda()
            yTensor2 = yTensor.view(1, -1) # convert to two dimensional tensor
            lossCrossEntropy = torch.nn.functional.cross_entropy(yTensor2, target)
            delta_E = delta_E + lossCrossEntropy

            #print(lossCrossEntropy) # DBG

            #yTensor = torch.nn.functional.softmax(yTensor) # softmax to compute probabilities
            #delta_E = delta_E + (yTargetTensor - yTensor).pow(2).sum()
            ####delta_E = delta_E + (1.0 - yTensor[predictedToken])
            

        
        # * weight decay
        weightDecay = 0.01 # maybe 0.1 is good,    set to 0.0 to disable weight decay!
        delta_E = delta_E + weightDecay*model.calcWeightMag()

        

    

    errorByDataIdx[dataIdx] = delta_E.item() # keep track of error by data index
    
    if True and (it % 2) == 0:
        #print('y='+str(yTensor)) # DBG
        
        lossVal = delta_E.item()
        print(f'trainingLoss={lossVal:.6f} dataIdx={dataIdx} lr={learningRate}     wallclockTime={time.time()-wallclockStart:.1f} it={it}')
        #print(f'training {nCorrect}/{nCnt} = {nCorrect/nCnt}')

    if (it % 300) == 0:
        # debug error by dataindex

        # debug loss by training data
        print('debug error by training data (calculate start)')

        logger.write('')
        logger.write(f'it={it} lr={learningRate} wallclockTime={time.time()-wallclockStart:.1f}')
        for iDataIdx in range(len(tokensOfTrainingFiles)):
            if iDataIdx in errorByDataIdx:
                logger.write(f'training loss by data[{iDataIdx}] trainingLoss={errorByDataIdx[iDataIdx]}')
        
        # compute sum for convinience
        lossOfAllSamples = 0.0
        for iDataIdx in range(len(tokensOfTrainingFiles)):
            if iDataIdx in errorByDataIdx:
                lossOfAllSamples += errorByDataIdx[iDataIdx]
        logger.write(f'sum of training loss by data   sumTrainingLoss={lossOfAllSamples}')

        print('debug error by training data (calculate finished)')

    if (time.time() - timeLastSaved) > 3.0*60.0: # is storing of the model necessary?
        # * store model to disk
        print('store model to disk...')
        model.saveToDisk('modernFastWeightProgrammerEmath.pth')
        print('...done')

        timeLastSaved = time.time()

    #if it == 5000:
    #    learningRate *= 0.3

    
    delta_E.backward()#retain_graph=True)
    
    model.learn(learningRate)

    del delta_E


    # * test set
    if it > 0 and (it % 100) == 0  and len(tokensOfTestFiles) > 0:
        print('compute test set...')

        model.resetInternalState()

        delta_E = torch.tensor([[0.0]])
        delta_E = delta_E.cuda()

        usedTestsetTokensOfTestSamples = tokensOfTestFiles

        dataIdx = random.randint(0, len(usedTestsetTokensOfTestSamples)-1) # index in data selection
        tokens = usedTestsetTokensOfTestSamples[dataIdx] # select tokens of random input file
        
        # iterate over token and feed stimulus into NN

        textAlreadyPredicted = '' # 

        for iTokenIdx in range(len(tokens)-4):
            
            currentTokenArr = []
            for idx in range(4):
                currentTokenArr.append( tokens[iTokenIdx+idx] )
            predictedToken = tokens[iTokenIdx+4]
            
            xTensor0 = vectorsByInputToken[currentTokenArr[0]] * 0.1
            xTensor1 = vectorsByInputToken[currentTokenArr[1]] * 0.1
            xTensor2 = vectorsByInputToken[currentTokenArr[2]] * 0.1
            xTensor3 = vectorsByInputToken[currentTokenArr[3]] * 0.1
            
            xTensor = torch.concat((xTensor0,xTensor1,xTensor2,xTensor3))
            xTensor = xTensor.cuda()
            yTargetTensor = vectorsByInputToken[predictedToken] * 0.1
            yTargetTensor = yTargetTensor.cuda()
            
            yTensor = model.forward(xTensor)
            #print('y='+str(yTensor)) # DBG

            #delta_E = delta_E + (yTargetTensor - yTensor).pow(2).sum() # update loss function

            yTensor = torch.nn.functional.softmax(yTensor) # softmax to compute probabilities
            delta_E = delta_E + (1.0 - yTensor[predictedToken])

            ###
            #### find best vector which matches best
            ###if True:
            ###    bestIdx = 0
            ###    bestSim = -1.0
            ###    idx = 0
            ###    for iVec in vectorsByInputToken:
            ###        thisSim = torch.nn.CosineSimilarity(dim=-1)(yTensor, iVec)
            ###        if thisSim > bestSim:
            ###            bestSim = thisSim
            ###            bestIdx = idx
            ###        idx+=1
            ###    
            ###    simToCorrectPredictionVec = torch.nn.CosineSimilarity(dim=-1)(yTensor, yTargetTensor) # similarity to correct prediction vector
            ###    
            ###    tempVal0 = textAlreadyPredicted.replace('\n', '\\n')
            ###    logger.write(f'before={tempVal0}?    decoded predicted correctly={bestIdx==predictedToken}    simToCorrectPredVec={simToCorrectPredictionVec}')

            textAlreadyPredicted += retLetterByTokenId(predictedToken)



        
        
        print('... done')

        lossVal = delta_E.item()
        logger.write(f'testLoss={lossVal:.6f}     wallclockTime={time.time()-wallclockStart:.1f} it={it}')
        

print("info: finished")


''' traning run on smallish dataset

it=348600 lr=5e-05 wallclockTime=65202.5
training loss by data[0] trainingLoss=8.471057891845703
training loss by data[1] trainingLoss=52.87592697143555
training loss by data[2] trainingLoss=28.337223052978516
training loss by data[3] trainingLoss=65.03510284423828
training loss by data[4] trainingLoss=1212.8179931640625
training loss by data[5] trainingLoss=12.58913803100586
training loss by data[6] trainingLoss=12.150028228759766
training loss by data[7] trainingLoss=11.741315841674805
training loss by data[8] trainingLoss=70.9011459350586
training loss by data[9] trainingLoss=81.2575454711914
training loss by data[10] trainingLoss=43.926544189453125
sum of training loss by data   sumTrainingLoss=1600.103021621704
compute test set...
before=?    decoded predicted correctly=True    simToCorrectPredVec=0.9947729110717773
before=:?    decoded predicted correctly=True    simToCorrectPredVec=0.997277557849884
before=: ?    decoded predicted correctly=False    simToCorrectPredVec=0.6079626083374023
before=: m?    decoded predicted correctly=True    simToCorrectPredVec=0.8763298392295837
before=: ma?    decoded predicted correctly=True    simToCorrectPredVec=0.6707082390785217
before=: mat?    decoded predicted correctly=True    simToCorrectPredVec=0.837509036064148
before=: math?    decoded predicted correctly=True    simToCorrectPredVec=0.9251056909561157
before=: math\n?    decoded predicted correctly=False    simToCorrectPredVec=-0.20394423604011536
before=: math\n3?    decoded predicted correctly=False    simToCorrectPredVec=0.10028758645057678
before=: math\n3+?    decoded predicted correctly=False    simToCorrectPredVec=-0.16665902733802795
before=: math\n3+1?    decoded predicted correctly=False    simToCorrectPredVec=0.09670333564281464
before=: math\n3+1=?    decoded predicted correctly=False    simToCorrectPredVec=0.05758863314986229
before=: math\n3+1=4?    decoded predicted correctly=False    simToCorrectPredVec=0.040982455015182495
before=: math\n3+1=4\n?    decoded predicted correctly=False    simToCorrectPredVec=-0.33087706565856934
before=: math\n3+1=4\nF?    decoded predicted correctly=False    simToCorrectPredVec=0.30757033824920654
before=: math\n3+1=4\nFI?    decoded predicted correctly=False    simToCorrectPredVec=-0.09473289549350739
... done
testLoss=391.182800     wallclockTime=65202.6 it=348600
compute test set...
before=?    decoded predicted correctly=True    simToCorrectPredVec=0.9949053525924683
before=:?    decoded predicted correctly=True    simToCorrectPredVec=0.9974330067634583
before=: ?    decoded predicted correctly=False    simToCorrectPredVec=0.6220145225524902
before=: m?    decoded predicted correctly=True    simToCorrectPredVec=0.8784787058830261
before=: ma?    decoded predicted correctly=True    simToCorrectPredVec=0.6723710298538208
before=: mat?    decoded predicted correctly=True    simToCorrectPredVec=0.8340929746627808
before=: math?    decoded predicted correctly=True    simToCorrectPredVec=0.929621696472168
before=: math\n?    decoded predicted correctly=False    simToCorrectPredVec=0.598127007484436
before=: math\n1?    decoded predicted correctly=True    simToCorrectPredVec=0.740565299987793
before=: math\n1+?    decoded predicted correctly=True    simToCorrectPredVec=0.778911828994751
before=: math\n1+0?    decoded predicted correctly=True    simToCorrectPredVec=0.6644244194030762
before=: math\n1+0=?    decoded predicted correctly=True    simToCorrectPredVec=0.698950469493866
before=: math\n1+0=1?    decoded predicted correctly=True    simToCorrectPredVec=0.7967137098312378
before=: math\n1+0=1\n?    decoded predicted correctly=False    simToCorrectPredVec=-0.18515640497207642
before=: math\n1+0=1\nF?    decoded predicted correctly=False    simToCorrectPredVec=0.2831985056400299
before=: math\n1+0=1\nFI?    decoded predicted correctly=False    simToCorrectPredVec=0.19213715195655823
... done
testLoss=84.473572     wallclockTime=65218.6 it=348700
compute test set...
before=?    decoded predicted correctly=True    simToCorrectPredVec=0.9950557947158813
before=:?    decoded predicted correctly=True    simToCorrectPredVec=0.9973294138908386
before=: ?    decoded predicted correctly=False    simToCorrectPredVec=0.6046138405799866
before=: m?    decoded predicted correctly=True    simToCorrectPredVec=0.8816596865653992
before=: ma?    decoded predicted correctly=True    simToCorrectPredVec=0.6734896302223206
before=: mat?    decoded predicted correctly=True    simToCorrectPredVec=0.8352953791618347
before=: math?    decoded predicted correctly=True    simToCorrectPredVec=0.9393747448921204
before=: math\n?    decoded predicted correctly=False    simToCorrectPredVec=-0.18656158447265625
before=: math\n3?    decoded predicted correctly=False    simToCorrectPredVec=0.1004447340965271
before=: math\n3+?    decoded predicted correctly=False    simToCorrectPredVec=-0.05817402899265289
before=: math\n3+2?    decoded predicted correctly=False    simToCorrectPredVec=-0.00100717693567276
before=: math\n3+2=?    decoded predicted correctly=False    simToCorrectPredVec=-0.2926347851753235
before=: math\n3+2=5?    decoded predicted correctly=False    simToCorrectPredVec=0.12622752785682678
before=: math\n3+2=5\n?    decoded predicted correctly=False    simToCorrectPredVec=0.1836717426776886
before=: math\n3+2=5\nF?    decoded predicted correctly=False    simToCorrectPredVec=0.2772703766822815
before=: math\n3+2=5\nFI?    decoded predicted correctly=False    simToCorrectPredVec=-0.1946711540222168
... done
testLoss=538.478760     wallclockTime=65233.9 it=348800

(aborted, probably not fully converged)

'''





# TODO< implement code to load model from file with "loadFromDisk(self, filepath)" >

