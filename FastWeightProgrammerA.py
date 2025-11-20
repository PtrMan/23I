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







logger = Logger2('log5893934857.txt')






d0 = {} # dictionary of token by key=char
id0 = 0 # id counter of token

def retLetterByTokenId(tokenId):
    for iChar, iTokenId in d0.items():
        if iTokenId == tokenId:
            return iChar
    raise Exception('tokenId was not found!')


"""
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


# cut training data short for testing
if True:
    tokensOfTrainingFiles = tokensOfTrainingFiles[:7000]


if False:
    print(tokensOfTrainingFiles)
"""



"""
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
"""



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
    clampVal = 0.07 # 0.2 # 0.07 # 0.2 # 7.0 # 0.1 # 0.6 #0.1
    return torch.clamp(grad, min=-clampVal, max=clampVal)


class FastNn(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        
        self.fc1_weight = torch.randn(inputSize, hiddenSize) * 0.02  # TODO : null init
        torch.nn.init.kaiming_uniform_(self.fc1_weight)
        self.fc1_weight = self.fc1_weight.cuda()
        self.fc1_bias = torch.zeros(hiddenSize, requires_grad=True)
        self.fc1_bias = self.fc1_bias.cuda()

        self.fc2_weight = torch.randn(hiddenSize, outputSize, requires_grad=True) * 0.02 # TODO : null init
        #torch.nn.init.kaiming_uniform_(self.fc2_weight)
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


# layer for "fast weight programmer"
class FwpLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inputSize = 0 # 22*4
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


        # bigger NN
        self.rnnHiddenstateSize = 190
        self.fastNnHiddensize = 250
    
    # builds the NN from the sizes etc.
    def build(self):



        # fast-NN
        self.fastNn = FastNn(self.inputSize+self.rnnHiddenstateSize, self.fastNnHiddensize, self.outputSize)
        
        #print(f'FFNN parameter count = {self.ffnnWeightupdateCalcWeightMatrix.numel()+self.ffnnWeightupdateCalcBias.numel()}')
        #print(f'fast NN parameter count = {self.ffnnFastNnWeightMatrixInitial.numel()+self.ffnnFastNnBias.numel()}')
        


        # RNN

        self.rnnWeightMatrix = torch.randn(self.rnnHiddenstateSize, self.rnnHiddenstateSize+self.inputSize) * 0.002 # 0.02 # not learned!!!
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
        

        # the RNN has a own learning rate
        learningRateOfRnn = 0.00002
        
        # commented because it is not anymore learned
        ##if self.rnnInitialHiddenstate.grad is not None: # can be None
        ##    #print('update rnnInitialHiddenstate') # DBG
        ##    self.rnnInitialHiddenstate = self.rnnInitialHiddenstate - clampGradients(self.rnnInitialHiddenstate.grad)*learningRate*0.08

        for idx in range(len(self.ffnnWeightupdateCalcWeightMatrixTrace)):
            # compute learning rate of each trace item
            # we spread the learning-rate over the trace items to make it more fair
            lr2 = ( learningRateOfRnn/len(self.ffnnWeightupdateCalcWeightMatrixTrace) ) * 1.0

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

    """ commented because not fully translated to generic code
    
    TODO : use functionality of pytorch for storing+saving of a module!
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
    """


    
    
    

class SimpleLinearLayer(torch.nn.Module):
    """
    A simple linear layer that applies a linear transformation to the incoming data.
    This is equivalent to torch.nn.Linear.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the layer's parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight tensor and wrap it in nn.Parameter to make it a trainable parameter. [1, 2]
        weight_tensor = torch.empty(out_features, in_features).cuda()
        self.weight = torch.nn.Parameter(weight_tensor)

        # Initialize bias tensor and wrap it in nn.Parameter.
        bias_tensor = torch.empty(out_features).cuda()
        self.bias = torch.nn.Parameter(bias_tensor)

        # Apply standard initialization to the weights and biases.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes or resets the layer's weights and biases.
        This initialization is the default used by PyTorch's nn.Linear layer. [3, 4]
        """
        # Kaiming uniform initialization is a common default for linear layers. [5]
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Calculate the fan-in (number of input features) to determine the bounds for bias initialization.
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)



class LogitHeadA(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def init(self, sizeIn: int, sizeOut: int):        
        # Create tensors for weight and bias
        self.m = torch.empty(sizeIn, sizeOut).cuda()
        self.bias = torch.empty(sizeOut).cuda()


        # initialize with very small values for the output head. so that logits are evenly distributed at first
        torch.nn.init.normal_(self.m, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.bias)
        
        """
        # Initialize tensors with a common strategy (Kaiming uniform for weights, uniform for bias)
        # PyTorch's default nn.Linear uses this initialization.
        torch.nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.m)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)
        """

        # Wrap the tensors in nn.Parameter to make them trainable.
        self.m = torch.nn.Parameter(self.m)
        self.bias = torch.nn.Parameter(self.bias)
    
    def forward(self, x):
        return x @ self.m + self.bias



class FwpNn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        ###self.sizeInput = 0

        self.sizeVocab = 0 # number of symbols in the vocabularity
        
        self.sizeEmbedding = 260 # dimension of the embedding

        self.sizeCrossbar = 320
        
        self.nLayers = 1

        # linear layer to translate input to crossbar
        self.inputLayer = None

        self.layers = None # ModuleList with layers
        
        self.logitHead = None
    
    def reset(self):
        
        # reset layers
        for itLayer in self.layers:
            itLayer.reset()
    

    # reset internal state of NN
    def resetInternalState(self):
        # reset internal state of layers
        for itLayer in self.layers:
            itLayer.resetInternalState()


    def learn(self, learningRate):
        # Propagate learning to layers
        for itLayer in self.layers:
            itLayer.learn(learningRate)
        
        # Update logit head parameters using gradient descent
        if self.logitHead.m.grad is not None:
            self.logitHead.m.data -= clampGradients(self.logitHead.m.grad) * learningRate
        if self.logitHead.bias.grad is not None:
            self.logitHead.bias.data -= clampGradients(self.logitHead.bias.grad) * learningRate
        
        
        
        if self.inputLayer.weight.grad is not None:
            self.inputLayer.weight.data -= clampGradients(self.inputLayer.weight.grad) * learningRate
        if self.inputLayer.bias.grad is not None:
            self.inputLayer.bias.data -= clampGradients(self.inputLayer.bias.grad) * learningRate
    
    
 
        if self.embedding.weight.grad is not None:
            # Embeddings often get sparse gradients, manual SGD works but is slow. 
            # Ensure we clamp and update.
            self.embedding.weight.data -= clampGradients(self.embedding.weight.grad) * learningRate
            
 



    def build(self):
        padding_idx = 0

        # Input Embedding
        self.embedding = nn.Embedding(self.sizeVocab, self.sizeEmbedding, padding_idx=padding_idx).cuda()



        
        self.inputLayer = SimpleLinearLayer(self.sizeEmbedding*4, self.sizeCrossbar)
        
        
        self.layers = nn.ModuleList([FwpLayer() for i in range(self.nLayers)])
        
        
        # set params of layers
        for itLayer in self.layers:
            itLayer.inputSize = self.sizeCrossbar
            itLayer.outputSize = self.sizeCrossbar
        
        # build layers
        for itLayer in self.layers:
            itLayer.build()
        
        
        # logit head
        self.logitHead = LogitHeadA()
        self.logitHead.init(self.sizeCrossbar, self.sizeVocab)

    ###def forward(self, x):    
    def forward(self, input_ids: torch.Tensor):
        
        # 1. Embed Input Tokens
        x = self.embedding(input_ids) # (batch, seq_len, d_model)
        x = torch.flatten(x)
        
        crossbar = self.inputLayer.forward(x)
        
        for itLayer in self.layers:

            nnOut = itLayer.forward(crossbar)
            crossbar = crossbar + nnOut # skip connection
        
        
        logitheadOut = self.logitHead.forward(crossbar)

        return logitheadOut

             
    def saveModel(self, file_path, epoch, optimizer):#, loss):
        # The state_dict contains all the learnable parameters of the model. [1, 2]
        state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss,
        }
        torch.save(state, file_path)
        #print(f"Model saved to {file_path}")

             
    def loadModel(self, file_path, optimizer=None):
        """
        Loads the model's state_dict from a file.
        """
        # The map_location argument is used to load a model to a specific device (e.g., 'cpu' or 'cuda'). [3]
        checkpoint = torch.load(file_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # load_state_dict() copies the parameters and buffers from the state_dict into the model. [4]
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        #loss = checkpoint.get('loss', float('inf'))
        
        #print(f"Model loaded from {file_path} at epoch {epoch} with loss {loss}")
        return epoch#, loss










#########################
# dataset utilities

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer # Use AutoTokenizer for flexibility



# --- Dataset Definition ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
             raise ValueError("Tokenizer must have a pad_token_id.")

        self.examples = []
        print("Tokenizing and chunking data...")
        for text in texts:
            if not text: continue # Skip empty lines
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)
            # Chunk into sequences of max_length
            for i in range(0, len(tokenized_text) - 1, max_length): # -1 because target is shifted
                chunk = tokenized_text[i : i + max_length + 1] # Get one extra token for target
                if len(chunk) < 2: continue # Need at least one input and one target

                input_chunk = chunk[:-1]
                target_chunk = chunk[1:]

                # Pad input_chunk
                padding_len_input = max_length - len(input_chunk)
                padded_input = input_chunk + [self.pad_token_id] * padding_len_input

                # Pad target_chunk and set padding targets to -100
                padding_len_target = max_length - len(target_chunk)
                padded_target = target_chunk + [-100] * padding_len_target # Use -100 for ignore_index

                self.examples.append({
                    "input_ids": torch.tensor(padded_input, dtype=torch.long),
                    "target_ids": torch.tensor(padded_target, dtype=torch.long)
                })
        print(f"Created {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]







  
import random
import os
import glob

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer # Use AutoTokenizer for flexibility


if __name__ == '__main__':


    CORPUS_DIRECTORY = '/zfsPoolF/TYPE_mlDatasets/txtForPrototypingA'

    MAX_SEQ_LEN = 16
    BATCH_SIZE = 1


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # --- Tokenizer ---
    tokenizer_name = 'gpt2' #  "bert-base-uncased" # Or choose another like "gpt2", "roberta-base"
    try:
        # Ensure the directory exists
        if not os.path.exists("./tokenizer_cache"):
            os.makedirs("./tokenizer_cache")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="./tokenizer_cache")
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        print("Please ensure you have internet connection or the model is cached.")
        exit()

    VOCAB_SIZE = tokenizer.vocab_size
    # Add pad token if it doesn't exist (common for GPT-2)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a default pad token. Adding '[PAD]'.")
        # Use add_special_tokens for potential resizing needed
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        VOCAB_SIZE = len(tokenizer) # Update vocab size
        print(f"New vocab size: {VOCAB_SIZE}")

    print(f"Tokenizer loaded: {tokenizer_name}, Vocab size: {VOCAB_SIZE}")
    PAD_TOKEN_ID = tokenizer.pad_token_id
    if PAD_TOKEN_ID is None:
        raise ValueError("Tokenizer failed to set a pad_token_id")
    print(f"Pad token: '{tokenizer.pad_token}', ID: {PAD_TOKEN_ID}")









    # --- Load Data from Directory ---
    all_texts = []
    if os.path.isdir(CORPUS_DIRECTORY):
        print(f"Attempting to load text files from directory: {CORPUS_DIRECTORY}")
        text_files = glob.glob(os.path.join(CORPUS_DIRECTORY, '*.txt'))

        if not text_files:
            print(f"Warning: No '.txt' files found in {CORPUS_DIRECTORY}.")
        else:
            print(f"Found {len(text_files)} '.txt' files.")
            for file_path in text_files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                        lines = f.readlines()
                        
                        #lines = [line.strip() for line in lines if line.strip() and len(line) > 10] # Basic filter
                        #all_texts.extend(lines)
                        
                        # fix
                        complete = '\n'.join(lines)
                        all_texts.append(complete)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}. Skipping.")

            if all_texts:
                print(f"Loaded a total of {len(all_texts)} lines from {len(text_files)} files.")
            else:
                 print(f"Warning: Found {len(text_files)} files, but couldn't load any valid text lines.")
    else:
        print(f"Directory '{CORPUS_DIRECTORY}' not found. No data loaded.")


    # make text short
    # FOR PROTOTYPING!!!
    all_texts = all_texts[:5000]

    # --- Dataset and DataLoader ---
    train_dataset = TextDataset(all_texts, tokenizer, max_length=MAX_SEQ_LEN)

    if len(train_dataset) == 0:
        print("ERROR: Created dataset is empty. Cannot train. Check data sources and tokenization.")
        exit()

    print(f"Successfully created dataset with {len(train_dataset)} items.")
    # Use pin_memory=True if using GPU for potentially faster data transfer
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True if DEVICE == torch.device("cuda") else False)
    print("DataLoader created.")



    model = FwpNn()

    ####model.outputSize = len(d0) # output size of the NN is the number of symbols

    ####model.sizeInput = 4*22

    model.nLayers = 2

    model.sizeVocab = VOCAB_SIZE

    model.build() # build the NN


    #xStimulusArr = [torch.randn(10)]

    ###yTarget = torch.randn(10)

    learningRate = 0.000005
    learningRate = 0.0000015
    #learningRate = 0.000002 # works well enough down to error=~15.0 which isn't great

    # for unittest
    learningRate = 0.00015 # worked very well for unittest
    #learningRate = 0.0006

    
    #learningRate = 0.00005 / len(tokensOfTrainingFiles) # worked well with training set size 10
    learningRate = 0.00005 # works fine for smallish sized dataset
    learningRate = 0.00009 # experimental, seems to be to fast according to failure of convergence on simple arithmetic problem
    learningRate = 0.00005

    #learningRate = 0.00001 # experimenting for NARSESE translation task, result: learns to a low loss


    learningRate = 0.0002


    # for PROTOTYPING!!!
    learningRate = 0.00005

    errorByDataIdx = {}

    import time

    wallclockStart = time.time()

    timeLastSaved = time.time() # time of the last saving of the model to disk


    runningLossAvg = None

    for it in range(500000):
        
        
        dataIdx = None
        
        

        
        for i, batch in enumerate(train_dataloader):
            
            model.reset()
            model.resetInternalState()
            
            delta_E = torch.tensor([[0.0]])
            delta_E = delta_E.cuda()
            
            
            
            
            input_ids = batch["input_ids"].to(DEVICE)   # Shape: [batch_size, seq_len]
            target_ids = batch["target_ids"].to(DEVICE) # Shape: [batch_size, seq_len]
            
            cntDat = 0
            
            
            ###dataIdx = random.randint(0, len(tokensOfTrainingFiles)-1) # index in training data selection
            ###tokens = tokensOfTrainingFiles[dataIdx] # select tokens of random input file
            
            tokens = input_ids[0].tolist()

            #print('tokens to train:')
            #print(tokens)

            # iterate over token and feed stimulus into NN
            for iTokenIdx in range(len(tokens)-4):
                
                cntDat += 1

                
                currentTokenArr = []
                for idx in range(4):
                    currentTokenArr.append( tokens[iTokenIdx+idx] )
                
                predictedToken = tokens[iTokenIdx+4]
                
                """
                xTensor0 = vectorsByInputToken[currentTokenArr[0]] * 0.1
                xTensor1 = vectorsByInputToken[currentTokenArr[1]] * 0.1
                xTensor2 = vectorsByInputToken[currentTokenArr[2]] * 0.1
                xTensor3 = vectorsByInputToken[currentTokenArr[3]] * 0.1
                
                xTensor = torch.concat((xTensor0,xTensor1,xTensor2,xTensor3))
                xTensor = xTensor.cuda()
                """

                ###
                ###yTargetTensor = vectorsByInputToken[predictedToken] * 0.1
                
                inputIds = currentTokenArr
                inputIds = torch.tensor(inputIds, dtype=torch.int64).cuda()

                #yTensor = model.forward(xTensor)
                yTensor = model.forward(inputIds)

                #print('y='+str(yTensor)) # DBG

                #delta_E = delta_E + (yTargetTensor - yTensor).pow(2).sum()

                #yTargetTensor = torch.ones(model.outputSize)*1e-5
                #yTargetTensor[predictedToken] = 1.0
                #yTargetTensor = yTargetTensor.cuda()

                # target = torch.randint(5, (3,), dtype=torch.int64)
                target = torch.tensor([predictedToken], dtype=torch.int64).cuda()
                yTensor2 = yTensor.view(1, -1) # convert to two dimensional tensor

                # IMPORTANT: use log softmax
                yTensor3 = torch.nn.functional.log_softmax(yTensor2)
                #yTensor3 = torch.nn.functional.softmax(yTensor2)
                #yTensor3 = yTensor2 # optimizes

                lossCrossEntropy = torch.nn.functional.cross_entropy(yTensor3, target)
                delta_E = delta_E + lossCrossEntropy
                print(lossCrossEntropy)


                yTensor4 = torch.nn.functional.softmax(yTensor2)
                #yTensor4 = yTensor3
                lossCrossEntropy2 = torch.nn.functional.cross_entropy(yTensor4, target)
                #print(lossCrossEntropy2)

                #print(lossCrossEntropy)
                

                #print('loss of prediction (cross entropy)=') # DBG
                #print(lossCrossEntropy) # DBG

                #yTensor = torch.nn.functional.softmax(yTensor) # softmax to compute probabilities
                #delta_E = delta_E + (yTargetTensor - yTensor).pow(2).sum()
                ####delta_E = delta_E + (1.0 - yTensor[predictedToken])
                

            
            # * weight decay
            enableWeightDecay = False

            if enableWeightDecay:
                weightDecay = 0.0001 # maybe 0.1 is good,    set to 0.0 to disable weight decay!
                delta_E = delta_E + weightDecay*model.calcWeightMag()

            

            ####errorByDataIdx[dataIdx] = delta_E.item() # keep track of error by data index
            

            if runningLossAvg is None:
                runningLossAvg = (delta_E.item() / cntDat)

            lossAvgFactor = 0.002
            runningLossAvg = runningLossAvg * (1.0-lossAvgFactor) + (delta_E.item() / cntDat)*lossAvgFactor


            if True and (it % 1) == 0:
                #print('y='+str(yTensor)) # DBG
                
                lossVal = delta_E.item() / cntDat
                print('')
                print(f'trainingLoss={lossVal:.6f} dataIdx={dataIdx} lr={learningRate}     wallclockTime={time.time()-wallclockStart:.1f} it={it}')
                print(f'avgTrainingLoss={runningLossAvg:.6f}')

                #print(f'training {nCorrect}/{nCnt} = {nCorrect/nCnt}')

            """
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
            """
            

            #if it == 5000:
            #    learningRate *= 0.3

            
            delta_E.backward()#retain_graph=True)
            
            model.learn(learningRate)

            del delta_E
        




        if (time.time() - timeLastSaved) > 3.0*60.0: # is storing of the model necessary?
            # * store model to disk
            print('store model to disk...')
            ####model.saveToDisk('modernFastWeightProgrammerEmath.pth')
            model.saveModel('modernFastWeightProgrammerEmath.pth', 0, None)
            print('...done')

            timeLastSaved = time.time()

        """
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
        """
    

    print("info: finished")






# TODO TODO TODO : implement the embedding layer, give the model only the ids of the tokens!
# TODO : let the model predict the symbol from the alphabet of the tokenizer

