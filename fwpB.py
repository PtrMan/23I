import torch
#import random


def makeNormalTensor(size, std):
    z = torch.empty(size[0]*size[1]).normal_(mean=0.0,std=0.05)
    z = torch.reshape(z, size)
    return z


### experimental
# 
# 07/08/2025 : basic idea

# slow net as described in https://arxiv.org/pdf/2102.11174 "Linear Transformers Are Secretly Fast Weight Programmers"
class SlowNetB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inSize = 0
        
    
    def buildNn(self):
        
        hiddenSize = self.inSize # hiddensize is the in-size
        
        
        # initial value for fastWeights
        # is a network parameter which is NOT learned!
        self.fastWeightsInit = torch.nn.parameter.Parameter(torch.rand(hiddenSize, hiddenSize, requires_grad=False)*(1.0 / hiddenSize))
        
        
        self.eta = torch.nn.parameter.Parameter(torch.tensor(1.0))
        
        # FIXME LOW : better weight initialization
        # weights = torch.rand(self.inSize, hiddenSize)*0.02
        weights = makeNormalTensor((self.inSize, hiddenSize), std=1.0)
        self.wa = torch.nn.parameter.Parameter(weights)
            
        # FIXME LOW : better weight initialization
        # weights = torch.rand(self.inSize, hiddenSize)*0.02
        weights = makeNormalTensor((self.inSize, hiddenSize), std=1.0)
        self.wb = torch.nn.parameter.Parameter(weights)
        
    def reset(self):
        
        self.fastWeights = self.fastWeightsInit.detach()
    
    def forwardAndUpdate(self, x):
        
        z0 = x @ self.wa
        z1 = x @ self.wb
        
        # compute the outer products as described in the paper on "fast weight programmers"
        z100 = torch.outer(z0, z1)
        
        # scale by learnable parameter for scaling to avoid vanishing gradients
        z100 = z100 * self.eta
        
        # NOTE : should the nonlinearity be a different than SeLU?
        #z101 = torch.nn.functional.selu(self.fastWeights + z100) # works???
        z101 = torch.nn.functional.tanh(self.fastWeights + z100) # works fine for a shallow NN
        
        # algorithm: update fast weights
        self.fastWeights = z101
        
        y = x @ self.fastWeights
        
        return y
        

# nonlinear MLP which gets programmed by the FWP NN
class NonlinearLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inSize = 0
        self.hiddenSize = 0
        self.outSize = 0
        
    
    def buildNn(self):
        
        self.linearA = torch.nn.Linear(self.inSize, self.hiddenSize)
        self.linearB = torch.nn.Linear(self.hiddenSize, self.outSize)
    
    def forward(self, x):
        z0 = self.linearA.forward(x)
        z1 = torch.nn.functional.selu(z0)
        #z1 = torch.nn.functional.relu(z0) # works fine
        #z1 = torch.tanh(z0)
        z2 = self.linearB.forward(z1)
        return z2


# linear softmax output head
class LinearSoftmaxHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inSize = 0
        self.outSize = 0
    
    def buildNn(self):
        self.linear = torch.nn.Linear(self.inSize, self.outSize)
    
    def forward(self, x):
        z0 = self.linear.forward(x)
        #BUG #z1 = torch.nn.functional.softmax(z0, dim=1) # DONT DO THIS! else optimization fails and gets stuck!
        return z0
    
class FwpLmVariantC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.datapathSize = 0
        
        self.nonlinearAfterSlowHiddenSize = 20
        
        
        self.softmaxOutputHeadSize = 0

        self.xEmbeddingsNum = 0
        self.xEmbeddingsSize = 0
    
    def buildNn(self):

        

        # LAYER

        self.slowB = SlowNetB()
        self.slowB.inSize = self.datapathSize

        self.nonlinearAfterSlow = NonlinearLayer()
        self.nonlinearAfterSlow.inSize = self.slowB.inSize
        self.nonlinearAfterSlow.hiddenSize = self.nonlinearAfterSlowHiddenSize
        self.nonlinearAfterSlow.outSize = self.datapathSize

        self.slowB.buildNn()

        self.nonlinearAfterSlow.buildNn()






        
        self.logitHead = LinearSoftmaxHead()
        self.logitHead.inSize = self.datapathSize
        self.logitHead.outSize = self.softmaxOutputHeadSize
        
        self.logitHead.buildNn()


        self.xEmbeddings = torch.nn.Embedding(self.xEmbeddingsNum, self.xEmbeddingsSize)

    
    def reset(self):
        self.slowB.reset()
    
    # /param target tensor of target symbols. as batched vector
    def forwardAndUpdate(self, x, target):
        
        # feed X into "Fast Weight Programmer" layer
        z1 = self.slowB.forwardAndUpdate(x)
        

        #print(z1) # DBG

        # feed it into the nonlinear MLP so that the FWP layer can program this nonlinear layer. This is very important. See paper about what the nonlinear layer does in attention NN!
        z2 = self.nonlinearAfterSlow.forward(z1)
        #z2 = z1 # avoid nonlinear for debugging

        #print(z2) # DBG
        
        
        logits = z2
        
        # motivated by "Highway Network" paper
        # TODO LOW : use NN to compute the scaling factor as suggested in "Highway Network" paper. note to clamp scaling factor to [1.0e-5; 0.999]
        #logits2 = logits * 0.9 + x * 0.1 # was suggested by LLM.
        logits2 = logits * 0.5 + x * 0.5
        
        
        
        
        
        # linear + softmax head as NN module and call here into it
        logits = self.logitHead.forward(logits2)
        
        #print(logits) # DBG
        
        logits = logits.reshape((1, self.softmaxOutputHeadSize)) # we need to reshape because cross entropy loss expects this shape
        


        # we compute the loss in here because it is convinient
        loss = None
        if target is not None:
            loss = torch.nn.functional.cross_entropy(logits, target)

        return logits, loss

















# load custom library for LM utilities like dataset etc.
from lmUtilsC import *

from torch.utils.data.dataloader import DataLoader

if __name__ == "__main__":

    learningrate = 0.001


    outeriterations = 0
    #outeriterations = 4000 # for testing architecture design choices


    avgLoss = None





    modelIdStr = "Qwen/Qwen3-8B-base"
    tokenizer = AutoTokenizer.from_pretrained(modelIdStr)

    config = ConfigB()
    config.block_size = 32 # contextsize of 32 for testing!


    dataTxtArr = []
    # dataTxtArr.append("This is a small test to the the LM! A B C D E F G H I J K L M N O P Q R S T U V W X Y Z @ @ @ %")

    pathText = None

    # for manual testing / debugging of the architecture
    pathText = "/zfsPoolF/TYPE_mlDatasets/txtProto/languageSimpleA/shakespearMidsummer.txt"

    pathText = "/zfsPoolF/TYPE_mlDatasets/fullDatasetA/GPT-4 Technical Report.txt"

    f = open(pathText, "r")
    txt = f.read()
    dataTxtArr.append(txt[0:40000])
    del f



    dataset = TokenDataset(tokenizer, config, dataTxtArr)




    # setup the dataloader
    train_loader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=1,
        num_workers=1,
    )







    # train for n epochs
    nEpochs = 500.0
    outeriterations = int(len(dataset) / config.block_size * nEpochs)




    device = torch.device("cuda:0")

    fwpNn = FwpLmVariantC()


    # warning : number of dynamic parameters grows quadratically with this parameter!
    fwpNn.datapathSize = 180

    fwpNn.xEmbeddingsSize = fwpNn.datapathSize
    
    fwpNn.nonlinearAfterSlowHiddenSize = 160

    fwpNn.xEmbeddingsNum = dataset.get_vocab_size()


    fwpNn.softmaxOutputHeadSize = dataset.get_vocab_size()



    fwpNn.buildNn()

    fwpNn = fwpNn.to(device) # move to device


    optimizer = torch.optim.AdamW(fwpNn.parameters(), lr=learningrate)



    data_iter = iter(train_loader)




    



    for itOuter in range(outeriterations):


        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        #device = 'cpu'
        batch = [t.to(device) for t in batch]
        x, y = batch

        x, y = x[0], y[0] # take first batch


        #print(x)
        #print(y)






        optimizer.zero_grad()


        fwpNn.reset()







        lossSum = 0.0

        for innerItIdx in range(x.size()[0]):
            
            # DBG
            #print("")
            #print(f"inner it {innerItIdx}")
            
            helperA = torch.tensor([x[innerItIdx].item()])
            helperA = helperA.to(device)
            nnX = fwpNn.xEmbeddings(helperA)[0] # lookup embedding vector by symbol
            
            #print(nnX) # DBG

            targetSymbol = y[innerItIdx].item()

            target = torch.tensor([targetSymbol], dtype=torch.int64)
            target = target.to(device)

            # feed X into "Fast Weight Programmer" model
            logits, loss = fwpNn.forwardAndUpdate(nnX, target)

            # print(z1) # DBG

            # wrong
            #arr = [1e-7]*fwpNn.softmaxOutputHeadSize
            #arr[5] = 1.0
            #target = torch.tensor(arr)
            #target = torch.nn.functional.normalize(target, dim=0)
            
            
            
            
            
            if innerItIdx >= (2-1): # do only predict the following n tokens
                #loss = torch.nn.functional.cross_entropy(probs, target)
                #print(loss) # DBG

                lossSum += loss



        lossSum.backward()

        optimizer.step()



        avgLossOfTokens = lossSum.item()/(x.size()[0]-(2+1))

        if avgLoss is None:
            avgLoss = avgLossOfTokens
        else:
            lossStatFactor = 0.994

            avgLoss = avgLoss * lossStatFactor + avgLossOfTokens * (1.0 - lossStatFactor)


        if itOuter % 5 == 0:
            print("")
            print(f"outerIt={itOuter}")
            epoch = itOuter / (len(dataset) / config.block_size)
            print(f"epoch={epoch}")
            
            print(f"avgLossOfTokens={avgLossOfTokens}")
            print(f"avgLoss={avgLoss}")

        if itOuter > 0 and itOuter % 500 == 0:

            # save the latest model
            print("info: saving model")
            modelName = "fwpBmodel_A.torchmodel"
            import os
            pathCheckpoint = os.path.join(os.getcwd(), modelName)
            torch.save(fwpNn.state_dict(), pathCheckpoint)








print("FINISHED PROGRAM!")

