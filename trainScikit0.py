from sklearn.neural_network import MLPRegressor
import numpy as np


trainingTuples = []
#trainingTuples.append(([0.9, -0.2, 0.1, 0.2, 0.2, 0.1], [1.0, 0.1]))

print('import...')
#import dat055609 # load training data
#trainingTuples = dat055609.trainingTuples1Db

from Db import *

class TrainingDatSrcDb(object):
    def __init__(self):
        self.db=Db()
        self.db.open('./b.sqliteDb',False)
  
    def getDatAt(self, selIdx):
        dbRes=self.db.queryById(selIdx)
        #print(dbRes) #dbg
        return dbRes

    #def getDat(self):
    #  selIdx=1+random.randint(0,self.retCnt()-1)
    # return self.getDatAt(selfIdx)
  
    def retCnt(self):
        return 17935-1

datSrc = TrainingDatSrcDb()
for iIdx in range(datSrc.retCnt()):
    #print(iIdx)
    dat = datSrc.getDatAt(iIdx+1)
    trainingTuples.append((dat[0], dat[1]))

print('...done')


print('convert to scikit data...')

xSet = None
ySet = None

# convert training tuples to numpy arrays and append
cnt = 0
for xArr, yArr in trainingTuples:
    if (cnt%500)==0:
        print(cnt)
    
    
    if xSet is None:
        xSet = np.array([xArr])
    else:
        xSet = np.concatenate((xSet, np.array([xArr])), axis=0)

    if ySet is None:
        ySet = np.array([yArr])
    else:
        ySet = np.concatenate((ySet, np.array([yArr])), axis=0)
    
    cnt+=1

if False: # data normalization
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(xSet)

    print(scaler.mean_)
    print(scaler.scale_)
    
    xSetScaled = scaler.transform(xSet)
    xSet = xSetScaled

        
# implementation of "Extreme Learning Machines" (ELM)


print('training...')

# 300 neurons for 1500 lines looks good
nNeurons = 300#300#1000 #5000 #2600 #300 #190 #250 # 285
solverName = 'lbfgs' # ELM
solverName = 'adam'
elm = MLPRegressor(hidden_layer_sizes=(nNeurons,), activation='tanh', solver=solverName, max_iter=800000, verbose=True, tol=1e-6)#, random_state=1553)

# Fit the model to the training data
elm.fit(xSet, ySet)

print('...done')


if True:
    # Predict the test set
    xTest= np.array([trainingTuples[0][0]])
    print('xTest=')
    print(xTest)
    yPrediction = elm.predict(xTest)

    print(yPrediction)


if False: # debug weights+bias of learned NN
    print(len(elm.coefs_))
    
    for iArr in elm.coefs_[0]:
        print(iArr)
    
    print("---")
    
    for iArr in elm.coefs_[1]:
        print(iArr)
    
    
    print("====")
    
    for iV in elm.intercepts_:
        print(iV)

        
if True: # export json
    
    print('write output...')
    
    f=open('outModel.json','w')
    
    f.write('{\n')
    f.write('"layers": [\n')
    
    idx0=0
    
    for layerIdx in range(len(elm.coefs_)):

        paramsLayer = elm.coefs_[layerIdx]
        nWeights = len(paramsLayer)
        nNeurons = len(paramsLayer[0])
        
        f.write('{\n')
        f.write('"neurons": \n')
        f.write('[\n')
        
        idx1 = 0
        for iNeuronIdx in range(nNeurons):
            weightsArr = []
            for iWeightIdx in range(nWeights):
                weightsArr.append(paramsLayer[iWeightIdx][iNeuronIdx])
            
            bias = elm.intercepts_[layerIdx][iNeuronIdx]
            
            weightsInnerStr = ",".join(map(lambda iv:str(iv),weightsArr))
            
            f.write(f"""{{"w": [{weightsInnerStr}],"b": {bias}}}\n""")
            
            if idx1 != nNeurons-1:
                f.write(',')
            
            idx1+=1

        f.write(']\n')
        f.write('}\n')
        
        if idx0!=len(elm.coefs_)-1:
            f.write(',\n')
        
        idx0+=1
    
    f.write(']}\n')
    
    f.close()
    
    print('...done')

    
