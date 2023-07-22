import random
import math

class Candidate(object):
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.score = -1.0e15

class Z(object):
    # dot product
    @staticmethod
    def dot(a, b):
        z = 0.0
        for idx in range(len(a)):
            z += (a[idx]*b[idx])
        return z

    # sub
    @staticmethod
    def fn1(a, b):
        z = []
        for idx in range(len(a)):
            z.append(a[idx]-b[idx])
        return z
    

    @staticmethod
    def add(a, b):
        z = []
        for idx in range(len(a)):
            z.append(a[idx]+b[idx])
        return z
    
    # compute L2 norm
    @staticmethod
    def calcL2norm(a):
        z = 0.0
        for idx in range(len(a)):
            z+=(a[idx]*a[idx])
        return math.sqrt(z)
    
    @staticmethod
    def fn4(a, b):
        z = []
        for idx in range(len(a)):
            z.append(a[idx]+b[idx])
        return z
    
    @staticmethod
    def scale(v, s):
        z = []
        for idx in range(len(v)):
            z.append(v[idx]*s)
        return z
    
    @staticmethod
    def calcSigmoidActFn(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def calcReluActFn(x):
        return max(x, 0.0)
    
    @staticmethod
    def genNull(size):
        z = []
        for idx in range(size):
            z.append(0.0)
        return z

    @staticmethod
    def fn20(size):
        z = []
        for idx in range(size):
            z.append(random.random()*2.0-1.0)
        return z
    
    @staticmethod
    def fnGenVecGaussian(size):
        z = []
        for idx in range(size):
            z.append(Z.gaussianRng()[0])
        return z
    
    @staticmethod
    def gaussianRng(mu=0.0, sigma=1.0):
        u1 = random.random()
        u2 = random.random()

        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

        return mu + z1 * sigma, mu + z2 * sigma
    
    @staticmethod
    def calcScore(candidate):

        # stimuli of NN
        z0 = [0.56, 0.65, 0.5555, 0.1, 0.1,  0.1, 0.1, 0.1, 0.1, 0.1]
        
        biasIdx = 0
        weightIdx = 0

        y = []

        for iNeuronIdx in range(2):
            z2 = Z.dot(z0, candidate.weights[weightIdx:weightIdx+len(z0)])
            z3 = candidate.biases[biasIdx] + z2
            z4 = Z.calcReluActFn(z3)
            y.append(z4)

            weightIdx+=len(z0)
            biasIdx+=1



        

        #print(f'layer out ={y}') # DBG

        score = len(y)+3.0
        score = score-math.pow(abs(y[0] - 0.5), 1.8)
        score = score-math.pow(abs(y[1] - 0.9), 1.8)
        score = max(score, 1.0e-7) # prevent negative score
        return score
    
    @staticmethod
    def fn1000():
        vecSize = 20
        biasSize = 2

        populationCount = 38

        candidate = Candidate(Z.fnGenVecGaussian(vecSize), Z.fnGenVecGaussian(biasSize))


        for iteration in range(80000):
            print('')
            print(f'epoch={iteration}')


            nextPopulation = []
            for it in range(populationCount):

                # mutate weights
                weights = Z.add(candidate.weights, Z.scale(Z.fnGenVecGaussian(vecSize), 0.006))

                # mutate biases
                biases = candidate.biases[:]
                if random.random() < 0.3: # mutate biases?
                    selIdx=random.randint(0, len(biases)-1)
                    biases[selIdx]+=((random.random()*2.0-1.0)*0.001)

                candidate.biases = biases

                nextPopulation.append(Candidate(weights, biases))

            # score candidates
            for idx in range(len(nextPopulation)):
                score = Z.calcScore(nextPopulation[idx])
                print(f'score of candidate [{idx}]={score}')
                nextPopulation[idx].score = score

            # compute new candidate
            weightsSum = Z.genNull(len(nextPopulation[idx].weights))
            biasesSum = Z.genNull(len(nextPopulation[idx].biases))
            for idx in range(len(nextPopulation)):
                nextPopulation[idx].weights = Z.scale(nextPopulation[idx].weights, nextPopulation[idx].score)
                nextPopulation[idx].biases = Z.scale(nextPopulation[idx].biases, nextPopulation[idx].score)

                weightsSum = Z.add(weightsSum, nextPopulation[idx].weights)
                biasesSum = Z.add(biasesSum, nextPopulation[idx].biases)
            
            scoreSum = sum(map(lambda z:z.score, nextPopulation))

            weightsSum = Z.scale(weightsSum, 1.0/scoreSum)
            biasesSum = Z.scale(biasesSum, 1.0/scoreSum)

            candidate.weights = weightsSum
            candidate.biases = biasesSum




# entry
Z.fn1000()
