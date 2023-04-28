# PROTOtype of NAL based LM
# idea: learn the distribution as truth-values and select for prediction the ones with the highest exp

import std/strformat
import std/sequtils

type
  DetectorRef* = ref object
    premise*: seq[int] # tokens
    prediction*: int # token
    #tv*: Tv

# detector which looks out for a single token in the window and votes for the prediction
type
  DetectorSingleRef* = ref object
    premise*: int # tokens
    premiseIdx*: int # index
    prediction*: int # token
    #tv*: Tv

var nAlphabet: int = 127

type SymLmRef* = ref object
  detectors*: seq[DetectorRef]
  detectorsSingle*: seq[DetectorSingleRef]

  windowLenPlusPred*: int # PARAM - window size + 1 token for predicted token

  nAlphabet*: int

var symLm: SymLmRef = new (SymLmRef)
symLm.windowLenPlusPred = 16
symLm.nAlphabet = 127


proc train(seq0: seq[int]) =
  for iIdx in 0..<seq0.len-symLm.windowLenPlusPred:
    
    let sub: seq[int] = seq0[iIdx..iIdx+symLm.windowLenPlusPred-1]
    #echo("") # DBG
    #echo(sub) # DBG

    let premise: seq[int] = sub[0..sub.len-1-1]
    #echo(premise) # DBG
    let predictionY: int = sub[sub.len-1]
    #echo(predictionY) # DBG

    block:
      var d: DetectorRef = new (DetectorRef)
      d.premise = premise
      d.prediction = predictionY
      symLm.detectors.add(d)

    # build single predictor
    block:
      for iPremiseIdx in 0..<premise.len:
        var d: DetectorSingleRef = new (DetectorSingleRef)
        d.premise = premise[iPremiseIdx]
        d.premiseIdx = iPremiseIdx
        d.prediction = predictionY
        symLm.detectorsSingle.add(d)
        
      





# match detector

proc calcPredictedDistributionWithCount(stimulus: seq[int]): seq[int] =
    var votingOut: seq[int] = @[] # voting counters
    for z in 0..<symLm.nAlphabet:
        votingOut.add(0)
    
    for iDetector in symLm.detectors:
        if stimulus == iDetector.premise:
            let prediction: int = iDetector.prediction

            #echo(prediction) # DBG

            votingOut[prediction] = 1
            return votingOut # we found exact match as in training data, return



    
    for iPremiseIdx in 0..<stimulus.len:
        let iPremiseToken: int = stimulus[iPremiseIdx]

        for iDetector in symLm.detectorsSingle:
            # * vote for output token
            if iPremiseIdx == iDetector.premiseIdx and iPremiseToken == iDetector.premise:
                #echo(&"detectorSingle hit predictionY={iDetector.prediction}") # DBG
                votingOut[iDetector.prediction]+=1 # vote positive

    return votingOut


# find distribution of prediction
proc calcPredictedDistribution(stimulus: seq[int]): seq[float] =
    let disrtCounts: seq[int] = calcPredictedDistributionWithCount(stimulus)

    var votingOut: seq[float] = disrtCounts.map(proc(z: int): float = float(z))  # voting counters

    # normalize votes to get probability distribution
    var sum=0.0
    for z in votingOut:
        sum+=z
    
    votingOut = votingOut.map(proc(z: float): float = z/sum) # normalize

    # print probability distribution
    #echo(votingOut) # DBG

    return votingOut

# TODO< move into older code and use tokenizer! >









# training seq
var seq0: seq[int] = @[]

seq0 = @[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 7, 8, 9, 1, 1, 1, 5]
seq0 = @[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 
func tokenize2(text: string): seq[int] =
  var res: seq[int] = @[]
  for z in text:
    let `ord` = int(z) # compute ASCII code of letter
    res.add(`ord`)
  return res


# generate training data + tokenize
block:
  #var specialChars: seq[string] = @["0","1","2","3","4","5","6","7","8","9","-",">","(",")",";",",","-",">"]

  var txt0: seq[string] = @[
    "Y is a Z->isA(Y,Z)",
    "Y is an Z->isA(Y,Z)",
    "Y is Z->isA(Y,Z)",
    "X and Y are Z->isA(X,Z); isA(Y,Z)",
    "X and Y are a Z->isA(X,Z); isA(Y,Z)",
    "X and Y are an Z->isA(X,Z); isA(Y,Z)",
    "Y can Z->rel(Y,can,Z)",
    "X and Y can Z->rel(X,can,Z); rel(Y,can,Z)",
    "X is a Y and can Z->isA(X,Y); rel(X,can,Z)",
    "X and Y can Z->rel(Y,can,Z); rel(X,can,Z)",
    "Y has Z->rel(Y,has,Z)",
    "Y has a Z->rel(Y,has,Z)",
    "Y has an Z->rel(Y,has,Z)",
    "X and Y has Z->rel(Y,has,Z),rel(X,has,Z)",
  ]

  var txt1: string = ""
  for z in txt0:
    txt1=txt1&z&"\n"
  
  for z in txt0:
    # generate tokens for each line and append
    # TODO< don't append to seq0! create a new learning set! >
    seq0=seq0 & tokenize2(z)







# generate random looking tokens
block:
  var z0: float = 4.4333844787
  
  for z in 0..<50:
    var tokenId: int = int(z0*z0*58604.4847383994) mod symLm.nAlphabet
    z0 = z0 * 1.19383829034 + 0.483393
    seq0.add(tokenId)



train(seq0)

var stimulus: seq[int]
stimulus = @[0, 1, 2, 3, 0, 7, 8] # as in training data
stimulus = @[0, 1, 2, 3, 0, 7, 7] # not in training data
stimulus = @[0, 0, 0, 0, 0, 0, 50] # not in training data

#let t0: seq[int] = tokenize2("Y has Z->")
let t0: seq[int] = tokenize2("X and Y has Z->")
echo(t0)
stimulus = t0[max(t0.len-(symLm.windowLenPlusPred-1),0)..t0.len-1]# only take last n tokens
echo(stimulus)

# fill up
block:
  #let 
  for z in 0..<max(0, symLm.windowLenPlusPred-1-stimulus.len):
    stimulus=0&stimulus
  discard

echo(stimulus)


let predictionDistr: seq[float] = calcPredictedDistribution(stimulus)
echo(predictionDistr)


# TODO< implement rule to copy from a token at index X to prediction if token at index Y equals a observed token >

