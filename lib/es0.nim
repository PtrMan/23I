

# DRAFT of ES learning algorithm

# paper "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" https://arxiv.org/pdf/1703.03864.pdf

import std/strformat
import std/random
import math

# TODO REFACTOR<move out of this file>
# generate random vector
proc vecGenRng*(rng: var Rand, len:int): seq[float64] =
    var temp0: seq[float64] = @[]
    for i in 1..len:
        temp0 = temp0 & @[(1.0-rng.rand(2.0))]
    return temp0

proc vecGenNull*(len:int): seq[float64] =
    var temp0: seq[float64] = @[]
    for i in 0..<len:
        temp0 = temp0 & @[0.0]
    return temp0

proc vecGenGaussian*(sigma: float64, len: int, rng: var Rand): seq[float64] =
  var res: seq[float64] = @[]
  for idx in 0..<len:
    res.add(gauss(rng, 0.0, sigma))
  return res


proc vecAdd*(a: seq[float64], b: seq[float64]): seq[float64] =
  var res: seq[float64] = @[]
  for idx in 0..<a.len:
    res.add(a[idx]+b[idx])
  return res


func calcSigmoid*(x: float): float =
  return 1.0 / (1.0 + math.exp(-x))


type
  Unit1* = object
    r*: seq[float64]
    #d*: seq[float64]
    
    biasR*: float64
    #biasD*: float64

type
  Layer1* = object
    units*: seq[Unit1]
    
    # 0: identity
    # 1: relu
    # 2: sigmoid
    actFn*: int

type
  Network1* = object
    layers*: seq[Layer1]

proc dotProduct*(a: seq[float64], b: seq[float64]): float64 =
  var res=0.0
  for idx in 0..<a.len:
    res+=(a[idx]*b[idx])
  return res

proc scale*(v: seq[float64], s: float64): seq[float64] =
  var res: seq[float64] = @[]
  for iv in v:
    res.add(iv*s)
  return res

proc calcLayer*(self: Layer1, x: seq[float64]): seq[float64] =
  var res: seq[float64] = @[]

  for iUnit in self.units:
    var v = dotProduct(x, iUnit.r) + iUnit.biasR

    if self.actFn == 0: # identity
      discard
    elif self.actFn == 1: # relu
      v = max(v,0.0)
    elif self.actFn == 2: # sigmoid
      v = calcSigmoid(v)

    res.add(v)

  return res


# evolutionar algorithm: evolutionary strategies

type
  EsCandidate0Ref* = ref EsCandidate0
  EsCandidate0* = object
    params*: seq[float64]
    score*: float64
    

type
  EsCtx1* = object
    params*: seq[float64]
    #candidates*: seq[EsCandidate0]

    nCandidates*: int # PARAM
    sigma*: float64 # PARAM



# procedure to build NN from parameters
proc buildNn*(params: seq[float64]): Network1 =
  var idx = 0
  var layers: seq[Layer1] = @[]

  var kIdx = 0
  var k = @[7, 12, 12, 5] # width of input of layer
  
  var nLayers: int = k.len-1 # with output layer

  for iLayerIdx in 0..<nLayers:

    var createdLayer: Layer1 = Layer1(units: @[], actFn: 1)
    if iLayerIdx == 0:
      createdLayer.actFn = 0 # identity
    if iLayerIdx == nLayers-1:
      createdLayer.actFn = 2 # sigmoid

    for iUnitIdx in 0..<k[kIdx+1]:
      var createdUnit: Unit1 = Unit1(r: @[], biasR:0.0)
      createdUnit.biasR = params[idx]
      idx+=1

      for j in 0..<k[kIdx]:
        createdUnit.r.add(params[idx])
        idx+=1

      createdLayer.units.add(createdUnit)
    
    layers.add(createdLayer)

    kIdx+=1

  return Network1(layers: layers)




when isMainModule:

  
  proc evalCandidates(candidates: seq[EsCandidate0Ref]) =
    var verbosity: int = 0 # verbosity for debugging/tracing
    
    var bestMse: float64 = 10.0e10

    for candidateIdx in 0..<candidates.len:
      var selCandidate = candidates[candidateIdx]
      
      var params: seq[float64] = selCandidate.params
      #echo(&"params={params}") # DBG

      # * build

      var network: Network1 = buildNn(params)

      # * evaluate

      var stimulus: seq[float64] = @[0.5, 1.0, 0.8]
      
      var x: seq[float64] = stimulus

      for iLayer in network.layers:
        x = calcLayer(iLayer, x)
        #echo(x) # DBG
      
      var y: seq[float64] = x
      
      if verbosity>=2:
        echo(y)


      # compute similarity to y and score

      var yTarget: seq[float64] = @[0.1, 0.6, 0.7, 0.8, 0.3]
      
      var mse = 0.0

      for idx in 0..<y.len:
        let diff = y[idx]-yTarget[idx]
        let dist = abs(diff)
        mse+=(dist*dist)
      
      selCandidate.score = 60.0-pow(2.0+mse, 3.5) # might be buggy
      selCandidate.score = max(6.0-mse, 0.0)

      bestMse = min(bestMse, mse) # update statistics for this iteration of optimization


      if verbosity>=1:
        echo(&"mse={mse} score={selCandidate.score}")
    
    echo(&"bestMse={bestMse}")







  #var rng: Rand = initRand(47463)
  var rng: Rand = initRand()

  var vecLen: int = (9+1)*12*5

  var ctx: EsCtx1 = EsCtx1(params: @[], nCandidates: 50, sigma: 1.5)
  ctx.params = vecGenRng(rng, vecLen)


  


  for it in 0..<50000: # optimization loop

    if (it + int(2000/2)) mod 2000 == 0:
      ctx.sigma *= 0.5


    if true: # DBG?
      echo("")
      echo("")
      echo("")
    
    var candidates: seq[EsCandidate0Ref] = @[]

    block: # generate candidates
      for n in 0..<ctx.nCandidates:
        var candidate: EsCandidate0Ref = EsCandidate0Ref(params: @[], score: 0.0)
        candidate.params = vecAdd(ctx.params, vecGenGaussian(ctx.sigma, ctx.params.len, rng))
        candidates.add(candidate)
    
    
    evalCandidates(candidates)




    # compute best score
    var bestScore: float64 = -1e20
    for iCandidate in candidates:
      bestScore = max(bestScore, iCandidate.score)
    
    if true:
      echo(&"best score of iteration={it} sigma={ctx.sigma} bestScore={bestScore}")

    # * compute new parameters
    var paramAccu: seq[float64] = vecGenNull(vecLen)
    for iCandidate in candidates:
      var scaledParameters: seq[float64] = scale(iCandidate.params, iCandidate.score)
    
    paramAccu = scale(paramAccu, 1.0/(float64(candidates.len)*ctx.sigma)) # compute average
    
    ctx.params = vecAdd(ctx.params, paramAccu)






