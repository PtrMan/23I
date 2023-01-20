# run with
#    nim compile --run -d:release protoBp0.nim 

# version 12.1.2023

import std/strformat
import std/random
import math

proc adAdd*(ar: float64, ad: float64, br: float64, bd: float64): tuple[r:float64, d:float64] =
  return (r:ar+br,d:ad+bd)

proc adSub*(ar: float64, ad: float64, br: float64, bd: float64): tuple[r:float64, d:float64] =
  return (r:ar-br,d:ad-bd)

proc adMul*(ar: float64, ad: float64, br: float64, bd: float64): tuple[r:float64, d:float64] =
  return (r:ar*br,d:ad*br+bd*ar)

proc adDiv*(ar: float64, ad: float64, br: float64, bd: float64): tuple[r:float64, d:float64] =
  return (r:ar/br,d:(ad*br+bd*ar) / (br*br))


proc adSqrt*(r: float64, d: float64): tuple[r:float64, d:float64] =
  # DEBUG
  if r < 0.0 or r != r:
    echo(&"invalid input to adSqrt() r={r}")
    quit(1)
    
  if d != d:
    echo(&"invalid input to adSqrt() d={d}")
    quit(1)

  let resR: float64 = sqrt(r)
  let resD: float64 = d*0.5*pow(max(r, 1e-10), 0.5-1.0)

  # DEBUG
  if resD != resD:
    echo(&"result D adSqrt() is NaN")
    echo(&"in  r={r} d={d}")
    quit(1)
  
  
  return (r:resR, d:resD)




# activation function
proc adActRelu(r: float64, d: float64): tuple[r:float64, d:float64] =
  if r > 0.0:
    return (r:r,d:d)
  return (r:0.0,d:0.0)


type
  Unit*[s: static int] = object
    r*: array[s, float64]
    d*: array[s, float64]
    
    biasR*: float64
    biasD*: float64


# generator for training data
type DatGenerator* = object
  dat*: seq[tuple[inArrays:seq[seq[float64]], target:seq[float64], gradientStrength: float64]] # gradient strength: how strong is the gradient? usually 1.0 for :normal: supervised learning, equal to reward for policy gradient training
  rng*: Rand

proc retSample(this: var DatGenerator): tuple[inArrays:seq[seq[float64]], target:seq[float64], gradientStrength: float64] =
  let idx=this.rng.rand(this.dat.len-1)
  return this.dat[idx]





type
  NnOptimizerConfig* = object
    searchEpochs*: int64 # number of epochs for search
    lr*: float32 # learning rate


type CalcErrorPtr = (proc(nnOuts: seq[seq[tuple[r:float64,d:float64]]], target: seq[float64], selTargetIdx: int):tuple[r:float64,d:float64])

type
  Layer*[s: static int] = object
    units*: seq[Unit[s]]


# forward pass in NN
proc nnForward*[layer0StimulusWidth: static int, nUnitsPerLayer: static seq[int], nUnitsPerLayer0: static int](layer0: Layer[layer0StimulusWidth], layer1: Layer[nUnitsPerLayer0], inArray: seq[float64]): seq[tuple[r:float64,d:float64]] =
  proc calcUnitsOut[unitsPerLayer: static int, s: static int](layerIdx: int, units: seq[Unit[s]], br: var array[s, float64], bd: var array[s, float64], res0R: var array[unitsPerLayer, float64], res0D: var array[unitsPerLayer, float64]) =
    
    for iUnitIdx in 0..units.len-1:
      
      # implementation of dot-product
      var sumR: float64 = 0.0
      var sumD: float64 = 0.0
      
      for iidx in 0..units[iUnitIdx].r.len-1:
        let r0 = adMul(units[iUnitIdx].r[iidx], units[iUnitIdx].d[iidx], br[iidx], bd[iidx])
        #echo &"{r0.r} {r0.d}" # DBG
        
        let r1 = adAdd(sumR, sumD, r0.r, r0.d)
        sumR = r1.r
        sumD = r1.d
    
      # add bias  
      block:
        let r1 = adAdd(sumR, sumD, units[iUnitIdx].biasR, units[iUnitIdx].biasD)
        sumR = r1.r
        sumD = r1.d
      
      # compute activation function
      var actFnRes: tuple[r:float64,d:float64] = (sumR,sumD) # set to id
      if layerIdx == 0:
        actFnRes = adActRelu(sumR, sumD)


      res0R[iUnitIdx] = actFnRes.r
      res0D[iUnitIdx] = actFnRes.d

    
      #echo &"{sumR} {sumD}"
        

  # set stimulus
  var br: array[layer0StimulusWidth, float64]
  var bd: array[layer0StimulusWidth, float64]

  for iidx in 0..br.len-1:
    br[iidx] = inArray[iidx]


  # array for results of units
  var res0R: array[nUnitsPerLayer[0], float64]
  var res0D: array[nUnitsPerLayer[0], float64]
  calcUnitsOut(0, layer0.units, br,bd,  res0R,res0D)

  var res1R: array[nUnitsPerLayer[1], float64]
  var res1D: array[nUnitsPerLayer[1], float64]
  calcUnitsOut(1, layer1.units, res0R,res0D, res1R,res1D)
  

  # translate output of NN to seq
  var nnOut: seq[tuple[r:float64,d:float64]] = @[]
  for iidx in 0..res1R.len-1:
    nnOut.add((res1R[iidx],res1D[iidx]))
  
  return nnOut



type
  TrainingConfig* = object
    ticksOutProgress*: int # numer of ticks to debug next training progress to terminal
    latestWeights: seq[float64] # gets filled with the latest weights which were learned





import threadpool, os

type SharedChannel[T] = ptr Channel[T]

proc newSharedChannel[T](): SharedChannel[T] =
  result = cast[SharedChannel[T]](allocShared0(sizeof(Channel[T])))
  open(result[])

proc close[T](ch: var SharedChannel[T]) =
  close(ch[])
  deallocShared(ch)
  ch = nil

proc send[T](ch: SharedChannel[T], content: T) =
  ch[].send(content)


proc recv[T](ch: SharedChannel[T]): T =
  result = ch[].recv


# type for selection
type
  Addr = object
    layerIdx: int
    unitIdx: int
    weightIdx: int # can be -1 for bias

    selTargetIdx: int


type MsgTypeEnum = enum
  taskAssign, weightUpdate, terminate


type Msg0 = object
  case `type`: MsgTypeEnum 
  of taskAssign: # task assignment
    address: Addr
    selSample: tuple[inArrays:seq[seq[float64]], target:seq[float64], gradientStrength: float64]  # sample to optimize for
  of weightUpdate: # weight update
    address1: Addr
    deltaR: float64
  of terminate:
    discard


#type Msg0 = object
#  address: Addr
#  selSample: tuple[inArrays:seq[seq[float64]], target:seq[float64], gradientStrength: float64]  # sample to optimize for

# type for weight update
type WeightUpdate = object
  address: Addr
  nnOuts: seq[seq[ tuple[r:float64,d:float64] ]]
  selSample: tuple[inArrays:seq[seq[float64]], target:seq[float64], gradientStrength: float64]  # sample to optimize for
  #deltaR: float64 # delta of weight or bias


proc workerThread[
  layer0StimulusWidth: static int,
  nUnitsPerLayer: static seq[int],
  targetLen: static int
  ](ch: (SharedChannel[Msg0], SharedChannel[WeightUpdate])) {.thread.} =
  let (mainChannel, responseChannel) = ch
  


  var r = initRand(234)


  var layer0: Layer[layer0StimulusWidth]
  var layer1: Layer[nUnitsPerLayer[0]]

  
  

  for i in 0..nUnitsPerLayer[0]-1:
    var u0: Unit[layer0StimulusWidth] = Unit[layer0StimulusWidth]()
    u0.biasR = (1.0-r.rand(2.0))*0.8
    for iidx in 0..u0.r.len-1:
      let nFanIn = layer0StimulusWidth
      let nFanOut = nUnitsPerLayer[1]
      # see https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
      # He Uniform Initialization
      let v: float64 = r.rand(-sqrt(1.0/float64(nFanIn))..sqrt(1.0/float64(nFanOut)))

      u0.r[iidx] = v #(1.0-r.rand(2.0))*0.8
    
    layer0.units.add(u0)

  for i in 0..nUnitsPerLayer[1]-1:
    var u0: Unit[nUnitsPerLayer[0]] = Unit[nUnitsPerLayer[0]]()
    u0.biasR = (1.0-r.rand(2.0))*0.8
    for iidx in 0..u0.r.len-1:
      u0.r[iidx] = (1.0-r.rand(2.0))*0.8
    
    layer1.units.add(u0)


  
  
  while true: # loop to receive tasks, do optimization, send result
    ##echo("worker: recv wait...")
    let task: Msg0 = mainChannel.recv()
    ##echo("worker: ...recv'ed")

    case task.`type`
    of terminate:
      break # terminate this worker by exiting the loop
    
    of weightUpdate:

      let selLayerIdx: int = task.address1.layerIdx
      let selUnitIdx: int = task.address1.unitIdx
      let selWeightIdx: int = task.address1.weightIdx


      if selWeightIdx == -1:
        if selLayerIdx == 0:
          layer0.units[selUnitIdx].biasR += task.deltaR
        elif selLayerIdx == 1:
          layer1.units[selUnitIdx].biasR += task.deltaR

      else:
        if selLayerIdx == 0:
          layer0.units[selUnitIdx].r[selWeightIdx] += task.deltaR
        elif selLayerIdx == 1:
          layer1.units[selUnitIdx].r[selWeightIdx] += task.deltaR


    of taskAssign:

      let selLayerIdx: int = task.address.layerIdx
      let selUnitIdx: int = task.address.unitIdx
      let selWeightIdx: int = task.address.weightIdx



      if selWeightIdx == -1:
        if selLayerIdx == 0:
          layer0.units[selUnitIdx].biasD = task.selSample.gradientStrength
        elif selLayerIdx == 1:
          layer1.units[selUnitIdx].biasD = task.selSample.gradientStrength
        
      else:
        if selLayerIdx == 0:
          layer0.units[selUnitIdx].d[selWeightIdx] = task.selSample.gradientStrength
        elif selLayerIdx == 1:
          layer1.units[selUnitIdx].d[selWeightIdx] = task.selSample.gradientStrength
      

      
      var nnOuts: seq[seq[tuple[r:float64,d:float64]]] = @[]
      
      
      for iParNnIdx in 0..task.selSample.inArrays.len-1: # iterator to iterate over parallel instantiations of the same network with different stimulus but same weights
        

        # TODO REFACTORME< use the forward function! >
        
        proc calcUnitsOut[unitsPerLayer: static int, s: static int](layerIdx: int, units: seq[Unit[s]], br: var array[s, float64], bd: var array[s, float64], res0R: var array[unitsPerLayer, float64], res0D: var array[unitsPerLayer, float64]) =
          
          for iUnitIdx in 0..units.len-1:
            
            # implementation of dot-product
            var sumR: float64 = 0.0
            var sumD: float64 = 0.0
            
            for iidx in 0..units[iUnitIdx].r.len-1:
              let r0 = adMul(units[iUnitIdx].r[iidx], units[iUnitIdx].d[iidx], br[iidx], bd[iidx])
              #echo &"{r0.r} {r0.d}" # DBG
              
              let r1 = adAdd(sumR, sumD, r0.r, r0.d)
              sumR = r1.r
              sumD = r1.d
          
            # add bias  
            block:
              let r1 = adAdd(sumR, sumD, units[iUnitIdx].biasR, units[iUnitIdx].biasD)
              sumR = r1.r
              sumD = r1.d
            
            # compute activation function
            var actFnRes: tuple[r:float64,d:float64] = (sumR,sumD) # set to id
            if layerIdx == 0:
              actFnRes = adActRelu(sumR, sumD)


            res0R[iUnitIdx] = actFnRes.r
            res0D[iUnitIdx] = actFnRes.d

        
          #echo &"{sumR} {sumD}"
            

        # set stimulus
        var br: array[layer0StimulusWidth, float64]
        var bd: array[layer0StimulusWidth, float64]

        for iidx in 0..br.len-1:
          br[iidx] = task.selSample.inArrays[iParNnIdx][iidx]


        # array for results of units
        var res0R: array[nUnitsPerLayer[0], float64]
        var res0D: array[nUnitsPerLayer[0], float64]
        calcUnitsOut(0, layer0.units, br,bd,  res0R,res0D)

        var res1R: array[nUnitsPerLayer[1], float64]
        var res1D: array[nUnitsPerLayer[1], float64]
        calcUnitsOut(1, layer1.units, res0R,res0D, res1R,res1D)
        

        # translate output of NN to seq
        var nnOut: seq[tuple[r:float64,d:float64]] = @[]
        for iidx in 0..res1R.len-1:
          nnOut.add((res1R[iidx],res1D[iidx]))
        nnOuts.add(nnOut)

      # function to calculate error
      #
      # is in a function to allow for flexible computation with a mathematical function between the output of the network and the output of the function to be optimized for
      #proc calcError(nnOuts: seq[seq[float64]], target: seq[float64], selTargetIdx: int): float64 =
      #  return  Ad(r:target[selTargetIdx] - nnOuts[0][selTargetIdx], d:res0D[selTargetIdx]) # compute error


      ##var nnTarget: seq[float64] = task.selSample.target
      #block:
      #  for iv in task.target:
      #    nnTarget.add(iv)

      ##var err1 = calcErrorFn(nnOuts, nnTarget,  task.address.selTargetIdx)

      # DEBUG
      #[
      if err1.r != err1.r: # encountered NaN
        echo("error - encountered NaN (err1.r)")
        quit(1)
      
      if err1.d != err1.d: # encountered NaN
        echo("error - encountered NaN (err1.d)")
        quit(1)
      ]#

      ##let deltaR: float = err1.r*err1.d*searchConfig.lr
      
      # DEBUG
      #if deltaR != deltaR: # encountered NaN
      #  echo("error - encountered NaN (deltaR)")
      #  quit(1)



      if selWeightIdx == -1:
        if selLayerIdx == 0:
          layer0.units[selUnitIdx].biasD = 0.0
        elif selLayerIdx == 1:
          layer1.units[selUnitIdx].biasD = 0.0
      
      else:
        if selLayerIdx == 0:
          layer0.units[selUnitIdx].d[selWeightIdx] = 0.0
        elif selLayerIdx == 1:
          layer1.units[selUnitIdx].d[selWeightIdx] = 0.0



      var response: WeightUpdate
      response.address = task.address
      #response.deltaR = deltaR
      response.nnOuts = nnOuts
      response.selSample = task.selSample

      responseChannel.send(response)
    






type
  WorkerCtxRef = ref WorkerCtx
  WorkerCtx = object
    fromMainToWorkerChannel: SharedChannel[Msg0]
    fromWorkerToMainChannel: SharedChannel[WeightUpdate]


# const layer0StimulusWidth: int = 5*(19) # count of stimulus real values
# nUnitsLayer0 = 5 #  how many units in layer 0

# optimization algorithm for training
proc z*[
  layer0StimulusWidth: static int,
  nUnitsPerLayer: static seq[int],
  targetLen: static int
  ](gen: var DatGenerator, calcErrorFn: CalcErrorPtr, searchConfig: NnOptimizerConfig, trainingConfig: var TrainingConfig) =
  
  var r = initRand(234)

  echo("start training...")

  #[
  var layer0: Layer[layer0StimulusWidth]
  var layer1: Layer[nUnitsPerLayer[0]]

  

  for i in 0..nUnitsPerLayer[0]-1:
    var u0: Unit[layer0StimulusWidth] = Unit[layer0StimulusWidth]()
    u0.biasR = (1.0-r.rand(2.0))*0.8
    for iidx in 0..u0.r.len-1:
      let nFanIn = layer0StimulusWidth
      let nFanOut = nUnitsPerLayer[1]
      # see https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
      # He Uniform Initialization
      let v: float64 = r.rand(-sqrt(1.0/float64(nFanIn))..sqrt(1.0/float64(nFanOut)))

      u0.r[iidx] = v #(1.0-r.rand(2.0))*0.8
    
    layer0.units.add(u0)

  for i in 0..nUnitsPerLayer[1]-1:
    var u0: Unit[nUnitsPerLayer[0]] = Unit[nUnitsPerLayer[0]]()
    u0.biasR = (1.0-r.rand(2.0))*0.8
    for iidx in 0..u0.r.len-1:
      u0.r[iidx] = (1.0-r.rand(2.0))*0.8
    
    layer1.units.add(u0)
  ]#
  
  var avgMse: float64 = 1.0


  


  # start worker threads
  #var threads: seq[ Thread[(SharedChannel[Msg0], SharedChannel[WeightUpdate])] ] = @[]
  var workerCtxs: seq[ WorkerCtxRef ] = @[]
  for iCnt in 0..1-1:
    var
      fromMainToWorkerChannel = newSharedChannel[Msg0]()
      fromWorkerToMainChannel = newSharedChannel[WeightUpdate]()
      th: Thread[(SharedChannel[Msg0], SharedChannel[WeightUpdate])]
  
    createThread(th, workerThread[layer0StimulusWidth, nUnitsPerLayer, targetLen], (fromMainToWorkerChannel, fromWorkerToMainChannel))
    #threads.add(th)

    let ctx: WorkerCtxRef = WorkerCtxRef(fromMainToWorkerChannel:fromMainToWorkerChannel, fromWorkerToMainChannel:fromWorkerToMainChannel)
    workerCtxs.add(ctx)
  



  for it in 0..layer0StimulusWidth*nUnitsPerLayer[0]*gen.dat.len*searchConfig.searchEpochs-1:
    let selSample = gen.retSample()

    # target to optimize for
    var target: array[targetLen, float64]
    for iidx in 0..targetLen-1:
        target[iidx] = selSample.target[iidx]

    
    

    for iWorkerCtx in workerCtxs:
      let selLayerIdx: int = r.rand(nUnitsPerLayer.len-1)

      var selUnitIdx = r.rand(nUnitsPerLayer[selLayerIdx]-1)
      
      # -1 indicates adaption of bias
      var selWeightIdx: int
      if selLayerIdx == 0:
        selWeightIdx = r.rand(layer0StimulusWidth-1 + 1)-1
      elif selLayerIdx == 1:
        selWeightIdx = r.rand(nUnitsPerLayer[0]-1 + 1)-1

      # index of target to optimize for
      let selTargetIdx: int = r.rand(target.len-1)

      block:
        var addr0: Addr
        addr0.layerIdx = selLayerIdx
        addr0.unitIdx = selUnitIdx
        addr0.weightIdx = selWeightIdx # can be -1 for bias
        addr0.selTargetIdx = selTargetIdx

        var msgOut: Msg0 = Msg0(`type`:taskAssign,address:addr0,selSample:selSample)

        ##echo("main: send taskAssign")
        iWorkerCtx.fromMainToWorkerChannel.send(msgOut)
      

    # wait for all responses
    var responses: seq[WeightUpdate] = @[]
    for iii in 0..workerCtxs.len-1:
      ##echo("main: recv...")
      let recvMsg: WeightUpdate = workerCtxs[iii].fromWorkerToMainChannel.recv()
      ##echo("main: ...recv'ed")
      responses.add(recvMsg)
    
    # compute weight updates
    var weightUpdateDeltaRs: seq[float64] = @[] # array of values of weight update deltas for each "response"

    for iweightUpdateIdx in 0..responses.len-1:
      let iResponse: WeightUpdate = responses[iweightUpdateIdx]
      

      ##var nnTarget: seq[float64] = task.selSample.target
      #block:
      #  for iv in task.target:
      #    nnTarget.add(iv)

      var err1 = calcErrorFn(iResponse.nnOuts, iResponse.selSample.target,  iResponse.address.selTargetIdx)

      # DEBUG
      #[
      if err1.r != err1.r: # encountered NaN
        echo("error - encountered NaN (err1.r)")
        quit(1)
      
      if err1.d != err1.d: # encountered NaN
        echo("error - encountered NaN (err1.d)")
        quit(1)
      ]#


      block: # compute and report avg-MSE
        if it == 0:
          avgMse = (err1.r*err1.r)

        if (it mod 1000) == 0 and selSample.gradientStrength > 0.0:
          let adaptFactor = 0.001
          avgMse = avgMse*(1.0-adaptFactor) + (err1.r*err1.r)*adaptFactor

        if (it mod trainingConfig.ticksOutProgress) == 0:
          echo &"avgMse={avgMse}\tmse={err1.r*err1.r}         errR[{iResponse.address.selTargetIdx}]={err1.r}"






      let deltaR: float = err1.r*err1.d*searchConfig.lr
      
      # DEBUG
      #if deltaR != deltaR: # encountered NaN
      #  echo("error - encountered NaN (deltaR)")
      #  quit(1)

      weightUpdateDeltaRs.add(deltaR)



    

    # broadcast weight updates
    block:
      for iWorkerCtx in workerCtxs:

        for iweightUpdateIdx in 0..responses.len-1:
          let iResponse: WeightUpdate = responses[iweightUpdateIdx]

          var msgOut: Msg0 = Msg0(`type`:weightUpdate,address1:iResponse.address, deltaR:weightUpdateDeltaRs[iweightUpdateIdx])

          ##echo("main: send weightUpdate")
          iWorkerCtx.fromMainToWorkerChannel.send(msgOut)







    #[ TOREFACTOR TOREFACTOR TOREFACTOR
    
    if (it mod (20000*150)) == 0:
      echo "store..."

      var wb: seq[float64] = @[]
      
      for iUnit in layer0.units:
        wb.add(iUnit.biasR)
        for iv in iUnit.r:
          wb.add(iv)
      for iUnit in layer1.units:
        wb.add(iUnit.biasR)
        for iv in iUnit.r:
          wb.add(iv)
      
      echo(wb)

      trainingConfig.latestWeights = wb

    let deltaR: float = err1.r*err1.d*searchConfig.lr
    
    # DEBUG
    if deltaR != deltaR: # encountered NaN
      echo("error - encountered NaN (deltaR)")
      quit(1)

    
    ]#






# TODO REFACTOR<move out of this file>
# generate random vector
proc vecGenRng*(rng: var Rand, len:int): seq[float64] =
    var temp0: seq[float64] = @[]
    for i in 1..len:
        temp0 = temp0 & @[(1.0-rng.rand(2.0))]
    return temp0
