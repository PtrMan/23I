# motivation: simple experiment of reservoir computing

import std/random
import std/math
import std/strformat


import bpeBase0



import math

proc zNormalize(arr: seq[float]): seq[float] =
  # chatGPT prompt: 
  # implement Z Normalization(Standardization) in Nim

  var mean = 0.0
  for i in 0..<arr.len:
    mean += arr[i]
  mean /= float(arr.len)

  var stdDev = 0.0
  for i in 0..<arr.len:
    stdDev += pow((arr[i] - mean), 2)
  stdDev = sqrt(stdDev / float(arr.len))
  
  var res: seq[float]
  for i in 0..<arr.len:
    res.add( (arr[i] - mean) / stdDev )

  return res


# helper to normalize data
proc normalizeHelper(v: seq[float], mean: float, stdDev: float): seq[float] =
  var res: seq[float]
  for i in 0..<v.len:
    res.add( (v[i] - mean) / stdDev )
  return res



var rng = initRand(4995843)

type WeightsAndBias1* = object
  w*: seq[float]
  b*: float

proc actFn1(x: float): float =
  return tanh(x)

proc actFn2(x: float): float =
  return cos(x)*x



func dot(a: seq[float], b: seq[float]): float =
  var res: float = 0.0
  for iIdx in 0..a.len-1:
    res += (a[iIdx]*b[iIdx])
  return res

func vecAdd(a: seq[float], b: seq[float]): seq[float] =
  var res: seq[float] = @[]
  for iIdx in 0..a.len-1:
    res.add(a[iIdx]+b[iIdx])
  return res

proc genRngVec(len: int, scale: float): seq[float] =
  var res: seq[float] = @[]
  for i in 0..len-1:
    res.add((rng.rand(2.0)-1.0)*scale)
  return res

proc genNullVec(len: int): seq[float] =
  var res: seq[float] = @[]
  for i in 0..len-1:
    res.add(0.0)
  return res

proc vecNormL2(v: seq[float]): float =
  return sqrt(dot(v,v))

proc vecCosSimilarity(a: seq[float], b: seq[float]): float =
  return dot(a, b) / (vecNormL2(a)*vecNormL2(b))

proc vecMakeVal(val: float, len: int): seq[float] =
  var res: seq[float] = @[]
  for i in 0..len-1:
    res.add(val)
  return res

#proc convToPythonArrStr(v: seq[float]): string =
#  var res: string = ""
#  for iIdx in 0..v.len-1-1:
#    res=res& &"{v[iIdx]}"&","
#  res=res& &"{v[v.len-1]}"
#  return "["&res&"]"

proc convToStrArrStr(v: seq[float]): string =
  var res: string = ""
  for iIdx in 0..v.len-1-1:
    res=res& &"{v[iIdx]}"&";"
  res=res& &"{v[v.len-1]}"
  return res


var trainingTokens: seq[int] = @[]

var bpeCtx: BpeCtx = BpeCtx(tokenIdCnt:256, tokenMap: @[])
for i in 0..bpeCtx.tokenIdCnt-1:
  bpeCtx.tokenMap.add(TokenIndirection(parents: @[])) # add "root" token

block: # convert text to BPE
  #var str: string = "A t-test is any statistical hypothesis test in which the test statistic follows a Student's t-distribution under the null hypothesis. It is most commonly applied when the test statistic would follow a normal distribution if the value of a scaling term in the test statistic were known (typically, the scaling term is unknown and therefore a nuisance parameter). When the scaling term is estimated based on the data, the test statistic—under certain conditions—follows a Student's t distribution. The t-test's most common application is to test whether the means of two populations are different. History William Sealy Gosset, who developed the \"t-statistic\" and published it under the pseudonym of \"Student\" The term \"t-statistic\" is abbreviated from \"hypothesis test statistic\".[1] In statistics, the t-distribution was first derived as a posterior distribution in 1876 by Helmert[2][3][4] and Lüroth.[5][6][7] The t-distribution also appeared in a more general form as Pearson Type IV distribution in Karl Pearson's 1895 paper.[8] However, the T-Distribution, also known as Student's t-distribution, gets its name from William Sealy Gosset who first published it in English in 1908 in the scientific journal Biometrika using the pseudonym \"Student\"[9][10] because his employer preferred staff to use pen names when publishing scientific papers.[11] Gosset worked at the Guinness Brewery in Dublin, Ireland, and was interested in the problems of small samples – for example, the chemical properties of barley with small sample sizes. Hence a second version of the etymology of the term Student is that Guinness did not want their competitors to know that they were using the t-test to determine the quality of raw material (see Student's t-distribution for a detailed history of this pseudonym, which is not to be confused with the literal term student). Although it was William Gosset after whom the term Student is penned, it was actually through the work of Ronald Fisher that the distribution became well known as. The cell nuclei contain the genetic material chromatin (red). The proteins making up the cells cytoskeleton have been stained with different colors: actin is blue and microtubules are yellow. DR Torsten Wittmann/Science Photo Library/Getty Image Updated on July 10, 2019 Biology is a wondrous science that inspires us to discover more about the world around us. While science may not have the answers to every question, some biology questions are answerable. Have you ever wondered why DNA is twisted or why some sounds make your skin crawl? Discover answers to these and other intriguing biology questions. Why Is DNA Twisted? DNA is known for its familiar twisted shape. This shape is often described as a spiral staircase or twisted ladder. DNA is a nucleic acid with three main components: nitrogenous bases, deoxyribose sugars, and phosphate molecules. Interactions between water and the molecules that compose DNA cause this nucleic acid to take on a twisted shape. This shape aids in the packing of DNA into chromatin fibers, which condense to form chromosomes. The helical shape of DNA also makes DNA replication and protein synthesis possible. When necessary, the double helix unwinds and opens to allow DNA to be copied. Why Do Certain Sounds Make Your Skin Crawl? Nails scraping against a chalkboard Nails scraping against a chalkboard is one of ten most hated sounds. Tamara Staples/Stone/Getty Images Nails on a chalkboard, squealing brakes, or a crying baby are all sounds that can make one's skin crawl. Why does this happen? The answer involves how the brain processes sound. When we detect a sound, sound waves travel to our ears and the sound energy is converted to nerve impulses. These impulses travel to the auditory cortex of the brain's temporal lobes for processing. Another brain structure, the amygdala, heightens our perception of the sound and associates it with a particular emotion, such as fear or unpleasantness. These emotions can elicit a physical response to certain sounds, such as goose bumps or a sensation that something is crawling over your skin. What Are the Differences Between Eukaryotic and Prokaryotic Cells? The primary characteristic that differentiates eukaryotic cells from prokaryotic cells is the cell nucleus. Eukaryotic cells have a nucleus that is surrounded by a membrane, which separates the DNA within from the cytoplasm and other organelles. Prokaryotic cells do not have a true nucleus in that the nucleus is not surrounded by a membrane. Prokaryotic DNA is located in an area of the cytoplasm called the nucleoid region. Prokaryotic cells are typically much smaller and less complex than eukaryotic cells. Examples of eukaryotic organisms include animals, plants, fungi and protists (ex. algae)."
  
  var allContent: string = ""
  
  # @["./trainDat/miniQa0.txt", "./trainDat/declQuestion.txt", "./trainDat/dmAiMini0.txt",   "./trainDat/qa110.txt"]   300 neurons looks good
  # @["./trainDat/miniQa0.txt", "./trainDat/declQuestion.txt", "./trainDat/dmAiMini0.txt",   "./trainDat/qa110.txt", "./trainDat/trivia700.txt"]   300 neurons looks not so good

  for iFilePath in @["./trainDat/miniQa0.txt", "./trainDat/declQuestion.txt", "./trainDat/dmAiMini0.txt",   "./trainDat/qa110.txt", "./trainDat/trivia700.txt", "./trainDat/trivia250.txt"]:#, "./trainDat/llm0.txt", "./trainDat/wikiAnimalsSmall.txt","./trainDat/tree0.txt","./trainDat/wireheading0.txt"]:
    let fileContent: string = readFile(iFilePath)
    allContent=allContent & fileContent

  var a: seq[int] = convStrToTokens(allContent)


  let lenBefore: int64 = a.len


  for it in 0..4000:
    if not tryCompressBpe(a, bpeCtx):
      break

  let lenAfter: int64 = a.len
  echo(&"ratio={float(lenBefore)/float(lenAfter)}")


  echo(a) # debug complete sequence for debugging

  trainingTokens = a




var embeddingLen: int = 10

var embeddingByToken: seq[ seq[float] ] = @[]

for i in 0..bpeCtx.tokenIdCnt-1:
  let embedding: seq[float] = genRngVec(embeddingLen, 0.1)
  embeddingByToken.add(embedding)



# RNN which implements reservour
type
  Rnn0Ref* = ref Rnn0
  Rnn0* = object
    state*: seq[float]

    units: seq[WeightsAndBias1]
  

proc makeRnn0(stimulusLen: int, weightScale: float, biasScale: float): Rnn0Ref =
  var rnn: Rnn0Ref = Rnn0Ref(state: genNullVec(stimulusLen), units: @[])
  
  var wPlusBias: seq[WeightsAndBias1] = @[]
  # fill weights+biases of neurons
  for i in 0..stimulusLen-1:
    rnn.units.add( WeightsAndBias1(w: genRngVec(stimulusLen, weightScale), b: (rng.rand(2.0)-1.0) * biasScale) )
  
  return rnn

proc computeNextState(rnn: Rnn0Ref, stimulus: seq[float]) =

  let stimulus2: seq[float] = vecAdd(stimulus, rnn.state)
  
  var y: seq[float] = @[]
  
  for i in 0..stimulus2.len-1:
    var a0: float = dot(stimulus2, rnn.units[i].w) + rnn.units[i].b
    var y0: float = actFn2(a0)
    y.add(y0)

  #echo("")
  #echo("")
  #echo(convToPythonArrStr(y))

  rnn.state = y # new state of reservour is y of network







# feed forward network
type Layer1* = object
  units*: seq[WeightsAndBias1]
  
  # 0 : linear
  # 1 : tanh
  # 2 : cos(x)*x
  actFnType: int

proc ffnnCalcLayerOut(layer: var Layer1, x: seq[float]): seq[float] =
  var y: seq[float]
  
  for iUnit in layer.units:
    var a0: float = dot(x, iUnit.w) + iUnit.b
    var y0: float = 0.0
    if layer.actFnType == 0:
      y0 = a0
    elif layer.actFnType == 1:
      y0 = actFn1(a0)
    else:
      y0 = actFn2(a0)
    
    y.add(y0)
  
  return y
    


type Network1* = object
  layers: seq[Layer1]






type TrainingTuple2* = object
  x*: seq[float]
  y*: seq[float]





import std/json
import os
import std/strutils

if isMainModule:
  # 0: generate training data to train NN
  # 1: do inference by loading NN weights+biases
  var programMode: int = parseInt(paramStr(1))



  var stimulusLen: int = 60 #30 # this is the size of the vector of the that of the reservour

  var weightScale: float = 0.3
  var biasScale: float = 0.1

  var rnn: Rnn0Ref = makeRnn0(stimulusLen, weightScale, biasScale)
  


  var ffnn: Network1 # feed forward NN
  
  if programMode==1:
    # load model which is trained from JSON
    echo("load ff-nn weights...")
    block:
      let f = open("model0.json", fmRead)
      let fileContent: string = readAll(f)
      f.close()
      let jsonNode = parseJson(fileContent)
      
      # helper to read a layer from json to units stored as WeightsAndBias1
      proc readUnitsOfLayer(jsonNode: JsonNode): seq[WeightsAndBias1] =
        var units: seq[WeightsAndBias1] = @[]

        for iNeuronJsonObj in jsonNode["neurons"]:
          #echo(iNeuronJsonObj.kind) # DBG

          var w: seq[float] = @[]
          for iJsonObj in iNeuronJsonObj["w"]:
            w.add(iJsonObj.getFloat())
          let bias: float = iNeuronJsonObj["b"].getFloat()
          
          var wAndB: WeightsAndBias1 = WeightsAndBias1(w: w, b: bias)
          units.add(wAndB)
        
        return units
      
      let unitsLayer0: seq[WeightsAndBias1] = readUnitsOfLayer(jsonNode["layers"][0])
      let unitsLayer1: seq[WeightsAndBias1] = readUnitsOfLayer(jsonNode["layers"][1])
      
      ffnn.layers.add(Layer1(units: unitsLayer0, actFnType: 1))
      ffnn.layers.add(Layer1(units: unitsLayer1, actFnType: 0))
    
    echo("...done")

    echo(&"INFO: ffnn has n={ffnn.layers[0].units.len} hidden units")
    
    echo("")
  


  var readoffTokens: seq[int] # tokens to be read of
  var readoffTokenIdx: int = 0
  
  if programMode == 0: # generate training data
    readoffTokens = trainingTokens
  elif programMode == 1: # inference for start-tokens
    readoffTokens = @[]
    readoffTokens.add(trainingTokens[50])
    readoffTokens.add(trainingTokens[51])
    readoffTokens.add(trainingTokens[52])
    readoffTokens.add(trainingTokens[53])
    readoffTokens.add(trainingTokens[54])
    readoffTokens.add(trainingTokens[55])

  # debug all tokens
  block:
    for iToken in readoffTokens:
      echo(retStrByToken(bpeCtx, iToken)&"~")
  
  var inferenceLastToken: int = -1
  var inferencePredictedTokensCnt: int = 0

  
  var trainingTuples: seq[TrainingTuple2] = @[]

  while true:
    var currentToken: int
    if readoffTokenIdx < readoffTokens.len:
      currentToken = readoffTokens[readoffTokenIdx]

    if programMode == 1 and readoffTokenIdx >= readoffTokens.len:
      currentToken = inferenceLastToken
      inferencePredictedTokensCnt+=1
    
    
    var nextToken: int = -1
    if readoffTokenIdx+1<readoffTokens.len:
      nextToken = readoffTokens[readoffTokenIdx+1]

    readoffTokenIdx+=1

    if programMode == 1 and inferencePredictedTokensCnt >= 80:
      break



    if programMode == 0 and nextToken==(-1): # if in training mode and next token is not available
      break # then conclude training
    




    var oldState: seq[float] = rnn.state # store state


    var stimulus: seq[float] = genNullVec(stimulusLen)

    # read out token
    block:
      let embeddingOfCurrentToken: seq[float] = embeddingByToken[currentToken]
      
      # copy to stimulus
      for idx in 0..embeddingOfCurrentToken.len-1:
        stimulus[idx] = embeddingOfCurrentToken[idx]
    




    # compute next state of RNN
    rnn.computeNextState(stimulus)
    


    # append old state and stimulus
    var ffnnStimulus: seq[float] = oldState & stimulus
    block:      
      let mean: float = 0.0#0.006213205773880162#0.01882102070790212
      let stdDev: float = 1.0#0.2349989409320086#1.0#0.1275974493902191
      ffnnStimulus = normalizeHelper(ffnnStimulus, mean, stdDev)

    if programMode==0: # if it is training
      #let embeddingOfNextToken: seq[float] = embeddingByToken[nextToken]

      var nextTokenProbDistrArr: seq[float] = vecMakeVal(1.0e-4, bpeCtx.tokenIdCnt)
      nextTokenProbDistrArr[nextToken] = 0.8

      #nextTokenProbDistrArr = zNormalize(nextTokenProbDistrArr) # commented because not necessary or even something which is good

      # print as python tuple
      #trainingDatContent = trainingDatContent & (&"({convToPythonArrStr(ffnnStimulus)},{convToPythonArrStr(embeddingOfNextToken)}),\n")
      #trainingDatContent = trainingDatContent & (&"({convToPythonArrStr(ffnnStimulus)},{convToPythonArrStr(nextTokenProbDistrArr)}),\n")
      trainingTuples.add( TrainingTuple2(x: ffnnStimulus, y: nextTokenProbDistrArr) )

    elif programMode==1: # if it is in inference mode
      
      # feed state of RNN into FF-NN to compute the embedding of the predicated token
      var predictedFfnnY: seq[float]
      block:
        #var ffnnStimulus: seq[float] = oldState & stimulus

        #echo(&"ffnnStimulus={ffnnStimulus}") # DBG

        let yLayer0: seq[float] = ffnnCalcLayerOut(ffnn.layers[0], ffnnStimulus)
        let yLayer1: seq[float] = ffnnCalcLayerOut(ffnn.layers[1], yLayer0)

        predictedFfnnY = yLayer1
      
      #echo(&"ffnn y={predictedFfnnY}") # DBG

      # now we got the embedding of the predicted token, but we need to map it to the most likely embedding, we do this by cosine-distance
      var predictedToken: int = -1
      block:
        var bestProb: float = -1e10
        #var bestSim: float = -1.0
        var bestTokenIdx: int = -1
        
        for iIdx in 0..predictedFfnnY.len-1:
          let iProb: float = predictedFfnnY[iIdx]
          if iProb > bestProb:
            bestProb = iProb
            bestTokenIdx = iIdx


        predictedToken = bestTokenIdx
        echo(&"best tokenIdx by embedding: idx={bestTokenIdx} bestProb={bestProb}")
        echo(&"predicted token text={retStrByToken(bpeCtx, bestTokenIdx)}~")
      
      #echo(&"rnn state after stimulus is: state={rnn.state}") # DBG

      inferenceLastToken = predictedToken


  if programMode==0: # write trainingdata
    # compute mean of all values
    var mean=0.0
    var n:int = 0
    for iTuple in trainingTuples:
      for iv in iTuple.x:
        mean+=iv
        n+=1
    mean = mean / float(n)

    var stdDev = 0.0
    for iTuple in trainingTuples:
      for iv in iTuple.x:
        stdDev += pow((iv - mean), 2)
    stdDev = sqrt(stdDev / float(n))

    echo(&"mean={mean}")
    echo(&"stdDev={stdDev}")


    for iTupleIdx in 0..trainingTuples.len-1:
      for iIdx in 0..trainingTuples[iTupleIdx].x.len-1:
        trainingTuples[iTupleIdx].x[iIdx] = (trainingTuples[iTupleIdx].x[iIdx]-mean) / stdDev


    
    let filepath: string = "dat055609.sql"
    let f = open(filepath, fmWrite)

    var trainingDatContent: string = "" # file content of generated training data
    #trainingDatContent = trainingDatContent & "trainingTuples1Db=[\n"
    trainingDatContent=trainingDatContent&("CREATE TABLE IF NOT EXISTS a (\nid INTEGER PRIMARY KEY     AUTOINCREMENT,\na           TEXT    NOT NULL,\nb           TEXT    NOT NULL\n);")
    trainingDatContent=trainingDatContent&("INSERT INTO a (a,b) VALUES ")


    var cnt:int = 0

    f.write(trainingDatContent)

    for iTuple in trainingTuples:
      #trainingDatContent = trainingDatContent & (&"({convToPythonArrStr(iTuple.x)},{convToPythonArrStr(iTuple.y)}),\n")

      #trainingDatContent = trainingDatContent & (&"('{convToStrArrStr(iTuple.x)}','{convToStrArrStr(iTuple.y)}')\n")
      f.write(&"('{convToStrArrStr(iTuple.x)}','{convToStrArrStr(iTuple.y)}')\n")
      if cnt<trainingTuples.len-1:
        #trainingDatContent = trainingDatContent & (",\n")
        f.write(",\n")
      cnt+=1

    #trainingDatContent = trainingDatContent & ";\n"
    f.write(";\n")
    #writeFile("dat055609.sql", trainingDatContent)

    f.close()


    echo(&"wrote tuples n={trainingTuples.len}")



#
# inToken -> system -> nextToken
#
# system cycle:
# a)
#
#  inToken   ----> + -> actFn() ---> rnnState
#                  ^                     
#                  |                     
#  rnnState  ------+---------------------------> FF-NN --> nextToken
#                                                  ^
#                                               inToken
# b)
#  then we take nextToken as inToken for the next step!







# DONE < generate distribution of tokens as output instead of embedding of next token > 

# TODO< normalize input to NN in training and inference >
