# experiment for training of autorecurrent model

#
# using pytorch indirectly by emitting a sqlite *.sql file











# 12.01.2023

import strformat
import tables

var c: string = ""


#c=c&"def x():\n  v0=p0/p1\n  return v0\nRES:\nPUSH EAX  [ESI+0]\n  DIV EAX, [ESI+8]\nRET\n"
#c=c&"def x():\n  v0=p0*p1\n  return v0\nRES:\nPUSH EAX  [ESI+0]\n  MUL EAX, [ESI+8]\nRET\n"
#c=c&"def x():\n  v0=p0+p1\n  return v0\nRES:\nPUSH EAX  [ESI+0]\n  ADD EAX, [ESI+8]\nRET\n"


# generate synthetic training data of addition
#for ia in -25..25:
#  for ib in -25..25:
#    let temp: string = &"{ia}+{ib}=Q#Q\nres={ia}+{ib}\nQ#Qres={ia+ib}\n{ia+ib}"
#    c=c&temp&"\n"

#for ia in -25..25:
#  for ib in -25..25:
#    let temp: string = &"{ia}-{ib}=Q#Q\nres={ia}-{ib}\nQ#Qres={ia+ib}\n{ia-ib}"
#    c=c&temp&"\n"

#for ia in -25..25:
#  for ib in -25..25:
#    let temp: string = &"{ia}*{ib}=Q#Q\nres={ia}*{ib}\nQ#Qres={ia*ib}\n{ia*ib}"
#    c=c&temp&"\n"



for ia in -5..5:
  for ib in -5..5:
    let temp: string = &"{ia}+{ib}=Q#Q\nres={ia}+{ib}\nQ#Qres={ia+ib}\n{ia+ib}"
    c=c&temp&"\n"



var a: seq[int] = @[]
for ichar in c:
  let ascii: int = int(ichar)
  a.add(ascii)



type
  BpeCtx* = object
    tokenIdCnt: int # counter for unused token id number



# tries to find a better more compressed token-stream by using a BPE'ish scheme
proc tryCompressBpe(bpeCtx: var BpeCtx): bool =
  #echo(&"{a}") #DBG
  
  # helper to convert 
  
  var cnt = initTable[tuple[a:int,b:int], int64]()
  for iidx in 0..a.len-1-1:
    let subseq: tuple[a:int,b:int] = (a[iidx],a[iidx+1])
    if not (subseq in cnt):
      cnt[subseq] = 0
    cnt[subseq] += 1  
  
  var highestKey: tuple[a:int,b:int] = (-1,-1)
  var highestCnt: int64 = 0
  for k, v in cnt.mpairs:
    if v > highestCnt:
      highestCnt = v
      highestKey = k

  if highestCnt <= 1:
    return false # not worth it

  
  echo(&"{highestKey} cnt={highestCnt}")
  
  # replace all matches by new id
  var iidx = a.len-2
  while iidx >= 0:
    let v0: int = a[iidx]
    let v1: int = a[iidx+1]
  
    if v0 == highestKey.a and v1 == highestKey.b:
      a[iidx] = bpeCtx.tokenIdCnt
      a.delete(iidx+1)
      iidx-=1 # optimization
  
    iidx-=1

  bpeCtx.tokenIdCnt+=1 # generate new token id
  
  #echo(&"a={a}") # detailed DBG
  echo(&"len={a.len}")


  return true

let lenBefore: int64 = a.len

var bpeCtx: BpeCtx = BpeCtx(tokenIdCnt:256)

for it in 0..5000:
  if not tryCompressBpe(bpeCtx):
    break

let lenAfter: int64 = a.len
echo(&"ratio={float(lenBefore)/float(lenAfter)}")























# abstract away the vector of a embedding which encodes a symbol
type
  EmbeddingVec* = object
    v: seq[float64]

import protoBp1

import sequtils
import std/strformat

func join(l: seq[string], sep: char): string =
  var res: string = ""
  for iv in l[0..l.len-1-1]:
    res = res&iv&sep
  res = res&l[l.len-1]
  return res
    


when isMainModule:

  echo("start program")

  import std/random
  #import std/tables

  # implementation of tokenizer

  var tokenMax: int = 0
  var mapWordToTokenId = initTable[string, int]()








  var rng0: Rand = initRand(345) # rng to generate vectors




  var nEmbeddings: int = 5000 # number of embeddings
  # TODO< compute that from training data >

  var embeddingVecs: seq[EmbeddingVec] = @[]
  for i in 1..nEmbeddings:
      embeddingVecs.add(EmbeddingVec(v:vecGenRng(rng0, 5)))


  # fill training data
  var gen: DatGenerator = DatGenerator(rng:initRand(345))


  var tokens: seq[int] = @[]
  #tokens = tokens0
  tokens = a

  # call(0)+call(0)=call(0)
  #tokens = tokens & @[1, 2, 3, 2, 4,  5,  1, 2, 3, 2, 4,  6,  1, 2, 3, 2, 4]


  # iterator to slice the tokens into slices for training
  iterator genSlices(tokens: seq[int], sliceLen:int): seq[int] =
    for iEndIdx in sliceLen..tokens.len-1-1:
      let slice0: seq[int] = tokens[iEndIdx-sliceLen..iEndIdx-1]
      yield slice0




  # iterate over slices of training data and convert them to stimulus-target tuples
  #for slice0 in @[ @[0, 2, 2, 4, 5, 1, 6], @[2, 2, 4, 5, 1, 6, 1], @[2, 4, 5, 1, 6, 1, 2] ]:
  for slice0 in genSlices(tokens, 29+1):
      echo slice0

      if slice0.len != 29+1:
          # raise exception because we have the wrong length of the slice!
          echo slice0.len
          raise newException(Exception,fmt"expected slice len of {29+1}")

      let sliceBeforePredicted: seq[int] = slice0[0..^2]
      let predicted: seq[int] = @[slice0[^1]]

      
      var stimulus: seq[float64] = @[]
      for iv in sliceBeforePredicted:
          stimulus = stimulus & embeddingVecs[iv].v
      var target: seq[float64] = @[]
      target = target & embeddingVecs[ predicted[0] ].v
      
      gen.dat.add((@[stimulus], target, 1.0))
  

  for iDat in gen.dat:
    var stimulusStr: string = iDat.inArrays[0].map(proc(iv: float64): string = $iv).join(';')
    var targetStr: string = iDat.target.map(proc(iv: float64): string = $iv).join(';')

    #var stimulusStr: string = &"{iDat.inArrays[0]}"
    #var targetStr: string = &"{iDat.target}"
    echo(&"INSERT INTO a (a,b) VALUES ('{stimulusStr}','{targetStr}');")








