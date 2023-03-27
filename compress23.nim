import tables
import std/strutils
import std/strformat

# compression experiment
# version 2 from 27.03.2023

type Sym0TypeEnum* = enum
  name, binary

# symbol which can be compressed
type
  Sym0Ref* = ref Sym0
  Sym0* = object
    case `type`: Sym0TypeEnum
    of name:
      name*: string # usually single char
    of binary:
      l*: Sym0Ref
      r*: Sym0Ref

func convToStr(v: Sym0Ref): string =
  case v.`type`
  of name:
    return v.name
  of binary:
    #return "b("&convToStr(v.l)&" , "&convToStr(v.r)&")"
    return convToStr(v.l)&convToStr(v.r)

func eq(a: Sym0Ref, b: Sym0Ref): bool =
  case a.`type`
  of name:
    case b.`type`
    of name:
      return a.name == b.name
    of binary:
      return false
  of binary:
    case b.`type`
    of binary:
      return eq(a.l,b.l) and eq(a.r,b.r)
    of name:
      return false

func calcHash(sym: Sym0Ref): int64 =
  case sym.`type`
  of name:
    var res: int64 = 0
    for c in sym.name:
      res = (res mod (0x100000000)) * 10 + c.int - '0'.int
    return res
  of binary:
    return 23*(calcHash(sym.l) mod (0x100000000))+53*(calcHash(sym.r) mod (0x100000000))



type
  Sym0WithCntRef* = ref Sym0WithCnt
  Sym0WithCnt* = object
    sym*: Sym0Ref
    cnt*: int64


var symbols: seq[Sym0Ref] = @[]

var inStr: string = ""

block:
  let fileContent = readFile("text.txt")
  inStr = inStr&fileContent

inStr = inStr.toLowerAscii()

# convert raw string into symbols
for iChar in inStr:
  let iChar2: string = ""&iChar
  #echo(iChar2)

  symbols.add(Sym0Ref(`type`:name, name:iChar2))


# count how many occurrences occur of combinations
# and remove highest one
while true:
  # symbol+count by hash
  var symsAndCountByHash: Table[int64, seq[Sym0WithCntRef]] = initTable[int64, seq[Sym0WithCntRef]]()
  
  for iIdx in 0..symbols.len-1-1:
    let l = symbols[iIdx]
    let r = symbols[iIdx+1]
    let candidateSym = Sym0Ref(`type`:binary, l:l, r:r)
    let candidateHash: int64 = calcHash(candidateSym)
    #echo(candidateHash)

    if candidateHash in symsAndCountByHash:
      var found = false
      for iV in symsAndCountByHash[candidateHash]:
        if found:
          continue

        if eq(iV.sym, candidateSym):
          iV.cnt += 1 # bump count
    else:
      symsAndCountByHash[candidateHash] = @[Sym0WithCntRef(sym:candidateSym,cnt:1)]

  
  # * fold the one with the highest occurence count to reduce numbers of bits!
  var highestCnt: int64 = 0
  var highestSym: Sym0Ref
  for iKey, iValue in symsAndCountByHash:
    for iValue2 in iValue:
      if iValue2.cnt > highestCnt:
        highestCnt = iValue2.cnt
        highestSym = iValue2.sym

  if highestCnt <= 1:
    break # not worth it
  
  echo(&"highestCnt={highestCnt}")
  
  # search and replace

  echo ""
  echo symbols.len

  var idx: int = symbols.len-1-1
  while idx >= 0:
    if idx+1 <= symbols.len-1: # avoid out of bounds access
      let l = symbols[idx]
      let r = symbols[idx+1]
      let candidateSym = Sym0Ref(`type`:binary, l:l, r:r)
      if eq(candidateSym, highestSym):
        # replace
        symbols[idx] = highestSym
        symbols.delete(idx+1)
        idx += 1 # keep on same index for next iteration
  
    idx -= 1
  
  echo symbols.len




# build symbolic predictive model!

# TODO

# TODO< also introduce variables for simple Q&A! >



# debug compressed representation as string
for iSym in symbols:
  echo("-") 
  echo( convToStr(iSym) )
  


