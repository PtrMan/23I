# base code for "Byte-pair encoding"

import strformat
import tables

# this points to two parent tokens or none if it is a character
type TokenIndirection* = object
  parents*: seq[int]
  isSpecial: bool

type
  BpeCtx* = object
    tokenIdCnt*: int # counter for unused token id number

    tokenMap*: seq[TokenIndirection]


# helper to get string of token
proc retStrByToken*(ctx: BpeCtx, tokenId: int): string =
  let selTokenIndirection: TokenIndirection = ctx.tokenMap[tokenId]
  if selTokenIndirection.parents.len == 0: # do no parents exist? then it is a char
    let c: char = char(tokenId)
    let s: string = ""&c
    return s
  else:
    return retStrByToken(ctx, selTokenIndirection.parents[0]) & retStrByToken(ctx, selTokenIndirection.parents[1])

proc convStrToTokens*(str: string): seq[int] =
  var a: seq[int] = @[]
  for ichar in str:
    let ascii: int = int(ichar)
    a.add(ascii)
  return a


# tries to find a better more compressed token-stream by using a BPE'ish scheme
proc tryCompressBpe*(a: var seq[int], bpeCtx: var BpeCtx): bool =
  #echo(&"{a}") #DBG
  
  # helper to convert 
  
  var cnt = initTable[tuple[a:int,b:int], int64]()
  for iidx in 0..a.len-1-1:
    let subseq: tuple[a:int,b:int] = (a[iidx],a[iidx+1])


    let specialTokensChars = @[':', ';', '?', '!', '.', ',', '\'', '"', '+', '-', '*', '/', '#', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    var specialTokens: seq[int] = @[10]

    if true:
      for iChar in specialTokensChars:
        specialTokens.add(int(char(iChar)))

    if (subseq.a in specialTokens) or (subseq.b in specialTokens):
      continue


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

  echo(&"({highestKey.a},{highestKey.b}) <- {bpeCtx.tokenIdCnt}")
  bpeCtx.tokenMap.add(TokenIndirection(parents: @[highestKey.a, highestKey.b])) # store indirection of new token which "points" at children token
  
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
