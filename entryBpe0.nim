# program to read file and convert it to BPE encoded tokens

import std/strformat
import std/strutils # for tokenizer
import std/sequtils


import bpeBase0

var trainingTokens: seq[int] = @[]

var bpeCtx: BpeCtx = BpeCtx(tokenIdCnt:256, tokenMap: @[])
for i in 0..bpeCtx.tokenIdCnt-1:
  bpeCtx.tokenMap.add(TokenIndirection(parents: @[])) # add "root" token

block: # convert text to BPE
  #var str: string = "A t-test is any statistical hypothesis test in which the test statistic follows a Student's t-distribution under the null hypothesis. It is most commonly applied when the test statistic would follow a normal distribution if the value of a scaling term in the test statistic were known (typically, the scaling term is unknown and therefore a nuisance parameter). When the scaling term is estimated based on the data, the test statistic—under certain conditions—follows a Student's t distribution. The t-test's most common application is to test whether the means of two populations are different. History William Sealy Gosset, who developed the \"t-statistic\" and published it under the pseudonym of \"Student\" The term \"t-statistic\" is abbreviated from \"hypothesis test statistic\".[1] In statistics, the t-distribution was first derived as a posterior distribution in 1876 by Helmert[2][3][4] and Lüroth.[5][6][7] The t-distribution also appeared in a more general form as Pearson Type IV distribution in Karl Pearson's 1895 paper.[8] However, the T-Distribution, also known as Student's t-distribution, gets its name from William Sealy Gosset who first published it in English in 1908 in the scientific journal Biometrika using the pseudonym \"Student\"[9][10] because his employer preferred staff to use pen names when publishing scientific papers.[11] Gosset worked at the Guinness Brewery in Dublin, Ireland, and was interested in the problems of small samples – for example, the chemical properties of barley with small sample sizes. Hence a second version of the etymology of the term Student is that Guinness did not want their competitors to know that they were using the t-test to determine the quality of raw material (see Student's t-distribution for a detailed history of this pseudonym, which is not to be confused with the literal term student). Although it was William Gosset after whom the term Student is penned, it was actually through the work of Ronald Fisher that the distribution became well known as. The cell nuclei contain the genetic material chromatin (red). The proteins making up the cells cytoskeleton have been stained with different colors: actin is blue and microtubules are yellow. DR Torsten Wittmann/Science Photo Library/Getty Image Updated on July 10, 2019 Biology is a wondrous science that inspires us to discover more about the world around us. While science may not have the answers to every question, some biology questions are answerable. Have you ever wondered why DNA is twisted or why some sounds make your skin crawl? Discover answers to these and other intriguing biology questions. Why Is DNA Twisted? DNA is known for its familiar twisted shape. This shape is often described as a spiral staircase or twisted ladder. DNA is a nucleic acid with three main components: nitrogenous bases, deoxyribose sugars, and phosphate molecules. Interactions between water and the molecules that compose DNA cause this nucleic acid to take on a twisted shape. This shape aids in the packing of DNA into chromatin fibers, which condense to form chromosomes. The helical shape of DNA also makes DNA replication and protein synthesis possible. When necessary, the double helix unwinds and opens to allow DNA to be copied. Why Do Certain Sounds Make Your Skin Crawl? Nails scraping against a chalkboard Nails scraping against a chalkboard is one of ten most hated sounds. Tamara Staples/Stone/Getty Images Nails on a chalkboard, squealing brakes, or a crying baby are all sounds that can make one's skin crawl. Why does this happen? The answer involves how the brain processes sound. When we detect a sound, sound waves travel to our ears and the sound energy is converted to nerve impulses. These impulses travel to the auditory cortex of the brain's temporal lobes for processing. Another brain structure, the amygdala, heightens our perception of the sound and associates it with a particular emotion, such as fear or unpleasantness. These emotions can elicit a physical response to certain sounds, such as goose bumps or a sensation that something is crawling over your skin. What Are the Differences Between Eukaryotic and Prokaryotic Cells? The primary characteristic that differentiates eukaryotic cells from prokaryotic cells is the cell nucleus. Eukaryotic cells have a nucleus that is surrounded by a membrane, which separates the DNA within from the cytoplasm and other organelles. Prokaryotic cells do not have a true nucleus in that the nucleus is not surrounded by a membrane. Prokaryotic DNA is located in an area of the cytoplasm called the nucleoid region. Prokaryotic cells are typically much smaller and less complex than eukaryotic cells. Examples of eukaryotic organisms include animals, plants, fungi and protists (ex. algae)."
  
  var allContent: string = ""
  
  # @["./trainDat/miniQa0.txt", "./trainDat/declQuestion.txt", "./trainDat/dmAiMini0.txt",   "./trainDat/qa110.txt"]   300 neurons looks good
  # @["./trainDat/miniQa0.txt", "./trainDat/declQuestion.txt", "./trainDat/dmAiMini0.txt",   "./trainDat/qa110.txt", "./trainDat/trivia700.txt"]   60 neurons looks 4 in range 0 to 10

  var filePaths: seq[string] = @[]
  
  #filePaths = @["./trainDat/simple0.txt"]
  #filePaths = @["./trainDat/miniQa0.txt"]
  filePaths.add("./trainDat/miniQa0.txt")
  filePaths.add("./trainDat/trivia700.txt")
  filePaths.add("./trainDat/wikiUseful0.txt")
  filePaths.add("./trainDat/wiki2.txt")
  filePaths.add("./trainDat/csound0.txt")
  

  #filePaths.add("./trainDat/qaNew0.txt")
  

  #filePaths = @["./trainDat/miniQa0.txt", "./trainDat/declQuestion.txt", "./trainDat/dmAiMini0.txt",   "./trainDat/qa110.txt", "./trainDat/trivia700.txt", "./trainDat/trivia250.txt"]#, "./trainDat/llm0.txt", "./trainDat/wikiAnimalsSmall.txt","./trainDat/tree0.txt","./trainDat/wireheading0.txt"]

  for iFilePath in filePaths:
    let fileContent: string = readFile(iFilePath)
    allContent=allContent & fileContent

  var a: seq[int] = convStrToTokens(allContent)


  let lenBefore: int64 = a.len


  for it in 0..4000:
    if not tryCompressBpe(a, bpeCtx):
      break

  let lenAfter: int64 = a.len
  echo(&"ratio={float(lenBefore)/float(lenAfter)}")


  #echo(a) # debug complete sequence for debugging

  trainingTokens = a

# print result
block:
  let f2 = open("outTokens0.txt", fmWrite) # fmWrite is the file mode constant
  defer: f2.close()

  #for line in lines:
  #  f.writeLine(line)
  
  var idx = 0
  for z in trainingTokens:
    f2.write(z)    
    if idx < trainingTokens.len-1:
      f2.write(", ")

    idx+=1

  f2.write("\n") # end line

# store table of tokens to map from token to text and from text to tokens
block:


  


  # functionality to convert a string of a token to a representation which can get serialized as a text file without worrying about escaping/etc.
  func convTokenTextToStrEncoding(strToEncode: string): string =
    #let strToEncode: string = "\r\nT" # string of the token which should get encoded

    let hexStrArr: seq[string] = map(strToEncode, (proc (z:char): string = (""&z).toHex())) # convert to array of hex strings
    let z0: string = join(hexStrArr, " ")
    return z0





  #
  let f2 = open("outTokenInfo.txt", fmWrite)
  defer: f2.close()

  for iTokenId in 0..<bpeCtx.tokenIdCnt:
    let str: string = retStrByToken(bpeCtx, iTokenId)    
    let strAsEncoding: string = convTokenTextToStrEncoding(str)
    let z: string = &"{iTokenId} {strAsEncoding}"
    f2.write(&"{z}\n")



block:
  # PROTO of tokenizer
  
  var tokensByIdx: seq[string] = @[] # string of tokens by index, used to search for longest string
  for iTokenId in 0..<bpeCtx.tokenIdCnt:
    let str: string = retStrByToken(bpeCtx, iTokenId)
    #echo(str) # DBG
    tokensByIdx.add(str)


  echo("")

  # code to compute tokens of given string
  var rem: string
  
  rem = "q% Compute 7+7 r%"
  #rem = "q% What is 4+5? r% "

  while rem.len > 0:

    # try to find longest matching token
    var longestTokenStr: string = ""
    var longestTokenIdx: int = -1

    for iIdx in 0..<tokensByIdx.len:
      let iTokenStr: string = tokensByIdx[iIdx]
      if rem.startsWith(iTokenStr) and iTokenStr.len > longestTokenStr.len:
        longestTokenStr = iTokenStr
        longestTokenIdx = iIdx
    
    # best token found

    echo(longestTokenIdx)

    rem = rem[longestTokenStr.len..rem.len-1] # cut away

block:
  # PROTO of conversion of tokens to text

  var tokensToConvertToText: seq[int] = @[321, 55, 43, 55, 265, 37, 97, 63, 34, 52, 282, 48, 50, 53, 46]
  
  var outTxt: string = ""
  for iTokenId in tokensToConvertToText:
    let str0: string = retStrByToken(bpeCtx, iTokenId)
    outTxt = outTxt & str0
  
  echo(outTxt)


