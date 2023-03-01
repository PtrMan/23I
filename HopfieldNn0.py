import torch

# transpose helper
def transpose2(m):
    return torch.transpose(m, 0, 1)


#   t000 = x * w1 # map actual input to vector which encodes the combination of attended features
#   
#   t001 = hopfield(mask(t000), w0) # apply hopfied to it
#   
#   y = sigmoid(t001, w2, w3)
#   y = sigmoid(t001+x, w2, w3)

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
device = torch.device(dev)

class Model0(torch.nn.Module):
    @staticmethod
    def _hopfieldCalc(phi, xMat):
        
        phi2 = transpose2(phi)
        
        beta = 1.0
        

        t0 = beta*(transpose2(xMat)@phi2)
        t1 = torch.softmax(t0, 0) # compute softmax of hopfield NN
        phi2 = xMat@t1
        
        return transpose2(phi2)
    
    def __init__(self):
        global device

        super().__init__()
        
        self.verbosity = 0
        
        self.widthLatent = 20 # how big is the latent space vector?
        
        torch.manual_seed(443)
        
        xSize = (15)*5 # size of stimulus X of the NN
        
        # weights to convert stimuli to  stimuli fed into hopfield NN
        self.w1 = torch.nn.Parameter( ((0.01--0.01)*torch.rand(xSize, self.widthLatent*2)+(-0.01)).requires_grad_().to(device) )
        
        
        
        # weights for hopfield NN
        """w0 = torch.tensor([
            [1.00000988, 0.000121, 0.0000111, 0.00007211,   0.00003432, 0.0004532, 0.00004453, 0.000006772],
            [0.00000987, 0.000122, 1.0000112, 0.00007212,   0.00003433, 0.0004533, 0.00004454, 0.000006773],
            [0.00000986, 1.000123, 0.0000113, 0.00007213,   0.00003434, 0.0004534, 0.000044544, 0.000006774],        
        ], requires_grad = True)
        """
        nHopfieldVecs = 60 # how many different vectors does the hopfied NN have? - determines memory capacity of hopfield NN. - is independent on everything else
        w0 = ((0.1--0.1)*torch.rand(self.widthLatent*2, nHopfieldVecs)+(-0.1)).to(device).requires_grad_()
        
        self.w0 = torch.nn.Parameter(w0)
        
        
        nUnitsOutput = 3300 # how many output units does the bottom layer have?
        
        n2 = 120
        
        self.w2 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(self.widthLatent*2 + xSize, n2)+(-1.0)).to(device).requires_grad_() )
        #######self.w2 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(nUnitsOutput, nUnitsOutput)+(-1.0)).to(device).requires_grad_() )
        self.w3 = torch.nn.Parameter( torch.rand(1, nUnitsOutput).requires_grad_().to(device) ) # bias

        
        self.w4 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(n2, nUnitsOutput)+(-1.0)).to(device).requires_grad_() )

        
        #self.p5 = torch.nn.Parameter( ((1.0-(-1.0))*torch.rand(n2, nUnitsOutput)+(-1.0)).to(device).requires_grad_() )

        
    def forward(self, x):
        
        t2 = x.unsqueeze(0) # convert to matrix
        #t2 = t2.requires_grad_()
        #print(t2)
        
        
        t1 = torch.matmul(t2, self.w1)
        #print('<')
        #print(self.w1)
        #print(t2)
        #print(self.w0)
        #print('>')
        
        # mask to mask out value from hopfield attention
        mask0 = [1.0]*self.widthLatent + [0.0]*self.widthLatent
        mask0AsTensor = torch.tensor(mask0).to(device)
        
        #print(t1)
        #print(mask0AsTensor)
        
        t5 = torch.mul(t1, mask0AsTensor.unsqueeze(0))
        #t5 = t1
        #print('hehehhe')
        #print(t5)
        
        # input vector
        # first part is the key, second part is free
        t0 = Model0._hopfieldCalc(t5, self.w0)
        
        #print(t0)
        
        
        #t5 = t0
        #print(t0)
        #print(t2)
        t5 = torch.cat((t0, t2), 1)
        #print(t5)
        
        
        t3 = torch.matmul(t5, self.w2)
        
        
        t6 = torch.special.sinc(t3)
        #print("t6.size="+str(t6.size()))
        
        
        t7 = torch.matmul(t6, self.w4)
        #print("t7.size="+str(t7.size()))
        
        
        
        #t4 = t3 # commented on 28.02.2023
        #t4 = t7
        t4 = torch.sigmoid(t7+self.w3)
        
        #print(str(t4))
        
        #jiijijji()
        
        return t4[0]


    def string(self):
        return f'<NN>'

import random

tokenEmbeddings = []
for i in range(3300):
    tokenEmbeddings.append(torch.rand(1, 5).tolist())


# data generator (used for training)
class DatGen0(object):
    def __init__(self):
        self.seqs = [] # list with all sequences
        
        pass
    
    # returns tuple of (None, list of stimuli tokens, predictedToken)
    # first result is reserved for RNN context vector
    def sample(self):
        selSeqIdx = random.randint(0, len(self.seqs)-1)
        selSeq = self.seqs[selSeqIdx]
                
        sliceLen = 15+1
        
        selStartIdxRangeMin = 0
        selStartIdxRangeMax = len(selSeq)-sliceLen
        #print('len='+str(len(selSeq)))
        #print('endIdx='+str(selStartIdxRangeMax))
        
        selStartIdx = random.randint(selStartIdxRangeMin, selStartIdxRangeMax)
        
        slice_ = selSeq[selStartIdx:selStartIdx+sliceLen]
        #print(slice_)
        
        return (None, slice_[:-1], slice_[-1])
        
datGen = DatGen0()
datGen.seqs.append([1777, 46, 10, 81, 58, 2412, 63, 10, 65, 58, 10, 1777, 46, 10, 10, 2979, 46, 10, 10, 81, 58, 1288, 1308, 63, 10, 65, 58, 10, 2979, 46, 10, 10, 81, 58, 2788, 63, 10, 65, 58, 10, 2239, 392, 2595, 437, 46, 917, 691, 264, 913, 920, 2032, 46, 10, 10, 10, 981, 474, 2230, 46, 10, 68, 111, 308, 2955, 594, 2230, 46, 10, 2178, 1036, 594, 701, 1432, 46, 10, 2097, 1228, 737, 103, 295, 437, 46, 10, 417, 814, 416, 303, 110, 39, 265, 1455, 46, 10, 2371, 46, 10, 81, 58, 1288, 1590, 804, 63, 10, 65, 58, 10, 2371, 46, 10, 81, 58, 2194, 44, 1409, 1324, 46, 10, 65, 58, 10, 2709, 10, 81, 58, 2194, 44, 1409, 2070, 256, 1324, 46, 10, 65, 58, 10, 2709, 10, 81, 58, 795, 2356, 39, 1880, 44, 1409, 1153, 1207, 2070, 101, 46, 10, 65, 58, 10, 881, 290, 2643, 889, 1396, 10, 81, 58, 1606, 559, 610, 44, 2411, 46, 10, 65, 58, 10, 1779, 10, 81, 58, 32, 2597, 648, 63, 10, 65, 58, 10, 71, 1207, 562, 1637, 46, 10, 81, 58, 32, 1779, 44, 2411, 46, 10, 65, 58, 10, 1779, 10, 81, 58, 32, 881, 2941, 889, 2882, 868, 44, 1409, 346, 283, 2450, 46, 10, 65, 58, 10, 34, 1953, 434, 2890, 34, 10, 10, 1767, 562, 804, 46, 10, 2711, 647, 1177, 46, 10, 2820, 487, 2995, 39, 265, 1346, 46, 10, 70, 520, 2275, 669, 926, 46, 10, 87, 2357, 277, 119, 356, 46, 10, 2647, 610, 46, 10, 417, 575, 1333, 3008, 610, 46, 10, 2239, 3032, 46, 10, 981, 922, 1766, 3032, 46, 10, 70, 1215, 2955, 2995, 39, 265, 102, 285, 46, 10, 10, 981, 562, 437, 46, 10, 81, 58, 2412, 63, 10, 65, 58, 10, 1777, 46, 10, 1953, 2266, 46, 10, 2097, 2391, 401, 46, 10, 81, 58, 1875, 551, 2388, 63, 10, 65, 58, 10, 1100, 46, 795, 551, 562, 2391, 339, 315, 736, 477, 2266, 46, 10, 81, 58, 364, 298, 289, 2980, 63, 10, 1100, 46, 10, 10, 10, 81, 58, 2788, 63, 10, 65, 58, 10, 1302, 10, 10, 81, 58, 10, 565, 788, 257, 63, 10, 65, 58, 10, 1767, 1030, 46, 10, 10, 10, 81, 58, 10, 565, 684, 63, 10, 65, 58, 10, 2711, 562, 1177, 46, 10, 10, 10, 81, 58, 10, 73, 1342, 923, 861, 922, 472, 775, 1050, 63, 10, 65, 58, 10, 1100, 10, 10, 81, 58, 546, 1618, 1603, 330, 1408, 339, 1366, 1685, 788, 423, 46, 10, 65, 58, 546, 325, 919, 797, 46, 10, 10, 81, 58, 1186, 2328, 315, 804, 259, 797, 63, 10, 65, 58, 10, 1302, 10, 81, 58, 1186, 804, 656, 2328, 2272, 797, 63, 10, 65, 58, 10, 1302, 10, 81, 58, 364, 1455, 259, 610, 63, 10, 65, 58, 10, 1599, 46, 2079, 46, 10, 81, 58, 32, 1107, 289, 1397, 373, 63, 10, 65, 58, 10, 1767, 434, 106, 1146, 46, 10, 81, 58, 341, 110, 1649, 259, 594, 2255, 1432, 44, 920, 2509, 307, 728, 63, 10, 65, 58, 10, 1599, 46, 2079, 46, 10, 81, 58, 32, 873, 2946, 63, 10, 65, 58, 10, 80, 101, 1077, 256, 392, 564, 602, 46, 10, 81, 58, 1875, 1029, 342, 454, 352, 63, 10, 65, 58, 10, 1599, 46, 1824, 816, 282, 46, 10, 81, 58, 32, 403, 1793, 63, 10, 65, 58, 10, 1793, 747, 374, 1533, 463, 536, 271, 549, 1588, 1282, 46, 10, 81, 58, 1875, 324, 285, 98, 473, 1793, 63, 10, 65, 58, 10, 1302, 46, 10, 10, 81, 58, 1186, 2120, 487, 797, 63, 10, 65, 58, 10, 1100, 46, 10, 10, 81, 58, 478, 111, 2501, 508, 2405, 32, 1637, 63, 10, 65, 58, 10, 1100, 46, 10, 10, 438, 264, 2980, 63, 1384, 44, 750, 508, 653, 39, 1880, 33, 10, 1726, 551, 610, 63, 1384, 44, 750, 2356, 39, 1880, 33, 10, 1726, 1766, 758, 63, 1384, 44, 750, 1766, 3008, 758, 44, 2312, 2954, 2987, 39, 265, 594, 1898, 46, 10, 1726, 663, 758, 63, 1384, 44, 750, 663, 653, 39, 265, 758, 33, 10, 438, 1151, 281, 758, 63, 705, 279, 44, 750, 98, 281, 2226, 1749, 2954, 474, 1898, 46, 10, 10, 69, 2941, 2615, 290, 2968, 716, 46, 2190, 2968, 2822, 2615, 883, 501, 111, 46, 484, 957, 589, 102, 356, 116, 46, 2190, 2643, 589, 1057, 101, 825, 1788, 46, 10, 809, 708, 589, 1179, 1345, 122, 46, 10, 2593, 2298, 46, 10, 2178, 267, 704, 2764, 46, 10, 1468, 968, 559, 2764, 46, 10, 2593, 544, 506, 116, 425, 603, 964, 46, 10, 417, 1600, 920, 3033, 278, 1830, 115, 46, 10, 2647, 1504, 1293, 100, 824, 2206, 2890, 46, 760, 119, 1759, 376, 1598, 264, 67, 1018, 100, 256, 71, 97, 583, 256, 2678, 63, 32, 10, 10, 263, 58, 1165, 44, 1367, 10, 10, 1597, 1276, 1283, 63, 10, 10, 263, 58, 341, 100, 272, 10, 10, 914, 402, 119, 2962, 319, 63, 10, 10, 263, 58, 32, 629, 87, 1549, 2344, 98, 10, 10, 326, 2805, 446, 481, 1364, 1808, 513, 2047, 282, 97, 76, 2897, 319, 10, 10, 263, 58, 515, 101, 1832, 533, 100, 289, 86, 1362, 105, 10, 10, 299, 701, 1934, 328, 2385, 345, 2843, 1826, 2309, 1143, 360, 108, 686, 2392, 339, 1115, 63, 10, 10, 263, 58, 2952, 640, 2807, 10, 10, 1597, 1132, 848, 1101, 63, 10, 10, 263, 58, 1734, 87, 2474, 2976, 10, 10, 1117, 87, 2866, 103, 1592, 65, 3028, 3015, 111, 122, 463, 63, 10, 10, 263, 58, 1881, 447, 1809, 257, 10, 10, 2819, 1329, 70, 304, 343, 904, 87, 477, 949, 63, 10, 10, 263, 58, 493, 2304, 10, 10, 382, 289, 1430, 2232, 44, 2616, 294, 115, 1016, 433, 2951, 112, 296, 324, 63, 10, 10, 263, 58, 895, 2067, 1621, 10, 10, 299, 1727, 1401, 63, 10, 10, 263, 58, 355, 307, 285, 415, 100, 45, 454, 2702, 65, 118, 282, 44, 32, 2814, 10, 10, 2361, 1736, 266, 2921, 321, 666, 464, 63, 10, 10, 263, 58, 2476, 378, 464, 97, 44, 2258, 10, 10, 1013, 481, 1920, 790, 2665, 559, 116, 378, 283, 319, 63, 10, 10, 263, 58, 2287, 2041, 342, 10, 10, 429, 1835, 2653, 2294, 442, 63, 10, 10, 263, 58, 400, 327, 106, 2364, 2360, 97, 44, 478, 618, 765, 10, 10, 1597, 932, 988, 2038, 63, 10, 10, 263, 58, 485, 320, 283, 2078, 80, 592, 843, 279, 10, 10, 1213, 823, 556, 309, 2872, 63, 10, 10, 263, 58, 1918, 66, 765, 107, 275, 44, 355, 2103, 396, 10, 10, 403, 84, 316, 2904, 393, 392, 481, 415, 63, 10, 10, 263, 58, 383, 2442, 44, 1121, 2777, 830, 44, 32, 637, 285, 1522, 10, 10, 1852, 1401, 2100, 1134, 476, 63, 10, 10, 263, 58, 406, 116, 354, 112, 388, 10, 10, 1733, 1214, 367, 1781, 869, 63, 10, 10, 263, 58, 561, 1006, 10, 10, 1792, 394, 324, 736, 1299, 256, 674, 608, 748, 63, 10, 10, 263, 58, 705, 354, 283, 10, 10, 292, 2236, 277, 1060, 1263, 298, 2013, 112, 371, 423, 63, 10, 10, 263, 58, 2282, 607, 100, 1805, 10, 10, 299, 2120, 302, 1269, 1264, 1796, 946, 275, 115, 287, 1455, 63, 10, 10, 263, 58, 389, 1403, 10, 10, 617, 799, 2567, 63, 458, 288, 296, 63, 10, 10, 263, 58, 32, 2032, 46, 341, 399, 2163, 287, 947, 121, 114, 396, 10, 10, 2942, 46, 2093, 2658, 2871, 2809, 700, 63, 10, 10, 263, 58, 389, 2398, 97, 10, 10, 469, 1337, 256, 312, 1391, 116, 273, 1624, 844, 1308, 1568, 111, 441, 63, 10, 10, 263, 58, 836, 789, 2362, 3029, 2773, 401, 10, 10, 856, 2261, 330, 455, 1346, 63, 10, 10, 263, 58, 406, 2492, 808, 510, 10, 10, 1187, 547, 799, 2657, 115, 1827, 63, 10, 10, 263, 58, 902, 45, 81, 270, 97, 451, 121, 121, 290, 2657, 290, 70, 1778, 44, 1977, 822, 111, 10, 10, 429, 709, 1865, 2584, 46, 83, 63, 10, 10, 263, 58, 478, 979, 302, 1045, 524, 2678, 44, 1466, 2963, 10, 10, 2219, 324, 344, 305, 465, 442, 63, 10, 10, 263, 58, 467, 118, 1796, 2577, 566, 10, 10, 495, 1471, 2501, 1559, 908, 100, 302, 855, 1101, 63, 10, 10, 263, 58, 2844, 10, 10, 323, 806, 2535, 2509, 1095, 63, 10, 10, 263, 58, 2130, 907, 114, 1014, 256, 272, 267, 728, 260, 303, 10, 71, 670, 636, 10, 10, 1433, 481, 611, 442, 63, 10, 10, 263, 58, 2250, 10, 10, 2892, 472, 1474, 108, 284, 2554, 115, 1154, 63, 10, 10, 263, 58, 406, 381, 450, 734, 911, 10, 10, 1277, 1752, 2849, 1034, 1913, 352, 115, 63, 10, 10, 263, 58, 1839, 10, 10, 299, 1095, 1802, 2506, 1919, 533, 559, 594, 109, 387, 260, 63, 10, 10, 263, 58, 2724, 2256, 10, 10, 1974, 2522, 302, 372, 290, 2268, 1147, 63, 10, 10, 263, 58, 1903, 272, 596, 116, 45, 70, 510, 302, 1515, 1116, 44, 2343, 639, 10, 10, 326, 1460, 68, 78, 65, 63, 10, 10, 263, 58, 2873, 338, 114, 291, 77, 1129, 2297, 10, 10, 661, 357, 608, 1627, 110, 330, 276, 608, 281, 264, 2006, 863, 2138, 671, 99, 554, 103, 1147, 63, 10, 10, 263, 58, 406, 787, 2900, 1144, 264, 787, 566, 10, 10, 469, 1087, 1624, 1185, 63, 10, 10, 263, 58, 836, 55, 46, 50, 832, 430, 431, 1087, 115, 10, 10, 657, 370, 1594, 290, 369, 614, 107, 1746, 63, 10, 10, 263, 58, 1114, 377, 89, 1896, 104, 2895, 10, 10, 862, 1380, 391, 871, 2848, 442, 63, 10, 10, 263, 58, 406, 67, 443, 2848, 278, 2996, 10, 10, 2025, 743, 114, 2861, 63, 10, 10, 263, 58, 458, 1641, 10, 10, 873, 264, 1150, 2274, 851, 63, 10, 10, 263, 58, 679, 2683, 10, 10, 2002, 1787, 730, 358, 265, 398, 2761, 63, 10, 10, 263, 58, 406, 107, 378, 371, 111, 10, 10, 429, 2517, 309, 842, 876, 63, 10, 10, 263, 58, 3022, 335, 272, 289, 68, 279, 912, 10, 10, 1653, 2228, 2405, 339, 494, 63, 10, 10, 263, 58, 32, 569, 665, 115, 1395, 10, 10, 994, 2259, 279, 99, 1292, 63, 10, 10, 263, 58, 2760, 76, 541, 506, 560, 121, 10, 10, 2180, 2763, 776, 63, 10, 10, 263, 58, 400, 117, 283, 111, 259, 2083, 279, 10, 10, 495, 1451, 1497, 101, 63, 10, 10, 263, 58, 484, 2721, 1975, 10, 10, 429, 1727, 787, 1408, 444, 283, 63, 10, 10, 263, 58, 400, 2348, 44, 1445, 10, 10, 632, 536, 377, 1411, 353, 122, 1420, 335, 63, 10, 10, 263, 58, 515, 2009, 10, 10, 1117, 1855, 99, 287, 2496, 63, 10, 10, 263, 58, 3001, 356, 609, 635, 324, 535, 10, 10, 495, 871, 366, 2172, 534, 448, 289, 311, 1151, 2977, 380, 876, 63, 10, 10, 263, 58, 546, 327, 1052, 10, 10, 2858, 1985, 1361, 63, 10, 10, 263, 58, 355, 602, 107, 531, 10, 10, 572, 924, 926, 102, 1074, 442, 63, 10, 10, 263, 58, 341, 980, 320, 679, 349, 2997, 1236, 1778, 2235, 97, 10, 10, 994, 386, 102, 440, 514, 794, 2206, 313, 257, 101, 63, 10, 10, 263, 58, 364, 2747, 1108, 45, 119, 2608, 1682, 44, 389, 1108, 284, 542, 10, 10, 2014, 635, 621, 121, 63, 10, 10])


# Create Tensors to hold input and outputs.
"""
trainingTuples = []
trainingTuples.append(([0, 0, 1, 2], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([0, 1, 2, 3], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([1, 2, 3, 1], [0.001, 0.9, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([2, 3, 0, 3], [0.001, 0.001, 0.9, 0.001,    0.001, 0.001]))
trainingTuples.append(([1, 2, 3, 1], [0.001, 0.9, 0.001, 0.001,    0.001, 0.001]))
trainingTuples.append(([2, 3, 4, 3], [0.9, 0.001, 0.001, 0.001,    0.001, 0.001]))
"""

# Construct our model by instantiating the class defined above
modelA = Model0()

loadModel = False # load the model before training?

if loadModel:
    print('load model...')
    modelA.load_state_dict(torch.load('./lmB-checkpoint.pytorchModel'))
    print('...done')

# see https://stackoverflow.com/a/49201237/388614
pytorchTotalParams = sum(p.numel() for p in modelA.parameters() if p.requires_grad)
print(f'#params={pytorchTotalParams}')

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
#optimizer = torch.optim.SGD(modelA.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(modelA.parameters(), lr=0.001)

lossAvg = None

bestModelLoss = 10e6 # loss of best model

for it in range(30000000):
    if (it % 19000) == int(19000/2):
        print(f'store model as snapshot')
        torch.save(modelA.state_dict(), './lmB-checkpoint.pytorchModel')
        
        if lossAvg < bestModelLoss:
            bestModelLoss = lossAvg
            
            print(f'store best model with loss={lossAvg}')
            torch.save(modelA.state_dict(), './lmB-checkpoint-best.pytorchModel')
            
    
    #selIdx = random.randint(0, len(trainingTuples)-1)
    
    tupleRnnCtxVec, tupleStimuliTokens, tuplePredToken = datGen.sample()

    #x = torch.tensor(trainingTuples[selIdx][0])
    
    y = [0.01]*len(tokenEmbeddings)
    y[tuplePredToken] = 0.9
    yTorch = torch.tensor(y)
    yTorch = yTorch.to(device)
    
    x2 = map(lambda v : tokenEmbeddings[v], tupleStimuliTokens) # map index of embedding to actual embedding
    
    x3 = []
    for iv in x2:
        for iv2 in iv:
            x3.extend(iv2)
    
    xTorch = torch.tensor(x3)
    xTorch = xTorch.to(device)

    
    # Forward pass: Compute predicted y by passing x to the model
    yPred = modelA(xTorch)
    
    if (it % 800) == 0:
        pass
        #print('yPred='+str(yPred))
    

    # Compute and print loss
    printLossEvernN = 5800
    
    loss = criterion(yPred, yTorch)
    
    if lossAvg is None:
        lossAvg = loss.item()
    
    a = 0.9993
    lossAvg = a*lossAvg + (1.0-a)*loss.item()
    
    if (it % printLossEvernN) == (printLossEvernN-1):
        print('it=', it, loss.item(), 'lossAvg=', lossAvg)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
