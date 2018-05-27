#encoding:utf8
import os, sys, re, pdb
import codecs
import time
import numpy as np
from PIL import Image
import collections
import colorsys
from skimage.feature import hog, daisy
import matplotlib.pyplot as plt

def MaxColorNumber(img, colorCountThreshold = 10,
        VThreshold = 125):
    # Fixed length
    N = 126
    vec = []
    # colors = img.getcolors(maxcolors = 256 ** 3)
    rgbArray = list(img.getdata())

    # RGB histogram
    rArray, gArray, bArray = zip(*rgbArray)
    vec.extend(np.histogram(rArray, bins = 20)[0])
    vec.extend(np.histogram(gArray, bins = 20)[0])
    vec.extend(np.histogram(bArray, bins = 20)[0])

    # Counter
    majorColorSet = set()
    # colorCounter = collections.Counter()
    colorCounter = collections.defaultdict(int)
    for rgb in rgbArray:
        colorCounter[rgb] += 1
        if colorCounter[rgb] > colorCountThreshold:
            majorColorSet.add(rgb)
    colorCounterArray = list(colorCounter.items())
    colorCounterArray.sort(key = lambda x:x[1], reverse = True)
    colorCount = len(colorCounter)
    majorColorCount = len(majorColorSet)
    top10CommonColor = colorCounterArray[:10]
    top10Count = sum([x[1] for x in top10CommonColor])
    Top10Ratio = int(top10Count * 1e6 / len(rgbArray))
    majorColorRatio = int(sum([colorCounter[x] for x in majorColorSet]) * 1e6 / len(rgbArray))

    # HSV
    hsvArray = [[colorCounter[x],
        colorsys.rgb_to_hsv(x[0], x[1], x[2])] for x in colorCounter]
    count = 0
    avgS = 0
    Vcount = 0
    for c, hsv in hsvArray:
        count += c
        avgS += hsv[1]
        if hsv[2] > VThreshold:
            Vcount += c

    avgS = int(avgS * 1e6 / count)
    VcountRatio = int(Vcount * 1e6 / count)

    # HSV histogram
    vec.extend(np.histogram([x[1][0] for x in hsvArray], bins = 20)[0])
    vec.extend(np.histogram([x[1][1] for x in hsvArray], bins = 20)[0])
    vec.extend(np.histogram([x[1][2] for x in hsvArray], bins = 20)[0])

    vec.append(majorColorRatio)
    vec.append(colorCount)
    vec.append(majorColorCount)
    vec.append(Top10Ratio)
    vec.append(avgS)
    vec.append(VcountRatio)
    assert len(vec) == N
    return vec
    
def GetImageFeatures(imagePath):
    # Load Image
    img = Image.open(imagePath)
    img = img.convert('RGB')
    # img.show()

    # Append Features
    features = []
    # Color Features
    features.extend(MaxColorNumber(img))

    features.extend(MorphFeatures(img))

    # Close
    img.close()
    return features

def MorphFeatures(img):
    vecN = 32 #+ 32 * 2
    vec = []
    greyArray = list(img.convert('L').getdata())
    M = img.height
    N = img.width
    cellN = min(N, M)
    image = np.reshape(greyArray, [M, N])
    
    # HOG
    res = hog(image, orientations=32,
            pixels_per_cell=(cellN, cellN),
                    cells_per_block=(1, 1),
                    block_norm = 'L2-Hys',
                    feature_vector = True)
    res = res.reshape((-1,32))
    res = np.average(res, 0)
    vec.extend(res.tolist())

    # Daisy
    # res = daisy(image, radius = cellN / 2 - 1, histograms = 1)
    # daisyShape = res.shape
    # res = res.reshape(daisyShape[0] * daisyShape[1], daisyShape[2])
    # res *= 1e6
    # res = res.astype(int)
    # vec.extend(res[0,:].tolist())
    # vec.extend(res[-1,:].tolist())
    if len(vec) != vecN:
        raise Exception('res.shape = {}'.format(res.shape))
    return vec
    
def main():
    # imagePath = './data/cc/n1.jpg'
    # imagePath = './data/cc/F0F9D13586BDD65EC48AE36F313C3A16C71ACFF8.jpg'
    imagePath = './data/part500/F08233C3426280CE2C8E6B4C711ACDFED87E443E.jpg'
    start_time = time.time()
    vec = GetImageFeatures(imagePath)
    print('{} s'.format(time.time() - start_time))
    print('Number of features:', len(vec))

if __name__ == '__main__':
    main()
