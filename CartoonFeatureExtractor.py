#encoding:utf8
import os, sys, re, pdb
import codecs
import numpy as np
from PIL import Image
import colorsys

def MaxColorNumber(img, colorCountThreshold = 10,
        VThreshold = 125):
    # Fixed length
    N = 125
    vec = []
    colors = img.getcolors(maxcolors = 256 ** 3)
    rgbArray = list(img.getdata())

    # RGB histogram
    rArray, gArray, bArray = zip(*rgbArray)
    vec.extend(np.histogram(rArray, bins = 20)[0])
    vec.extend(np.histogram(gArray, bins = 20)[0])
    vec.extend(np.histogram(bArray, bins = 20)[0])
    colorCountArray = [x[0] for x in colors]
    colorCount = len(colors)
    majorColorCount = len(filter(lambda x: x[0] > colorCountThreshold, colors))
    top10Count = sum(sorted(colorCountArray, reverse = True)[:10])
    Top10Ratio = int(top10Count * 1000 / sum(colorCountArray))

    # HSV
    hsvArray = [[x[0], colorsys.rgb_to_hsv(x[1][0], x[1][1], x[1][2])] for x in colors]
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
    features.extend(MaxColorNumber(img))
    
    # Color Features

    # Close
    img.close()
    return features

def main():
    # imagePath = './data/cc/n1.jpg'
    imagePath = './data/cc/F0F9D13586BDD65EC48AE36F313C3A16C71ACFF8.jpg'
    vec = GetImageFeatures(imagePath)
    print('Number of features:', len(vec))

if __name__ == '__main__':
    main()
