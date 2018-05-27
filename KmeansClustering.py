#encoding:utf8
import os, sys, re, pdb
import codecs
import numpy as np
from PIL import Image
import colorsys
from skimage.feature import hog, daisy
import matplotlib.pyplot as plt
import glob
from CartoonFeatureExtractor import GetImageFeatures
from progressbar import ProgressBar
import time
import pickle
from sklearn.cluster import KMeans
import shutil

def kmeans(featureMat, files):
    km = KMeans(n_clusters = 8)
    km.fit(featureMat)
    print(km.labels_)
    for ind in range(8):
        path = './output/clusters/{}'.format(ind)
        if os.path.exists(path):
            continue
        os.mkdir(path)
    for path, label in zip(files, km.labels_):
        name = os.path.split(path)[-1]
        shutil.copyfile(path, './output/clusters/{}/{}'.format(label, name))
    
    


def LoadFeatureMatrix(inPath):
    try:
        with open(inPath, 'r') as fin:
            return pickle.load(fin)
    except Exception as e:
        print(e)
        return None

def CalcFeatures():
    files = glob.glob('./data/part500/*')
    files = files[:50]
    featureMat = []
    bar = ProgressBar()
    count = 0
    tcList = list()
    bar(range(len(files)))
    for imagePath in files:
        try:
            start_time = time.time()
            featureMat.append(GetImageFeatures(imagePath))
            tcList.append(time.time() - start_time)
        except Exception as e:
            print(e)
            print('Processing imagePath: {}'.format(imagePath))
            break
        bar.next()
    fM = np.array(featureMat)
    with open('fm.pkl', 'w') as fout:
        pickle.dump((fM, files), fout)
    print(fM.shape)
    print('Average time:{} s'.format(np.average(tcList)))

def main():
    files = glob.glob('./data/part500/*')
    files = files[:50]
    fm = LoadFeatureMatrix('fm.pkl')
    kmeans(fm, files)


if __name__ == '__main__':
    main()
