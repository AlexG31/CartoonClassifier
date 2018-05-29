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

def kmeans(featureMat, files,
        outputFolder = './output',
        n_clusters = 32):
    km = KMeans(n_clusters = n_clusters)
    km.fit(featureMat)
    with open(os.path.join(outputFolder, 'kmeans{}.pkl'.format(n_clusters)), 'w') as fout:
        pickle.dump(km, fout)
        print('Kmeans model saved.')
    print(km.labels_)
    for ind in range(n_clusters):
        path = os.path.join(outputFolder,
                'clusters{}/{}'.format(n_clusters, 
            ind))
        if os.path.exists(path):
            continue
        os.mkdir(path)
    for path, label in zip(files, km.labels_):
        name = os.path.split(path)[-1]
        shutil.copyfile(path, '{}/clusters{}/{}/{}'.format(outputFolder,
            n_clusters,
            label, name))
    
    


def LoadFeatureMatrix(inPath):
    try:
        with open(inPath, 'r') as fin:
            return pickle.load(fin)
    except Exception as e:
        print(e)
        return None

def CalcFeatures(inPath):
    files = glob.glob(os.path.join(inPath, '*'))
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
    # inPath = sys.argv[1]
    # CalcFeatures(inPath)
    # files = glob.glob('./data/part500/*')
    # files = files[:50]
    fm, files = LoadFeatureMatrix('Features-10000.pkl')
    kmeans(fm, files)


if __name__ == '__main__':
    main()
