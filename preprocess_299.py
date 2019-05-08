import numpy as np
import os
import sys
import random
import warnings
from tqdm import tqdm as tqdm

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import cv2
import matplotlib

import multiprocessing as mp
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
originals_ids = next(os.walk('data_task3/image/test/'))[2]
#gt_ids = next(os.walk('C://Users//sahan//ipthw//Melanoma_segmentation//data_task2//gt//'))[2]



def Image_preprocess(ID):


        path ='data_task3/image/test/'
        train_resize_path = 'data_task3/image/test_299/'
        img = cv2.imread(path+ID)
        img = cv2.resize( img, (299, 299))
        cv2.imwrite(train_resize_path+ ID,img)
        return
def Mask_preprocess(ID):


        mask_path = 'data_task2/gt/'
        mask_resize_path = 'data_task2/gt_224/'
        img = cv2.imread(mask_path+ID, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize( img, (299, 299))
        cv2.imwrite(mask_resize_path + ID,img)
        return

pool = ThreadPool(12)
pool.map( Image_preprocess,originals_ids )
#pool.map( Mask_preprocess, gt_ids )
