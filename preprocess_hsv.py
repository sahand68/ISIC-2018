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


        path ='data_task3/image/test_299/'
        train_resize_path = 'data_task3/image/hsv/'
        img = cv2.imread(path+ID)
        out = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(train_resize_path+ ID,out)
        return


pool = ThreadPool(12)
pool.map( Image_preprocess,originals_ids )
#pool.map( Mask_preprocess, gt_ids )
