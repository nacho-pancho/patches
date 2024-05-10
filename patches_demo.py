#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import patches
import matplotlib.pyplot as plt
import numpy.linalg as la
import argparse 
import skimage.io as imgio

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Patch extraction demo.",epilog="Epilog.")
    ap.add_argument("-u","--use-mmap",action="store_true")
    ap.add_argument("-w","--width",type=int,default=8)
    ap.add_argument("-s","--stride",type=int,default=8)
    ap.add_argument("-d","--dilation",type=int,default=8)
    ap.add_argument("-i","--input",type=str,help="Input image")
    ap.add_argument("-m","--mask",type=str,help="Binary mask. Only patches which are completely covered by the mask are extracted.")
    ap.add_argument("-o","--output",type=str,help="Output image (dictionary)")

    args = ap.parse_args()

    if args["use_mmap"]:
        print('USING MMAP')
        create_patches_matrix = patches.create_patches_matrix_mmap
    else:
        print('USING PHYSICAL MEMORY')
        create_patches_matrix = patches.create_patches_matrix

    #
    # INITIALIZATION
    #
    img = imgio.imread(args["input"])
    N1,N2,_ = img.shape
    w1 = w2 = args["width"]
    s1 = s2 = args["stride"]
    d1 = d2 = args["dilation"]

    tic = time.time()
    tic00 = tic
    #
    EXTRACT_VALID = 0
    _mask_ = None
    patches.init_mapinfo(N1, N2, w1, w2, s1, s2,_mask_,EXTRACT_VALID)
    #
    # auxiliary patches matrix
    # output, also used as temporary storage
    Y = np.zeros(img.shape)
    Z      = create_patches_matrix()
    n,m    = Z.shape
    patches.extract_partial_to(Y, Z)
    patches.stitch_partial_to(Z, Y)
    
