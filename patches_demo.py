#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import pacore
import matplotlib.pyplot as plt
import numpy.linalg as la

if __name__ == "__main__":
    if use_mmap:
        print('USING MMAP')
        create_patches_matrix = pacore.create_patches_matrix_mmap
        create_image_matrix   = pacore.create_image_matrix_mmap
    else:
        print('USING PHYSICAL MEMORY')
        create_patches_matrix = pacore.create_patches_matrix
        create_image_matrix   = pacore.create_image_matrix
    #
    # INITIALIZATION
    #
    tic = time.time()
    tic00 = tic
    #
    pacore.init_mapinfo(N1, N2, N3, w1, w2, w3, s1, s2, s3,_mask_,0)
    #
    # auxiliary patches matrix
    # output, also used as temporary storage
    Y = create_image_matrix()
    Z      = create_patches_matrix()
    n,m    = Z.shape
    pacore.extract_partial_to(Y, Z)
    pacore.stitch_partial_to(Z, Y)
    
