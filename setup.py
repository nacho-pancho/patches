#!/usr/bin/python3
import numpy as np
from distutils.core import setup, Extension

eca=["-O2", "-march=native", "-mtune=native"]
#eca=["-ggdb", "-O0", "-march=native", "-mtune=native"]

setup(name             = "patches",
      version          = "1.0",
      description      = "patch extraction and stitching.",
      author           = "Ignacio Francisco Ramirez Paulino",
      author_email     = "nacho@fing.edu.uy",
      maintainer       = "nacho@fing.edu.uy",
      url              = "https://iie.fing.edu.uy/personal/nacho/",
      ext_modules      = [
          Extension(
              'patches', ['src/patches.c','src/mmap.c','src/mapinfo.c'],
              extra_compile_args=eca)
     ],
     include_dirs=[np.get_include()]
)
