#!/usr/bin/python3

from distutils.core import setup, Extension

eca=["-fopenmp", "-O3", "-march=native", "-mtune=native"]
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
              'pacore', ['src/pacore.c','src/mapinfo.c','src/mmap.c'],
              libraries = ['gomp'],
              extra_compile_args=eca)
     ]
)
