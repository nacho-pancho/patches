#ifndef PACO_MMAP_H
#define PACO_MMAP_H

//#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

void* mmap_alloc(npy_uint64 size);

#endif
