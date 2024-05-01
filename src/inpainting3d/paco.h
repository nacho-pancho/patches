#ifndef PACO_H
#define PACO_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#if 0
typedef npy_float32 sample_t;
#define  SAMPLE_TYPE_ID NPY_FLOAT32
#else
typedef npy_float64 sample_t;
#define  SAMPLE_TYPE_ID NPY_FLOAT64
#endif
#endif
