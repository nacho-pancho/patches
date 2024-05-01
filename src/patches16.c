#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>

///
/// Contains all the relevant parameters  about the patch decomposition procedure.
///
#define CCLIP(x,a,b) ( (x) > (a) ? ( (x) < (b) ? (x) : (b) ) : (a) )
#define UCLIP(x,a) ( (x) < (a) ? (x) : (a)-1 )

typedef enum {
    EXTRACT_EXACT =0, ///< only extract patches which contain true pixels, possibly leaving bordering pixels out
    EXTRACT_FULL, ///<  extract patches so that whole image is covered, extrapolating border pixels as needed
} extract_t;

typedef struct mapinfo_s {
    npy_int64 M; ///< size of signal along dim1; provided
    npy_int64 N; ///< size of signal along dim1; provided
    npy_int64 L; ///< signal linear length = M*N
    npy_int64 stride1; ///< stride  along dim1, provided
    npy_int64 stride2; ///< stride along dim2, provided
    npy_int64 covering; ///< signal covering strategy provided
    npy_int64 width1; ///< width of patches along dim1, provided, possibly corrected
    npy_int64 width2; ///< width of patches along dim2, provided, possibly corrected
    npy_int64 m; ///< patch space dimension; computed
    npy_int64 nx; ///< number of patches along dim 2; computed
    npy_int64 ny; ///< number of patches along dim 2; computed
    npy_int64 n; ///< number of patches; computed
    npy_int64 l; ///< patches matrix linear length
} mapinfo;

mapinfo build_mapinfo(const npy_int64 _M,
                      const npy_int64 _N,
                      const npy_int64 _w,
                      const npy_int64 _s,
                      const npy_int64 _cov);


/// Python adaptors
static PyObject *create_patches_matrix(PyObject* self, PyObject* args);
static PyObject *create_norm_matrix   (PyObject* self, PyObject* args);
static PyObject *extract              (PyObject *self, PyObject *args);
static PyObject *extract_to           (PyObject *self, PyObject *args);
static PyObject *stitch               (PyObject *self, PyObject *args);
static PyObject *stitch_to            (PyObject *self, PyObject *args);
static PyObject *pad                  (PyObject *self, PyObject *args);


/*****************************************************************************
 * Python/NumPy -- C boilerplate
 *****************************************************************************/
//
//--------------------------------------------------------
// function declarations
//--------------------------------------------------------
//
static PyMethodDef methods[] = {
    { "create_patches_matrix", create_patches_matrix, METH_VARARGS, "Creates a matrix for allocating patches."},
    { "create_norm_matrix", create_norm_matrix, METH_VARARGS, "Creates a normalization matrix for using when stitching. Only one such matrix is needed for each combination of signal dimensions, width and stride ."},
    { "extract", extract, METH_VARARGS, "Extracts patches from a signal to a new patches matrix"},
    { "extract_to", extract_to, METH_VARARGS, "Extracts patches from a signal to a preallocated patches matrix."},
    { "stitch", stitch, METH_VARARGS, "Stitches patches into a new signal.."},
    { "stitch_to", stitch_to, METH_VARARGS, "Stitches patches into a preallocated signal."},
    { "pad", pad, METH_VARARGS, "Increases signal dimension (at the ends of the dimensions) so that an exact number of patches of the given width and stride fit in it."},
    { NULL, NULL, 0, NULL } /* Sentinel */
};
//
//--------------------------------------------------------
// module initialization
//--------------------------------------------------------
//
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT,
				     "patches16",
				     "Patch mapping, half precision",
				     -1, methods};
PyMODINIT_FUNC PyInit_patches16(void) {
  Py_Initialize();
  return PyModule_Create(&module);
}

//
//--------------------------------------------------------
// compute size of patches grid
//--------------------------------------------------------
//
static npy_int64 compute_grid_size(const npy_int64 size, const npy_int64 width, const npy_int64 stride, unsigned extract_type) {
    // return M, the number of patches to be extracted along a dimension
    // EXTRACT_EXACT: last index (li) must be < m so
    // li = (M-1)*stride + width -1 <= m -1 => M = floor [ (m - width + stride) / stride ]
    // EXTRACT_FULL: first index of last patch (fi) must be < m so
    // fi = (M-1)*stride  <= m - 1 => stride*M <= m + stride - 1 => M = floor [(m -1 + stride) / stride ]
    return extract_type == EXTRACT_EXACT ? (size+stride-width)/stride : (size + stride - 1)/stride;
}
//
//--------------------------------------------------------
// create patches matrix
//--------------------------------------------------------
//
static PyObject *create_patches_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    npy_int64 M, N, w, s;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llll",
                         &M,
                         &N,
                         &w,
                         &s
                        )) {
        return NULL;
    }

    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
// create normalization matrix
//--------------------------------------------------------
//
static PyObject *create_norm_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    npy_int64 M, N, w, s;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llll",
                         &M,
                         &N,
                         &w,
                         &s
                        )) {
        return NULL;
    }

    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {M,N};
    py_R = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_R,0);
    const npy_int64 mg = map.ny;
    const npy_int64 ng = map.nx;
    const npy_int64 stride1 = map.stride1;
    const npy_int64 stride2 = map.stride2;
    const npy_int64 width1 = map.width1;
    const npy_int64 width2 = map.width2;

    for (npy_int64 ig = 0, i = 0, k = 0; ig < mg; ++ig, i += stride1) { // k = patch index
        for (npy_int64 jg = 0, j = 0; jg < ng; ++jg, ++k, j += stride2) {
            for (npy_int64 ii = 0, l = 0; ii < width1; ++ii) {
                for (npy_int64 jj = 0; jj < width2; ++l, ++jj) { // l = dimension within patch
                    *((npy_double*)PyArray_GETPTR2(py_R,i+ii,j+jj)) += 1.0; // increase number of copies of this pixel
                } // for ii
            }// for jj
        } // for j
    } // for i

    PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject*)py_R);
    if (iter == NULL)
        return NULL;

    while (PyArray_ITER_NOTDONE(iter)) {
        npy_double* Rij = PyArray_ITER_DATA(iter);
        if (*Rij) *Rij = 1.0/(*Rij);
        PyArray_ITER_NEXT(iter);
    }

    return PyArray_Return(py_R);
}
//
//--------------------------------------------------------
// pad
//--------------------------------------------------------
//
static PyObject *pad(PyObject *self, PyObject *args) {
    PyArrayObject *py_I, *py_P;
    npy_int64 M, N, w, s;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!ll",
                         &PyArray_Type, &py_I, &w, &s)) {
        return NULL;
    }
    M = PyArray_DIM(py_I,0);
    N = PyArray_DIM(py_I,1);
    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);
    //
    // compute dimensions of padded image
    //
    npy_int64 M2 = s*(map.ny-1) + w;
    npy_int64 N2 = s*(map.nx-1) + w;
    npy_intp dims[2] = {M2,N2};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    //
    // copy padded image
    //
    for (npy_int64 i = 0; i < M2; i++) {
        for (npy_int64 j = 0; j < N2; j++) {
            *((npy_double*)PyArray_GETPTR2(py_P,i,j)) = *(npy_double*)PyArray_GETPTR2(py_I, UCLIP(i,M), UCLIP(j,N) );
        }
    }
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
// stitch
//--------------------------------------------------------
//
void _stitch_(PyArrayObject* P, mapinfo* map, PyArrayObject* I, PyArrayObject* R) {
    const npy_int64 M = map->M;
    const npy_int64 N = map->N;
    const npy_int64 mg = map->ny;
    const npy_int64 ng = map->nx;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 width1 = map->width1;
    const npy_int64 width2 = map->width2;

    npy_int64 k = 0;
    npy_int64 i = 0;
    for (npy_int64 ig = 0; ig < mg; ++ig) {
        for (npy_int64 jg = 0, j = 0; jg < ng; ++k, ++jg, j += stride2) {
            for (npy_int64 ii = 0, l = 0; ii < width1; ++ii) {
                for (npy_int64 jj = 0; jj < width2; ++l, ++jj) {
                    npy_double* pIij = (npy_double*)PyArray_GETPTR2(I,i+ii,j+jj);
                    *pIij += *((npy_double*)PyArray_GETPTR2(P,k,l));
                } // for ii
            }// for jj
        } // for jg
        i+= stride1;
    } // for ig
    //
    // normalization
    //

    for (npy_int64 i = 0; i < M; ++i ) {
        for (npy_int64 j = 0; j < N; ++j ) {
            const double Rij = *((npy_double*)PyArray_GETPTR2(R,i,j));
            *((npy_double*)PyArray_GETPTR2(I,i,j)) *= Rij;
        } // for j
    } // for i
}

static PyObject *stitch(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I, *py_R;
    npy_int64 M, N, w, s;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llllO!O!",
                         &M,
                         &N,
                         &w,
                         &s,
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_R)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {M,N};
    py_I = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,&map,py_I,py_R);
    return PyArray_Return(py_I);
}
//
//--------------------------------------------------------
// stitch
//--------------------------------------------------------
//
static PyObject *stitch_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I, *py_R;
    npy_int64 M, N, w, s;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llllO!O!O!",
                         &M,
                         &N,
                         &w,
                         &s,
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_R)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,&map,py_I,py_R);

    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
// extract
//--------------------------------------------------------
//
int _extract_(PyArrayObject* I, mapinfo* map, PyArrayObject* P) {
    const npy_int64 mg = map->ny;
    const npy_int64 ng = map->nx;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 width1 = map->width1;
    const npy_int64 width2 = map->width2;

    npy_int64 k = 0, i = 0;
    //#ifdef _OPENMP
    //#pragma omp parallel for
    //#endif
    for (npy_int64 ig = 0; ig < mg; ++ig) { // k = patch index
        for (npy_int64 jg = 0, j = 0; jg < ng; ++jg, ++k, j += stride2) {
            for (npy_int64 ii = 0, l = 0; ii < width1; ++ii) {
                for (npy_int64 jj = 0; jj < width2; ++l, ++jj) { // l = dimension within patch
                    const npy_double* aux = (npy_double*)PyArray_GETPTR2(I, i+ii, j+jj);
                    *((npy_double*)PyArray_GETPTR2(P,k,l)) =  *aux;
                } // for ii
            }// for jj
        } // for jg
        i += stride1;
    } // for ig

    return 1;
}

static PyObject *extract_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    npy_int64 M, N, w, s;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llllO!O!",
                         &M,
                         &N,
                         &w,
                         &s,
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_P
                        )
      ) {
        return NULL;
    }
    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);

    _extract_(py_I,&map,py_P);
    Py_RETURN_NONE;
}

static PyObject *extract(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    npy_int64 M, N, w, s;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llllO!",
                         &M,
                         &N,
                         &w,
                         &s,
                         &PyArray_Type, &py_I
                        )
      ) {
        return NULL;
    }
    mapinfo map = build_mapinfo(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    _extract_(py_I,&map,py_P);
    return PyArray_Return(py_P);
}

//
//*****************************************************************************
//  Python/NumPy -- C adaptor functions
//*****************************************************************************

mapinfo build_mapinfo(const npy_int64 _M,
                      const npy_int64 _N,
                      const npy_int64 _w,
                      const npy_int64 _s,
                      const npy_int64 _cov) {
    mapinfo map;
    map.M = _M;
    map.N = _N;
    map.L = _M*_N;
    map.stride1 = _s;
    map.stride2 = _s;
    map.covering = _cov;
    map.width1 = _w > _M ? _M : _w;
    map.width2 = _w > _N ? _N : _w;
    map.m = map.width1*map.width2;
    map.nx = compute_grid_size(map.N,map.width2,map.stride2,map.covering);
    map.ny = compute_grid_size(map.M,map.width1,map.stride1,map.covering);
    map.n = map.nx*map.ny;
    map.l = map.m*map.n;
    return map;
}


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

///
/// Contains all the relevant parameters  about the patch decomposition procedure.
///
#define CCLIP(x,a,b) ( (x) > (a) ? ( (x) < (b) ? (x) : (b) ) : (a) )
#define UCLIP(x,a) ( (x) < (a) ? (x) : (a)-1 )

typedef enum {
    EXTRACT_EXACT =0, ///< only extract patches which contain true pixels, possibly leaving bordering pixels out
    EXTRACT_FULL, ///<  extract patches so that whole image is covered, extrapolating border pixels as needed
} extract_t;

void* mmap_alloc(npy_uint64 size) {
    static unsigned mmap_file_num = 0;
    static char tmp[128];
    snprintf(tmp,128,"/datos/data/mmap/mmap_file_%05d.mmap",mmap_file_num++);
    int fd = open(tmp,O_RDWR | O_CREAT, S_IRWXU);
    printf("requested size=%luMB\t",size>>20);
    const npy_uint64 psize = getpagesize();
    //const npy_uint64 psize = (1UL<<21); // 2MB
    npy_int64 npages = (size + psize)/psize;
    printf("page size=%luB\tnum. pages=%luMB\t",psize,npages);
    size =  npages * psize;
    printf("paged size=%luMB\n",size>>20);
    const unsigned long t = 0;
    const npy_int64 nwrites = size / sizeof(unsigned long);
    FILE* FD = fdopen(fd,"w");
    for (npy_int64 i = 0; i < nwrites; i++) {
        //write(fd,&t,sizeof(unsigned long));
        fwrite(&t,sizeof(unsigned long),1,FD);
    }
    //close(fd);
    fclose(FD);
    fd = open(tmp,O_RDWR);
    void* data = mmap(NULL,size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    //void* data = mmap(NULL,size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT), fd, 0);
    if (data == MAP_FAILED) {
        printf("mmap failed with errno %d!!\n",errno);
        exit(errno);
    }
    return data;
}
#define MMAP_FLAGS  (MAP_HUGETLB)

typedef struct mapinfo_s {
    npy_int64 N1; ///< size of signal along dim1; provided
    npy_int64 N2; ///< size of signal along dim2; provided
    npy_int64 N3; ///< size of signal along dim3; provided
    npy_int64 L; ///< signal linear length = M*N*T
    npy_int64 stride1; ///< stride  along dim1, provided
    npy_int64 stride2; ///< stride along dim2, provided
    npy_int64 stride3; ///< stride along dim3, provided
    npy_int64 covering; ///< signal covering strategy, provided
    npy_int64 width1; ///< width of patches along dim1, provided, possibly corrected
    npy_int64 width2; ///< width of patches along dim2, provided, possibly corrected
    npy_int64 width3; ///< width of patches along dim3, provided, possibly corrected
    npy_int64 m; ///< patch space dimension; computed
    npy_int64 n1; ///< number of patches along dim 2; computed
    npy_int64 n2; ///< number of patches along dim 1; computed
    npy_int64 n3; ///< number of patches along dim 3; computed
    npy_int64 n; ///< number of patches; computed n1*n2*n3
    npy_int64 l; ///< patches matrix linear length
} mapinfo;

mapinfo build_mapinfo(const npy_int64 _N1,
                          const npy_int64 _N2,
                          const npy_int64 _T,
                          const npy_int64 _w1,
                          const npy_int64 _w2,
                          const npy_int64 _w3,
                          const npy_int64 _s1,
                          const npy_int64 _s2,
                          const npy_int64 _s3,
                          const npy_int64 _cov);


/// Python adaptors
static PyObject *create_patches_matrix_mmap(PyObject* self, PyObject* args);
static PyObject *create_image_matrix_mmap  (PyObject* self, PyObject* args);
static PyObject *create_norm_matrix_mmap   (PyObject* self, PyObject* args);
static PyObject *destroy_matrix_mmap    (PyObject* self, PyObject* args);
static PyObject *create_image_matrix  (PyObject* self, PyObject* args);
static PyObject *create_patches_matrix(PyObject* self, PyObject* args);
static PyObject *create_norm_matrix   (PyObject* self, PyObject* args);
static PyObject *extract                (PyObject *self, PyObject *args);
static PyObject *extract_to             (PyObject *self, PyObject *args);
static PyObject *stitch                 (PyObject *self, PyObject *args);
static PyObject *stitch_to              (PyObject *self, PyObject *args);
static PyObject *pad                    (PyObject *self, PyObject *args);


/*****************************************************************************
 * Python/NumPy -- C boilerplate
 *****************************************************************************/
//
//--------------------------------------------------------
// function declarations
//--------------------------------------------------------
//
static PyMethodDef methods[] = {
    {
        "create_patches_matrix",      create_patches_matrix, METH_VARARGS,
        "Creates a matrix for allocating 3D patches."
    },
    {
        "create_image_matrix",        create_image_matrix, METH_VARARGS,
        "Create. Only one such matrix is needed for each combination of signal dimensions, width and stride."
    },
    {
        "create_norm_matrix",         create_norm_matrix, METH_VARARGS,
        "Creates a normalization matrix for using when stitching. Only one such matrix is needed for each combination of signal dimensions, width and stride."
    },
    {
        "create_patches_matrix_mmap", create_patches_matrix_mmap, METH_VARARGS,
        "."
    },
    {
        "create_image_matrix_mmap",   create_image_matrix_mmap, METH_VARARGS,
        "."
    },
    {
        "create_norm_matrix_mmap",    create_norm_matrix_mmap, METH_VARARGS,
        "."
    },
    {
        "destroy_matrix_mmap",          destroy_matrix_mmap, METH_VARARGS,
        "."
    },
    {
        "extract",                      extract, METH_VARARGS,
        "Extracts 3D patches from a signal to a new patches matrix"
    },
    {
        "extract_to",                   extract_to, METH_VARARGS,
        "Extracts 3D patches from a signal to a preallocated patches matrix."
    },
    {
        "stitch",                       stitch, METH_VARARGS,
        "Stitches 3D patches into a new signal.."
    },
    {
        "stitch_to",                    stitch_to, METH_VARARGS,
        "Stitches 3D patches into a preallocated signal."
    },
    {
        "pad",                          pad, METH_VARARGS,
        "Increases signal dimension (at the ends of the dimensions) so that an exact number of patches of the given width and stride fit in it."
    },
    { NULL, NULL, 0, NULL } /* Sentinel */
};
//
//--------------------------------------------------------
// module initialization
//--------------------------------------------------------
//
PyMODINIT_FUNC initpatches(void) {
    (void) Py_InitModule("patches", methods);
    import_array();
}

//
//--------------------------------------------------------
// compute size of patches grid
//--------------------------------------------------------
//
static npy_int64 compute_grid_size(const npy_int64 size, const npy_int64 width, const npy_int64 stride, unsigned extract_type) {
    // return M, the number of patches to be extracted along a dimension
    // EXTRACT_EXACT: last index (li) must be < m so
    // li = (M-1)*stride + width -1 <= m -1 => M = floor [ (m - width + stride) / stride ]
    // EXTRACT_FULL: first index of last patch (fi) must be < m so
    // fi = (M-1)*stride  <= m - 1 => stride*M <= m + stride - 1 => M = floor [(m -1 + stride) / stride ]
    return extract_type == EXTRACT_EXACT ? (size+stride-width)/stride : (size + stride - 1)/stride;
}
//
//--------------------------------------------------------
// create patches matrix
//--------------------------------------------------------
//
static PyObject *create_patches_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllll",
                         &N1, &N2, &N3, &w1, &w2, &w3, &s1, &s2, &s3)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(N1, N2, N3, w1, w2, w3,s1, s2, s3, EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    return PyArray_Return(py_P);
}

static PyObject *create_patches_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllll",
                         &N1, &N2, &N3, &w1, &w2, &w3, &s1, &s2, &s3)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(N1, N2, N3, w1, w2, w3,s1, s2, s3, EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    npy_int64 size = map.n*map.m*sizeof(npy_double);
    void* data = mmap_alloc(size);
    py_P = (PyArrayObject*) PyArray_SimpleNewFromData(2,&dims[0],NPY_DOUBLE, data);
    return PyArray_Return(py_P);
}

static PyObject *destroy_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_X;
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &py_X)) {
        return NULL;
    }
    Py_DECREF(py_X);
    //munmap(PyArray_DATA(py_X));
    Py_RETURN_NONE;
}

static PyObject *create_image_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    npy_int64 N1, N2, N3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lll",
                         &N1, &N2, &N3)) {
        return NULL;
    }
    npy_intp dims[3] = {N1,N2,N3};
    npy_int64 size = N1*N2*N3*sizeof(npy_double);
    void* data = mmap_alloc(size);
    py_R = (PyArrayObject*) PyArray_SimpleNewFromData(3,&dims[0],NPY_DOUBLE, data);
    return PyArray_Return(py_R);
}

static PyObject *create_image_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    npy_int64 N1, N2, N3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lll",
                         &N1, &N2, &N3)) {
        return NULL;
    }

    npy_intp dims[3] = {N1,N2,N3};
    py_R = (PyArrayObject*) PyArray_SimpleNew(3,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_R,0);
    return PyArray_Return(py_R);
}

//
//--------------------------------------------------------
// create normalization matrix
//--------------------------------------------------------
//
static PyObject *create_norm_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllll",
                         &N1, &N2, &N3, &w1, &w2, &w3, &s1, &s2, &s3)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(N1, N2, N3, w1, w2, w3, s1, s2, s3, EXTRACT_EXACT);
    npy_intp dims[3] = {N1,N2,N3};
    py_R = (PyArrayObject*) PyArray_SimpleNew(3,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_R,0);
    const npy_int64 ng1     = map.n1;
    const npy_int64 ng2     = map.n2;
    const npy_int64 ng3     = map.n3;
    const npy_int64 stride1 = map.stride1;
    const npy_int64 stride2 = map.stride2;
    const npy_int64 stride3 = map.stride3;
    const npy_int64 width1  = map.width1;
    const npy_int64 width2  = map.width2;
    const npy_int64 width3  = map.width3;
    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; g1++, i1 += stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3) {
                for (npy_int64 ii1 = 0; ii1 < width1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < width2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < width3; ++ii3) {
                            *( (npy_double*)PyArray_GETPTR3(py_R,i1+ii1,i2+ii2,i3+ii3) ) += 1.0; // increase number of copies of this pixel
                        } //
                    } //
                } //
            }// for jj
        } // for j
    } // for i

    PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject*)py_R);
    if (iter == NULL)
        return NULL;

    while (PyArray_ITER_NOTDONE(iter)) {
        npy_double* Rij = PyArray_ITER_DATA(iter);
        if (*Rij) *Rij = 1.0/(*Rij);
        PyArray_ITER_NEXT(iter);
    }

    return PyArray_Return(py_R);
}


//
//--------------------------------------------------------
// create normalization matrix
//--------------------------------------------------------
//
static PyObject *create_norm_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllll",
                         &N1, &N2, &N3, &w1, &w2, &w3, &s1, &s2, &s3)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(N1, N2, N3, w1, w2, w3, s1, s2, s3, EXTRACT_EXACT);
    npy_intp dims[3] = {N1,N2,N3};
    npy_int64 size = N1*N2*N3*sizeof(npy_double);
    void* data = mmap_alloc(size);
    py_R = (PyArrayObject*) PyArray_SimpleNewFromData(3,&dims[0],NPY_DOUBLE, data);
    PyArray_FILLWBYTE(py_R,0);
    const npy_int64 ng1     = map.n1;
    const npy_int64 ng2     = map.n2;
    const npy_int64 ng3     = map.n3;
    const npy_int64 stride1 = map.stride1;
    const npy_int64 stride2 = map.stride2;
    const npy_int64 stride3 = map.stride3;
    const npy_int64 width1  = map.width1;
    const npy_int64 width2  = map.width2;
    const npy_int64 width3  = map.width3;
    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; g1++, i1 += stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3) {
                for (npy_int64 ii1 = 0; ii1 < width1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < width2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < width3; ++ii3) {
                            *( (npy_double*)PyArray_GETPTR3(py_R,i1+ii1,i2+ii2,i3+ii3) ) += 1.0; // increase number of copies of this pixel
                        } //
                    } //
                } //
            }// for jj
        } // for j
    } // for i

    PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject*)py_R);
    if (iter == NULL)
        return NULL;

    while (PyArray_ITER_NOTDONE(iter)) {
        npy_double* Rij = PyArray_ITER_DATA(iter);
        if (*Rij) *Rij = 1.0/(*Rij);
        PyArray_ITER_NEXT(iter);
    }

    return PyArray_Return(py_R);
}

//
//--------------------------------------------------------
// pad
//--------------------------------------------------------
//
static PyObject *pad(PyObject *self, PyObject *args) {
    PyArrayObject *py_I, *py_P;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!llllll",
                         &PyArray_Type, &py_I, &w1, &w2, &w3, &s1, &s2, &s3)) {
        return NULL;
    }
    N1 = PyArray_DIM(py_I,0);
    N2 = PyArray_DIM(py_I,1);
    N3 = PyArray_DIM(py_I,2);
    mapinfo map = build_mapinfo(N1,N2,N3,w1,w2,w3,s1,s2,s3,EXTRACT_EXACT);
    //
    // compute dimensions of padded image
    //
    npy_int64 N1b = s1*(map.n1-1) + w1;
    npy_int64 N2b = s2*(map.n2-1) + w2;
    npy_int64 N3b = s3*(map.n3-1) + w3;
    npy_intp dims[3] = {N1b,N2b,N3b};
    py_P = (PyArrayObject*) PyArray_SimpleNew(3,dims,NPY_DOUBLE);
    //
    // copy padded image
    //
    for (npy_int64 i1 = 0; i1 < N1b; i1++) {
        for (npy_int64 i2 = 0; i2 < N2b; i2++) {
            for (npy_int64 i3 = 0; i3 < N3b; i3++) {
                *((npy_double*)PyArray_GETPTR3(py_P,i1,i2,i3)) = *(npy_double*)PyArray_GETPTR3(py_I, UCLIP(i1,N1), UCLIP(i2,N2), UCLIP(i3,N3) );
            }
        }
    }
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
// stitch
//--------------------------------------------------------
//
void _stitch_(PyArrayObject* P, mapinfo* map, PyArrayObject* I, PyArrayObject* R) {
    const npy_int64 N1 = map->N1;
    const npy_int64 N2 = map->N2;
    const npy_int64 N3 = map->N3;
    const npy_int64 ng1 = map->n1;
    const npy_int64 ng2 = map->n2;
    const npy_int64 ng3 = map->n3;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 stride3 = map->stride3;
    const npy_int64 width1 = map->width1;
    const npy_int64 width2 = map->width2;
    const npy_int64 width3 = map->width3;

    register npy_int64 k = 0; // patch index
    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1+= stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2+= stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3+= stride3) {
                register npy_int64 l = 0; // patch dim
                for (npy_int64 ii1 = 0; ii1 < width1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < width2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < width3; ++ii3) {
                            npy_double* pIij = (npy_double*)PyArray_GETPTR3(I,i1+ii1,i2+ii2,i3+ii3);
                            const npy_double pPkl = *((npy_double*)PyArray_GETPTR2(P,k,l));
                            *pIij += pPkl;
                            l++;
                        } // for ii3
                    }// for ii2
                } // for ii1
                k++;
            } // for g3
        } // for g2
    } // for g1

    for (npy_int64 i1 = 0; i1 < N1; ++i1) {
        for (npy_int64 i2 = 0; i2 < N2; ++i2 ) {
            for (npy_int64 i3 = 0; i3 < N3; ++i3 ) {
                const double r = *((npy_double*)PyArray_GETPTR3(R,i1,i2,i3));
                *((npy_double*)PyArray_GETPTR3(I,i1,i2,i3)) *= r;
            }
        }
    }
}

static PyObject *stitch(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I, *py_R;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllllO!O!",
                         &N1, &N2, &N3, &w1, &w2, &w3, &s1, &s2, &s3,
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_R)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(N1,N2,N3,w1,w2,w3,s1,s2,s3,EXTRACT_EXACT);
    npy_intp dims[3] = {N1,N2,N3};
    py_I = (PyArrayObject*) PyArray_SimpleNew(3,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,&map,py_I,py_R);
    return PyArray_Return(py_I);
}
//
//--------------------------------------------------------
// stitch
//--------------------------------------------------------
//
static PyObject *stitch_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I, *py_R;
    npy_int64 N1,N2,N3,w1,w2,w3,s1,s2,s3;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllllO!O!O!",
                         &N1, &N2, &N3, &w1, &w2, &w3, &s1, &s2, &s3,
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_R)) {
        return NULL;
    }

    mapinfo map = build_mapinfo(N1,N2,N3,w1,w2,w3,s1,s2,s3,EXTRACT_EXACT);
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,&map,py_I,py_R);

    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
// extract
//--------------------------------------------------------
//
int _extract_(PyArrayObject* I, mapinfo* map, PyArrayObject* P) {
    const npy_int64 ng1 = map->n1;
    const npy_int64 ng2 = map->n2;
    const npy_int64 ng3 = map->n3;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 stride3 = map->stride3;
    const npy_int64 width1 = map->width1;
    const npy_int64 width2 = map->width2;
    const npy_int64 width3 = map->width3;


    //#ifdef _OPENMP
    //#pragma omp parallel for
    //#endif
    npy_int64 k = 0;
    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1 += stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3) {
                register npy_int64 l = 0;
                for (npy_int64 ii1 = 0; ii1 < width1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < width2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < width3; ++ii3) {
                            const npy_double* aux = (npy_double*)PyArray_GETPTR3(I, i1+ii1, i2+ii2, i3+ii3);
                            *((npy_double*)PyArray_GETPTR2(P,k,l)) =  *aux;
                            l++;
                        }
                    }
                } // one patch
                k++;
            }
        }
    }
    return 1;
}

static PyObject *extract_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllllO!O!",
                         &N1, &N2, &N3,
                         &w1, &w2, &w3,
                         &s1, &s2, &s3,
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_P)) {
        return NULL;
    }
    mapinfo map = build_mapinfo(N1,N2,N3,w1,w2,w3,s1,s2,s3,EXTRACT_EXACT);

    _extract_(py_I,&map,py_P);
    Py_RETURN_NONE;
}

static PyObject *extract(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    npy_int64 N1, N2, N3, w1, w2, w3, s1, s2, s3;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllllO!",
                         &N1, &N2, &N3,
                         &w1, &w2, &w3,
                         &s1, &s2, &s3,
                         &PyArray_Type, &py_I)) {
        return NULL;
    }
    mapinfo map = build_mapinfo(N1,N2,N3,w1,w2,w3,s1,s2,s3,EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    _extract_(py_I,&map,py_P);
    return PyArray_Return(py_P);
}

//
//*****************************************************************************
//  Python/NumPy -- C adaptor functions
//*****************************************************************************

mapinfo build_mapinfo(const npy_int64 _N1,
                          const npy_int64 _N2,
                          const npy_int64 _N3,
                          const npy_int64 _w1,
                          const npy_int64 _w2,
                          const npy_int64 _w3,
                          const npy_int64 _s1,
                          const npy_int64 _s2,
                          const npy_int64 _s3,
                          const npy_int64 _cov) {
    mapinfo map;
    map.N1 = _N1;
    map.N2 = _N2;
    map.N3 = _N3;
    map.L = _N1*_N2*_N3;
    map.stride1 = _s1;
    map.stride2 = _s2;
    map.stride3 = _s3;
    map.covering = _cov;
    map.width1 = _w1 > _N1 ? _N1 : _w1;
    map.width2 = _w2 > _N2 ? _N2 : _w2;
    map.width3 = _w3 > _N3 ? _N3 : _w3;
    map.n1 = compute_grid_size(map.N1, map.width1, map.stride1, map.covering);
    map.n2 = compute_grid_size(map.N2, map.width2, map.stride2, map.covering);
    map.n3 = compute_grid_size(map.N3, map.width3, map.stride3, map.covering);
    map.m = map.width1 * map.width2 * map.width3;
    map.n = map.n2 * map.n1 * map.n3;
    map.l = map.m * map.n;
    return map;
}


