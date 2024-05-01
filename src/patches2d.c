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
    EXTRACT_EXACT =0, ///< only extract2d patches which contain true pixels, possibly leaving bordering pixels out
    EXTRACT_FULL, ///<  extract2d patches so that whole image is covered, extrapolating border pixels as needed
} extract_t;

typedef struct mapinfo2d_s {
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
} mapinfo2d;

mapinfo2d build_mapinfo2d(const npy_int64 _M,
                          const npy_int64 _N,
                          const npy_int64 _w,
                          const npy_int64 _s,
                          const npy_int64 _cov);


/// Python adaptors
static PyObject *create_patches2d_matrix(PyObject* self, PyObject* args);
static PyObject *create_norm2d_matrix   (PyObject* self, PyObject* args);
static PyObject *extract2d              (PyObject *self, PyObject *args);
static PyObject *extract2d_to           (PyObject *self, PyObject *args);
static PyObject *stitch2d               (PyObject *self, PyObject *args);
static PyObject *stitch2d_to            (PyObject *self, PyObject *args);
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
    { "create_patches2d_matrix", create_patches2d_matrix, METH_VARARGS, "Creates a matrix for allocating patches."},
    { "create_norm2d_matrix", create_norm2d_matrix, METH_VARARGS, "Creates a normalization matrix for using when stitch2ding. Only one such matrix is needed for each combination of signal dimensions, width and stride ."},
    { "extract2d", extract2d, METH_VARARGS, "Extract2ds patches from a signal to a new patches matrix"},
    { "extract2d_to", extract2d_to, METH_VARARGS, "Extract2ds patches from a signal to a preallocated patches matrix."},
    { "stitch2d", stitch2d, METH_VARARGS, "Stitch2des patches into a new signal.."},
    { "stitch2d_to", stitch2d_to, METH_VARARGS, "Stitch2des patches into a preallocated signal."},
    { "pad", pad, METH_VARARGS, "Increases signal dimension (at the ends of the dimensions) so that an exact number of patches of the given width and stride fit in it."},
    { NULL, NULL, 0, NULL } /* Sentinel */
};
//
//--------------------------------------------------------
// module initialization
//--------------------------------------------------------
//
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT,
				     "patches2d",
				     "2D patch mapping, half precision",
				     -1, methods};

PyMODINIT_FUNC PyInit_patches2d(void) {
  Py_Initialize();
  return PyModule_Create(&module);
}

//
//--------------------------------------------------------
// compute size of patches grid
//--------------------------------------------------------
//
static npy_int64 compute_grid_size(const npy_int64 size, const npy_int64 width, const npy_int64 stride, unsigned extract2d_type) {
    // return M, the number of patches to be extract2ded along a dimension
    // EXTRACT_EXACT: last index (li) must be < m so
    // li = (M-1)*stride + width -1 <= m -1 => M = floor [ (m - width + stride) / stride ]
    // EXTRACT_FULL: first index of last patch (fi) must be < m so
    // fi = (M-1)*stride  <= m - 1 => stride*M <= m + stride - 1 => M = floor [(m -1 + stride) / stride ]
    return extract2d_type == EXTRACT_EXACT ? (size+stride-width)/stride : (size + stride - 1)/stride;
}
//
//--------------------------------------------------------
// create patches matrix
//--------------------------------------------------------
//
static PyObject *create_patches2d_matrix(PyObject *self, PyObject *args) {
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

    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
// create normalization matrix
//--------------------------------------------------------
//
static PyObject *create_norm2d_matrix(PyObject *self, PyObject *args) {
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

    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);
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
    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);
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
// stitch2d
//--------------------------------------------------------
//
void _stitch2d_(PyArrayObject* P, mapinfo2d* map, PyArrayObject* I, PyArrayObject* R) {
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

static PyObject *stitch2d(PyObject *self, PyObject *args) {
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

    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {M,N};
    py_I = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_I,0);
    _stitch2d_(py_P,&map,py_I,py_R);
    return PyArray_Return(py_I);
}
//
//--------------------------------------------------------
// stitch2d
//--------------------------------------------------------
//
static PyObject *stitch2d_to(PyObject *self, PyObject *args) {
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

    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);
    PyArray_FILLWBYTE(py_I,0);
    _stitch2d_(py_P,&map,py_I,py_R);

    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
// extract2d
//--------------------------------------------------------
//
int _extract2d_(PyArrayObject* I, mapinfo2d* map, PyArrayObject* P) {
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

static PyObject *extract2d_to(PyObject *self, PyObject *args) {
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
    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);

    _extract2d_(py_I,&map,py_P);
    Py_RETURN_NONE;
}

static PyObject *extract2d(PyObject *self, PyObject *args) {
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
    mapinfo2d map = build_mapinfo2d(M,N,w,s,EXTRACT_EXACT);
    npy_intp dims[2] = {map.n,map.m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    _extract2d_(py_I,&map,py_P);
    return PyArray_Return(py_P);
}

//
//*****************************************************************************
//  Python/NumPy -- C adaptor functions
//*****************************************************************************

mapinfo2d build_mapinfo2d(const npy_int64 _M,
                          const npy_int64 _N,
                          const npy_int64 _w,
                          const npy_int64 _s,
                          const npy_int64 _cov) {
    mapinfo2d map;
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


