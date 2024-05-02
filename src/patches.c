#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#if 1
typedef npy_float32 sample_t;
#define  SAMPLE_TYPE_ID NPY_FLOAT32
#else
typedef npy_float64 sample_t;
#define  SAMPLE_TYPE_ID NPY_FLOAT64
#endif

#include <math.h>
#include "mmap.h"
#include "mapinfo.h"
///
/// Contains all the relevant parameters  about the patch decomposition procedure.
///


/// Python adaptors
static PyObject *init_mapinfo              (PyObject* self, PyObject* args);
static PyObject *destroy_mapinfo           (PyObject* self, PyObject* args);
static PyObject *create_patches_matrix_mmap(PyObject* self, PyObject* args);
static PyObject *create_patches_matrix_mmap(PyObject* self, PyObject* args);
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
    { "init_mapinfo",               init_mapinfo, METH_VARARGS, "bla"},
    { "destroy_mapinfo",            destroy_mapinfo, METH_NOARGS, "bla"},
    { "create_patches_matrix", create_patches_matrix, METH_VARARGS, "Creates a matrix for allocating patches."},
    { "create_norm_matrix", create_norm_matrix, METH_VARARGS, "Creates a normalization matrix for using when stitching. Only one such matrix is needed for each combination of signal dimensions, width and stride ."},
    { "create_patches_matrix_mmap", create_patches_matrix_mmap, METH_NOARGS,"."},
    { "extract", extract, METH_VARARGS, "Extracts patches from a signal to a new patches matrix"},
    { "extract_to", extract_to, METH_VARARGS, "Extracts patches from a signal to a preallocated patches matrix."},
    { "stitch", stitch, METH_VARARGS, "Stitches patches into a new signal.."},
    { "stitch_to", stitch_to, METH_VARARGS, "Stitches patches into a preallocated signal."},
    { "pad", pad, METH_VARARGS, "Increases signal dimension (at the ends of the dimensions) so that an exact number of patches of the given width and stride fit in it."},
    { NULL, NULL, 0, NULL } /* Sentinel */
};
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT,
				     "patches",
				     "Patch mapping, full precision",
				     -1, methods};

PyMODINIT_FUNC PyInit_patches(void) {
  Py_Initialize();
  return PyModule_Create(&module);
}

//---------------------------------------------------------------------------------------
// mapping information; instantiated only once
//---------------------------------------------------------------------------------------

static PyObject *init_mapinfo(PyObject *self, PyObject *args) {
    PyArrayObject* pM;
    npy_int64 N1,N2,w1,w2,s1,s2,cov;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "llllllO!l",&N1,&N2,&w1,&w2,&s1,&s2,&PyArray_Type,&pM,&cov)) {
        return NULL;
    }
    _init_mapinfo_(N1,N2,w1,w2,s1,s2,pM,cov);
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------

static PyObject *destroy_mapinfo(PyObject *self, PyObject *args) {
    _destroy_mapinfo_();
    Py_RETURN_NONE;
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
    // Parse arguments.
    if(!PyArg_ParseTuple(args,"")) {
        return NULL;
    }
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[2] = {pmap->n,pmap->m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    return PyArray_Return(py_P);
}

static PyObject *create_patches_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[2] = {pmap->n,pmap->m};
    npy_int64 size = pmap->n*pmap->m*sizeof(sample_t);
    void* data = mmap_alloc(size);
    py_P = (PyArrayObject*) PyArray_SimpleNewFromData(2,&dims[0],SAMPLE_TYPE_ID, data);
    return PyArray_Return(py_P);
}

//
//--------------------------------------------------------
// create normalization matrix
//--------------------------------------------------------
//
static PyObject *create_norm_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    // Parse arguments.
    if(!PyArg_ParseTuple(args,"")) {
        return NULL;
    }
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[2] = {pmap->N1,pmap->N2};
    py_R = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_R,0);
    const npy_int64 mg = pmap->n1;
    const npy_int64 ng = pmap->n2;
    const npy_int64 s1 = pmap->s1;
    const npy_int64 s2 = pmap->s2;
    const npy_int64 m1 = pmap->m1;
    const npy_int64 m2 = pmap->m2;

    for (npy_int64 ig = 0, i = 0, k = 0; ig < mg; ++ig, i += s1) { // k = patch index
        for (npy_int64 jg = 0, j = 0; jg < ng; ++jg, ++k, j += s2) {
            for (npy_int64 ii = 0, l = 0; ii < m1; ++ii) {
                for (npy_int64 jj = 0; jj < m2; ++l, ++jj) { // l = dimension within patch
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
    npy_int64 M, N;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_I)) {
        return NULL;
    }
    M = PyArray_DIM(py_I,0);
    N = PyArray_DIM(py_I,1);
    //
    // compute dimensions of padded image
    //
    const mapinfo* map = _get_mapinfo_();
    npy_int64 M2 = map->s1*(map->n1-1) + map->m1;
    npy_int64 N2 = map->s2*(map->n2-1) + map->m2;
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
void _stitch_(PyArrayObject* P, PyArrayObject* I, PyArrayObject* R) {
    const mapinfo* map = _get_mapinfo_();
    const npy_int64 M = map->N1;
    const npy_int64 N = map->N2;
    const npy_int64 mg = map->n1;
    const npy_int64 ng = map->n2;
    const npy_int64 s1 = map->s1;
    const npy_int64 s2 = map->s2;
    const npy_int64 m1 = map->m1;
    const npy_int64 m2 = map->m2;

    npy_int64 k = 0;
    npy_int64 i = 0;
    for (npy_int64 ig = 0; ig < mg; ++ig) {
        for (npy_int64 jg = 0, j = 0; jg < ng; ++k, ++jg, j += s2) {
            for (npy_int64 ii = 0, l = 0; ii < m1; ++ii) {
                for (npy_int64 jj = 0; jj < m2; ++l, ++jj) {
                    npy_double* pIij = (npy_double*)PyArray_GETPTR2(I,i+ii,j+jj);
                    *pIij += *((npy_double*)PyArray_GETPTR2(P,k,l));
                } // for ii
            }// for jj
        } // for jg
        i+= s1;
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
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_R)) {
        return NULL;
    }

    const mapinfo* map = _get_mapinfo_();
    const npy_int64 M = map->N1;
    const npy_int64 N = map->N2;
    npy_intp dims[2] = {M,N};
    py_I = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,py_I,py_R);
    return PyArray_Return(py_I);
}
//
//--------------------------------------------------------
// stitch
//--------------------------------------------------------
//
static PyObject *stitch_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I, *py_R;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_R)) {
        return NULL;
    }
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,py_I,py_R);

    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
// extract
//--------------------------------------------------------
//
int _extract_(PyArrayObject* I, PyArrayObject* P) {
    const mapinfo* map = _get_mapinfo_();
    const npy_int64 mg = map->n1;
    const npy_int64 ng = map->n2;
    const npy_int64 s1 = map->s1;
    const npy_int64 s2 = map->s2;
    const npy_int64 m1 = map->m1;
    const npy_int64 m2 = map->m2;

    npy_int64 k = 0, i = 0;
    //#ifdef _OPENMP
    //#pragma omp parallel for
    //#endif
    for (npy_int64 ig = 0; ig < mg; ++ig) { // k = patch index
        for (npy_int64 jg = 0, j = 0; jg < ng; ++jg, ++k, j += s2) {
            for (npy_int64 ii = 0, l = 0; ii < m1; ++ii) {
                for (npy_int64 jj = 0; jj < m2; ++l, ++jj) { // l = dimension within patch
                    const npy_double* aux = (npy_double*)PyArray_GETPTR2(I, i+ii, j+jj);
                    *((npy_double*)PyArray_GETPTR2(P,k,l)) =  *aux;
                } // for ii
            }// for jj
        } // for jg
        i += s1;
    } // for ig

    return 1;
}

static PyObject *extract_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &py_I, &PyArray_Type, &py_P)){
        return NULL;
    }
    _extract_(py_I,py_P);
    Py_RETURN_NONE;
}

static PyObject *extract(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_I
                        )
      ) {
        return NULL;
    }
    const mapinfo* map = _get_mapinfo_();
    npy_intp dims[2] = {map->n,map->m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],NPY_DOUBLE);
    _extract_(py_I,py_P);
    return PyArray_Return(py_P);
}
