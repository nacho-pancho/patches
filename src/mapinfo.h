#ifndef MAPINFO_H
#define MAPINFO_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

///
/// Contains all the relevant parameters  about the patch decomposition procedure.
///
#define CCLIP(x,a,b) ( (x) > (a) ? ( (x) < (b) ? (x) : (b) ) : (a) )
#define UCLIP(x,a) ( (x) < (a) ? (x) : (a)-1 )

typedef npy_int64 idx_t;

typedef enum {
    EXTRACT_EXACT =0, ///< only extract2d patches which contain true pixels, possibly leaving bordering pixels out
    EXTRACT_FULL, ///<  extract2d patches so that whole image is covered, extrapolating border pixels as needed
} extract_t;

typedef struct  {
    npy_char  ndim; ///
    npy_int64 N1; ///< size of signal along dim1; provided
    npy_int64 N2; ///< size of signal along dim2; provided
    npy_int64 N; ///< signal linear length = M*N*T
    npy_int64 s1; ///< stride  along dim1, provided
    npy_int64 s2; ///< stride along dim2, provided
    npy_int64 covering; ///< signal covering strategy, provided
    npy_int64 m1; ///< width of patches along dim1, provided, possibly corrected
    npy_int64 m2; ///< width of patches along dim2, provided, possibly corrected
    npy_int64 m; ///< patch space dimension; computed
    npy_int64 n1; ///< number of patches along dim 2; computed
    npy_int64 n2; ///< number of patches along dim 1; computed
    npy_int64 n; ///< number of patches; computed n1*n2*n3
    npy_int64 l; ///< patches matrix linear length
    npy_int64* rel_idx_patch_pixels; ///< relative linear indexes of pixels within a patch
} mapinfo;


const mapinfo *_get_mapinfo_(void);
void _destroy_mapinfo_(void);

void _init_mapinfo_(const npy_int64 _N1,
                      const npy_int64 _N2,
                      const npy_int64 _w1,
                      const npy_int64 _w2,
                      const npy_int64 _s1,
                      const npy_int64 _s2,
                      PyArrayObject* M,
                      const npy_int64 _cov);

void _linear_to__idx(npy_int64 li, npy_int32* pi1, npy_int32* pi2);
npy_int64 __to_linear_idx(npy_int32 i1, npy_int32 i2);

#endif
