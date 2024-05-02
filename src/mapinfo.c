
#include <stdlib.h>
#include <stdio.h>
#undef NDEBUG
#include <assert.h>
#include "mapinfo.h"


static mapinfo* pmapinfo_singleton = NULL;

void _linear_to__idx(npy_int64 li, npy_int32* pi1, npy_int32* pi2) {
    const mapinfo* map = _get_mapinfo_();
    assert(li < map->N);
    *pi2 = li % map->N2;
    *pi1 = li / map->N2;
}



npy_int64 __to_linear_idx(const npy_int32 i1, const npy_int32 i2) {
    const mapinfo* map = _get_mapinfo_();
    assert(i1 < map->N1);
    assert(i2 < map->N2);
    return i1*map->N2+i2;
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


const mapinfo* _get_mapinfo_() {
    return pmapinfo_singleton;
}

void _destroy_mapinfo_(void) {
    if (pmapinfo_singleton) {
        if (pmapinfo_singleton->idx_incomplete_patches) {
            free(pmapinfo_singleton->rel_idx_patch_pixels);
        }
        free(pmapinfo_singleton);
    }
    pmapinfo_singleton = NULL;
}

/**
 * this returns a singleton instance to a structure that retains
 * the mapping used by the current problem.
 */
void _init_mapinfo_(const npy_int64 _N1,
                      const npy_int64 _N2,
                      const npy_int64 _w1,
                      const npy_int64 _w2,
                      const npy_int64 _s1,
                      const npy_int64 _s2,
                      PyArrayObject* M,
                      const npy_int64 _cov) {

    if (pmapinfo_singleton  != NULL)  {
        _destroy_mapinfo_();
    }
    pmapinfo_singleton = (mapinfo*) malloc(sizeof(mapinfo));
    // pmap is an alias here
    mapinfo* pmap = pmapinfo_singleton;
    pmap->N1 = _N1;
    pmap->N2 = _N2;
    pmap->N = _N1*_N2;
    pmap->s1 = _s1;
    pmap->s2 = _s2;
    pmap->covering = _cov;
    pmap->m1 = _w1 > _N1 ? _N1 : _w1;
    pmap->m2 = _w2 > _N2 ? _N2 : _w2;
    pmap->n1 = compute_grid_size(pmap->N1, pmap->m1, pmap->s1, pmap->covering);
    pmap->n2 = compute_grid_size(pmap->N2, pmap->m2, pmap->s2, pmap->covering);
    pmap->m = pmap->m1 * pmap->m2;
    pmap->n = pmap->n2 * pmap->n1;
    pmap->l = pmap->m * pmap->n;
    //
    // normalization pseudoimage (as a linear array)
    //
    if (!M) {
        pmap->num_incomplete_patches = 0;
        pmap->idx_incomplete_patches = NULL;
        return;
    }
    npy_bool* pM = (npy_bool*) PyArray_DATA(M);
    const npy_int64 ng1 = pmap->n1;
    const npy_int64 ng2 = pmap->n2;
    const npy_int64 s1 = pmap->s1;
    const npy_int64 s2 = pmap->s2;
    const npy_int64 m1 = pmap->m1;
    const npy_int64 m2 = pmap->m2;

    //
    // relative linear indexes of pixels within patches
    //
    {
        pmap->rel_idx_patch_pixels = (npy_int64*) malloc(sizeof(npy_int64)*pmap->m);
        for (npy_int64 ii1 = 0, k = 0; ii1 < m1; ++ii1) {
            for (npy_int64 ii2 = 0; ii2 < m2; ++ii2, ++k) {
                pmap->rel_idx_patch_pixels[k] = __to_linear_idx( ii1,ii2 );
            } // 2
        } // 1
    }
}
