
#include "paco_mapinfo.h"
#include <stdlib.h>
#include <stdio.h>
#undef NDEBUG
#include <assert.h>


static mapinfo* pmapinfo_singleton = NULL;

void _linear_to__idx(npy_int64 li, npy_int32* pi1, npy_int32* pi2, npy_int32* pi3) {
    const mapinfo* map = _get_mapinfo_();
    assert(li < map->N);
    *pi3 = li % map->N3;
    li /= map->N3;
    *pi2 = li % map->N2;
    *pi1 = li / map->N2;
}



npy_int64 __to_linear_idx(const npy_int32 i1, const npy_int32 i2, const npy_int32 i3) {
    const mapinfo* map = _get_mapinfo_();
    assert(i1 < map->N1);
    assert(i2 < map->N2);
    assert(i3 < map->N3);
    return (i1*map->N2+i2)*map->N3+i3;
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
            free(pmapinfo_singleton->idx_incomplete_patches);
            free(pmapinfo_singleton->rel_idx_patch_pixels);
            free(pmapinfo_singleton->idx_affected_pixels);
            free(pmapinfo_singleton->fact_affected_pixels);
            free(pmapinfo_singleton->idx_projected_pixels);
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
                      const npy_int64 _N3,
                      const npy_int64 _w1,
                      const npy_int64 _w2,
                      const npy_int64 _w3,
                      const npy_int64 _s1,
                      const npy_int64 _s2,
                      const npy_int64 _s3,
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
    pmap->N3 = _N3;
    pmap->N = _N1*_N2*_N3;
    pmap->stride1 = _s1;
    pmap->stride2 = _s2;
    pmap->stride3 = _s3;
    pmap->covering = _cov;
    pmap->m1 = _w1 > _N1 ? _N1 : _w1;
    pmap->m2 = _w2 > _N2 ? _N2 : _w2;
    pmap->m3 = _w3 > _N3 ? _N3 : _w3;
    pmap->n1 = compute_grid_size(pmap->N1, pmap->m1, pmap->stride1, pmap->covering);
    pmap->n2 = compute_grid_size(pmap->N2, pmap->m2, pmap->stride2, pmap->covering);
    pmap->n3 = compute_grid_size(pmap->N3, pmap->m3, pmap->stride3, pmap->covering);
    pmap->m = pmap->m1 * pmap->m2 * pmap->m3;
    pmap->n = pmap->n2 * pmap->n1 * pmap->n3;
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
    //printf("decomposotion parameters : %ld x %ld x %ld + %ld + %ld + %ld\n", pmap->m1, pmap->m2, pmap->m3, pmap->stride1, pmap->stride2, pmap->stride3);
    //printf("patch grid size: %ld x %ld x %ld\n", pmap->n1, pmap->n2, pmap->n3);
    //printf("last coordinate: %ld x %ld x %ld\n", 
    //    (pmap->n1-1)*pmap->stride1 + pmap->m1 - 1,
    //    (pmap->n2-1)*pmap->stride2 + pmap->m2 - 1,
    //    (pmap->n3-1)*pmap->stride3 + pmap->m3 - 1);
    //fflush(stdout);
    //printf("total patches: %ld\n", pmap->n);
    const npy_int64 ng1 = pmap->n1;
    const npy_int64 ng2 = pmap->n2;
    const npy_int64 ng3 = pmap->n3;
    const npy_int64 stride1 = pmap->stride1;
    const npy_int64 stride2 = pmap->stride2;
    const npy_int64 stride3 = pmap->stride3;
    const npy_int64 m1 = pmap->m1;
    const npy_int64 m2 = pmap->m2;
    const npy_int64 m3 = pmap->m3;

    //
    // relative linear indexes of pixels within patches
    //
    {
        pmap->rel_idx_patch_pixels = (npy_int64*) malloc(sizeof(npy_int64)*pmap->m);
        for (npy_int64 ii1 = 0, k = 0; ii1 < m1; ++ii1) {
            for (npy_int64 ii2 = 0; ii2 < m2; ++ii2) {
                for (npy_int64 ii3 = 0; ii3 < m3; ++ii3, ++k) {
                    pmap->rel_idx_patch_pixels[k] = __to_linear_idx( ii1,ii2,ii3 );
                    //	printf("%ld: %ld %ld %ld -> %ld\n",k, ii1,ii2,ii3, pmap->rel_idx_patch_pixels[k]);
                    //	npy_int32 aux1, aux2, aux3;
                    //	_linear_to__idx(pmap->rel_idx_patch_pixels[k], &aux1, &aux2, &aux3);
                    //	printf("%ld: %ld -> %d %d %d\n", k, pmap->rel_idx_patch_pixels[k], aux1, aux2, aux3 );
                } // 3
            } // 2
        } // 1
    }
    //
    // missing pixels.
    //
    {
        //
        // 1: count them
        //
        npy_int64 nmis = 0;
        for (npy_int64 i = 0; i < pmap->N; ++i) {
            if (pM[i] != 0)
                nmis++;
        }
        //printf("Total number of missing pixels: %lu\n",nmis);
        pmap->num_missing_pixels = nmis;
        pmap->idx_missing_pixels = (npy_int64*) malloc(sizeof(npy_int64)*pmap->num_missing_pixels);
        //
        //   2: store their indexes 
        //
        for (npy_int64 i = 0, k = 0 ; i < pmap->N; ++i) {
            if (pM[i]) {
                pmap->idx_missing_pixels[k++] = i;
            }
        }
    }
    //
    // incomplete patches patches with missing pixels
    //
    {
        //
        // count them
        // 
        npy_int64 ninc = 0;
        for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1 += stride1) {
            for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
                for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3) {
                    for (npy_int64 ii1 = 0; ii1 < m1; ++ii1) {
                        for (npy_int64 ii2 = 0; ii2 < m2; ++ii2) {
                            for (npy_int64 ii3 = 0; ii3 < m3; ++ii3) {
                                if ( pM[ __to_linear_idx(i1+ii1, i2+ii2, i3+ii3)  ]  ) {
                                    ninc++;
                                    goto incomplete_patch;
                                }
                            } // 3
                        } // 2
                    } // 1
incomplete_patch:
                    {}
                }
            }
        }
        //
        // allocate storage for
        // linear indexes of pixels of incomplete patches;
        // this has the same size as a patches matrix, where each entry contains the
        // correspoding linear index of the patch pixel
        // notice that, as the same pixel appears in various patches, so do their indexes.
        //
        //printf("Incomplete patches  %lu out of %lu\n",ninc,pmap->n);
        pmap->num_incomplete_patches = ninc;
        //printf("Allocating %lu incomplete patch indexes of size %lu bytes each, totalling %lu MB\n",
        //    pmap->num_incomplete_patches, sizeof(npy_int64), (pmap->num_incomplete_patches * sizeof(npy_int64)) >> 20);
        pmap->idx_incomplete_patches = (npy_int64*) malloc(sizeof(npy_int64)*pmap->num_incomplete_patches);
        //
        // fill in idx_incomplete_patch_pixels
        //
        {
            npy_int64 jinc = 0;
            for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1 += stride1) {
                for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
                    for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3) {

                        for (npy_int64 ii1 = 0; ii1 < m1; ++ii1) {
                            for (npy_int64 ii2 = 0; ii2 < m2; ++ii2) {
                                for (npy_int64 ii3 = 0; ii3 < m3; ++ii3) {
                                    if ( ( i1+ii1 < _N1 ) && ( i2+ii2 < _N2 ) && ( i3+ii3 < _N3 ) ) {
                                        if (pM[ __to_linear_idx( i1+ii1, i2+ii2, i3+ii3) ] ) {
                                            pmap->idx_incomplete_patches[jinc++] = __to_linear_idx(i1,i2,i3);
                                            goto incomplete_patch2;
                                        }
                                    }
                                } // 3
                            } // 2
                        } // 1
                        // if we reached this point, we found no missing_pixels pixel in this patch
    incomplete_patch2: {}
    
                    }
                }
            }
        }
    } // end incomplete patch statistics

    //
    // affected pixels indexes and projected pixels
    //
    // affected pixels:
    //
    // These are the *unique* linear indexes of the signal  pixels which
    // are affected_pixels by the estimation process.
    //
    // This information is used to speed up the stitching operation
    // in conjunction with the pre-computed fact_affected_pixels factors
    //
    // what we do here is to create a temporary map
    // with the same size as the input signal.
    // each patch that inculdes at least one missing_pixels pixel has all its 'hits'
    // increased to 1.
    //
    // this will serve three purposes:
    // 2) determine the linear index of those pixels 
    // 2) determine the value of the normalization constant at
    //    those locations (fact_affected_pixels)
    //
    // projected pixels:
    //
    // These are the affected pixels that are actually known in the signal
    // collecting their indexes helps in speeding up the reprojection of
    // the estimated signal.
    //
    {
        npy_uint16* hits = (npy_uint16*) calloc(pmap->N,sizeof(npy_uint16));
        if (hits == NULL) {
            printf("Error allocating auxiliary hits array!\n");
        }
        const npy_int64 m = pmap-> m;
        const npy_int64 num_incomplete = pmap->num_incomplete_patches;
        const npy_int64* pJi = pmap->idx_incomplete_patches;
        const npy_int64* pJri = pmap->rel_idx_patch_pixels;
        // add 1 to each location when a patch refers to it
        for (npy_int64 j = 0; j < num_incomplete; j++)  {
            const npy_int64 lin_off = pJi[j];
            npy_int32 i1,i2,i3;
            _linear_to__idx(lin_off,&i1,&i2,&i3);
            for (npy_int64 r = 0; r < m; r++)  {
                const npy_int64 lin_rel = pJri[r];
                npy_int32 ii1,ii2,ii3;
                _linear_to__idx(lin_rel,&ii1,&ii2,&ii3);
                npy_int64 abs_idx = __to_linear_idx(i1 + ii1, i2 + ii2, i3 + ii3);
                if ( ( i1 + ii1 < _N1 ) && ( i2 + ii2 < _N2 ) && ( i3 + ii3 < _N3 ) ) {
                    hits[ abs_idx ] ++;
                } else {
                    printf("%d %d %d + %d %d %d out of range.\n",i1,i2,i3,ii1,ii2,ii3);
                }
            }
        }
        //
        // count number of non-zeros in hits map
        //
        npy_int64 num_affected_pixels = 0;
        npy_int64 num_projected_pixels = 0;
        for (npy_int64 k = 0; k < pmap->N; ++k) {
            if (hits[k] > 0) { // this pixel is affected
                num_affected_pixels++;
                if (pM[k] == 0) { // but NOT missing
                    num_projected_pixels++;
                }
            }
        }
        pmap->num_affected_pixels = num_affected_pixels;
        pmap->num_projected_pixels = num_projected_pixels;
        //
        // allocate fact_affected_pixels and Jaffected_pixels
        //
        printf("Number of affected pixels = %lu\n",pmap->num_affected_pixels);
        printf("Number of projected pixels = %lu\n",pmap->num_projected_pixels);
        printf("Check  projected (%ld) + missing (%ld) = %ld = affected %ld\n",
                pmap->num_projected_pixels,
                pmap->num_missing_pixels,
                pmap->num_projected_pixels + pmap->num_missing_pixels,
                pmap->num_affected_pixels);

	assert(pmap->num_projected_pixels + pmap->num_missing_pixels ==  pmap->num_affected_pixels);
	
        //printf("Allocating %lu  indexes of size %lu bytes each for a total of %lu MB\n",
        //        pmap->num_affected_pixels, sizeof(npy_int64), (pmap->num_affected_pixels*sizeof(npy_int64)) >> 20);
        pmap->idx_affected_pixels = (npy_int64*) calloc(pmap->num_affected_pixels,sizeof(npy_int64));

        //printf("Allocating %lu normalization constants  of size %lu bytes each for a total of %lu MB\n",
        //        pmap->num_affected_pixels, sizeof(npy_uint16), (pmap->num_affected_pixels*sizeof(npy_uint16)) >> 20);
        pmap->fact_affected_pixels = (npy_uint16*) calloc(pmap->num_affected_pixels,sizeof(npy_uint16));

        //printf("Allocating %lu projection indexes of size %lu bytes each for a total of %lu MB\n",
        //        pmap->num_projected_pixels, sizeof(npy_int64), (pmap->num_projected_pixels*sizeof(npy_int64)) >> 20);
        pmap->idx_projected_pixels = (npy_int64*) calloc(pmap->num_projected_pixels,sizeof(npy_int64));
        //printf("Filling in index info.\n");
        //
        //
        //
        npy_int64 ja = 0;
        npy_int64 jp = 0;
	FILE* fhits = fopen("hits.asc","w");
        for (npy_int64 k = 0; k < pmap->N; ++k) {
	    fprintf(fhits,"%d\n",hits[k]);
            if (hits[k] > 0) {
                pmap->idx_affected_pixels[ja] = k;
                pmap->fact_affected_pixels[ja] = (npy_uint16)hits[k];
                ja++;
                if (pM[k] == 0) { 
                    pmap->idx_projected_pixels[jp++] = k;
                }
            }
        }
	fclose(fhits);
	assert(jp == pmap->num_projected_pixels);
	assert(ja == pmap->num_affected_pixels);
        //printf("Cleanup.\n");
        free(hits);
    } // end projected and affected
}
