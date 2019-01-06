import cython
cimport cpython.array

import numpy as np
cimport numpy as np

cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    char GEOSContains_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry*) nogil

cdef GEOSContextHandle_t get_geos_context_handle():
    # Note: This requires that lgeos is defined, so needs to be imported as:
    from shapely.geos import lgeos
    cdef np.uintp_t handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


@cython.boundscheck(False)  # won't check that index is in bounds of array
@cython.wraparound(False) # array[-1] won't work
def contains_cy_insee(np.int64_t[:] array_usrs, np.int64_t[:]array_insee ):
    cdef Py_ssize_t idx
    cdef unsigned int n = array_usrs.size
    cdef unsigned int n_geo = array_insee.size
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] result = np.empty([n,n_geo],dtype=np.uint8)
    cdef GEOSContextHandle_t geos_handle
    cdef GEOSGeometry *geom1
    cdef GEOSGeometry *geom2
    cdef np.uintp_t geos_geom_usrs
    cdef np.uintp_t geos_geom_insee
    geos_h = get_geos_context_handle()
    for idx in xrange(n):
        for idx_geos in xrange(n_geo):
            geos_geom_usrs = array_usrs[idx]
            geom2 = <GEOSGeometry *>geos_geom_usrs
            geos_geom_insee = array_insee[idx_geos]
            geom1 = <GEOSGeometry *> geos_geom_insee
            # Put the result of whether the point is "contained" by the
            # prepared geometry into the result array.
            result[idx][idx_geos] = <np.uint8_t> GEOSContains_r(geos_h, geom1, geom2)
            #GEOSGeom_destroy_r(geos_h, geom2)
    return result.view(dtype=np.bool)

