#include "ext_chunk.h"

/*
 *     SET FIELD KERNEL
 *     Sets energy1 to energy0.
 */    

// Entry point for the the set field method.
void ext_set_field_kernel_(
        const int* chunk,
        double* energy0,
        double* energy)
{
    for(int ii = 0; ii != _chunk.x*_chunk.y*_chunk.z; ++ii)
    {
        energy[ii] = energy0[ii];
    }
}
