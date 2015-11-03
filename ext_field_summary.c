#include "ext_chunk.h"

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */	

// Entry point for field summary method.
void ext_field_summary_kernel_(
        const int* chunk,
        double* volume,
        double* density,
        double* energy0,
        double* u,
        double* volOut,
        double* massOut,
        double* ieOut,
        double* tempOut)
{
    double vol = 0.0;
    double ie = 0.0;
    double temp = 0.0;
    double mass = 0.0;

#pragma omp parallel for reduction(+:vol,ie,temp,mass)
    for(int ii = HALO_PAD; ii < _chunk.z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-HALO_PAD; ++kk)
            {
                int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                double cellVol = volume[index];
                double cellMass = cellVol*density[index];
                vol += cellVol;
                mass += cellMass;
                ie += cellMass*energy0[index];
                temp += cellMass*u[index];
            }
        }
    }

    *volOut = vol;
    *ieOut = ie;
    *tempOut = temp;
    *massOut = mass;
}
