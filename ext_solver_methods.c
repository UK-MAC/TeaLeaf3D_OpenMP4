#include "ext_chunk.h"
#include <stdlib.h>

/*
 *		SHARED SOLVER METHODS
 */

// Entry point to copy U.
void ext_solver_copy_u_(
        const int* chunk,
        double* u0,
        double* u)
{
    START_PROFILING;

#pragma omp target device(_chunk.device_id)
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < _chunk.z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-HALO_PAD; ++kk)
            {
                int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                u0[index] = u[index];	
            }
        }
    }

    STOP_PROFILING(__func__);
}

// Entry point for calculating residual.
void ext_calculate_residual_(
        const int* chunk,
        double* u,
        double* u0,
        double* r,
        double* kx,
        double* ky,
        double* kz)
{
    START_PROFILING;

#pragma omp target device(_chunk.device_id)
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < _chunk.z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-HALO_PAD; ++kk)
            {
                int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                const double smvp = SMVP(u);
                r[index] = u0[index] - smvp;
            }
        }
    }

    STOP_PROFILING(__func__);
}

// Entry point for calculating 2norm.
void ext_calculate_2norm_(
        const int* chunk,
        double* buffer,
        double* norm)
{
    START_PROFILING;

    double normTemp = 0.0;
#pragma omp target device(_chunk.device_id)
#pragma omp parallel for reduction(+:normTemp)
    for(int ii = HALO_PAD; ii < _chunk.z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-HALO_PAD; ++kk)
            {
                int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                normTemp += buffer[index]*buffer[index];			
            }
        }
    }

    *norm = normTemp;

    STOP_PROFILING(__func__);
}

// Entry point for finalising solution.
void ext_solver_finalise_(
        const int* chunk,
        double* energy,
        double* density,
        double* u)
{
    START_PROFILING;

#pragma omp target device(_chunk.device_id)
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < _chunk.z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-HALO_PAD; ++kk)
            {
                int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                energy[index] = u[index]/density[index];
            }
        }
    }

    STOP_PROFILING(__func__);
}
