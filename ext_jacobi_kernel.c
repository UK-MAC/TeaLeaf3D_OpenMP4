#include "ext_chunk.h"
#include <stdlib.h>

/*
 *		JACOBI SOLVER KERNEL
 */

// Entry point for Jacobi initialisation.
void ext_jacobi_kernel_init_(
        const int* chunk,
        double* density,
        double* energy,
        double* u0,
        double* u,
        double* r,
        double* kx,
        double* ky,
        double* kz,
        const int* coefficient,
        double* dt,
        double *rx,
        double *ry,
        double *rz)
{
    if(*coefficient < CONDUCTIVITY && *coefficient < RECIP_CONDUCTIVITY)
    {
        panic(__LINE__, __FILE__, "Coefficient %d is not valid.\n", *coefficient);
    }

#pragma omp target device(_chunk.device_id)
#pragma omp parallel for
    for(int ii = 1; ii < _chunk.z-1; ++ii)
    {
        for(int jj = 1; jj < _chunk.y-1; ++jj)
        {
            for(int kk = 1; kk < _chunk.x-1; ++kk)
            {
                const int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                double temp = energy[index]*density[index];
                u0[index] = temp;
                u[index] = temp;
            }
        }
    }

#pragma omp target device(_chunk.device_id)
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < _chunk.z-1; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-1; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-1; ++kk)
            {
                const int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                double densityCentre = (*coefficient == CONDUCTIVITY) 
                    ? density[index] : 1.0/density[index];
                double densityLeft = (*coefficient == CONDUCTIVITY) 
                    ? density[index-1] : 1.0/density[index-1];
                double densityDown = (*coefficient == CONDUCTIVITY) 
                    ? density[index-_chunk.x] : 1.0/density[index-_chunk.x];
                double densityBack = (*coefficient == CONDUCTIVITY) 
                    ? density[index-_chunk.page] : 1.0/density[index-_chunk.page];

                kx[index] = *rx*(densityLeft+densityCentre)/(2.0*densityLeft*densityCentre);
                ky[index] = *ry*(densityDown+densityCentre)/(2.0*densityDown*densityCentre);
                kz[index] = *rz*(densityBack+densityCentre)/(2.0*densityBack*densityCentre);
            }
        }
    }
}

// Entry point for Jacobi solver main method.
void ext_jacobi_kernel_solve_(
        const int* chunk,
        double* kx,
        double* ky,
        double* kz,
        double* u0,
        double* u,
        double* r,
        double* error)
{
#pragma omp target device(_chunk.device_id)
#pragma omp parallel for
    for(int ii = 0; ii < _chunk.z; ++ii)
    {
        for(int jj = 0; jj < _chunk.y; ++jj)
        {
            for(int kk = 0; kk < _chunk.x; ++kk)
            {
                const int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                r[index] = u[index];	
            }
        }
    }

    double err=0.0;
#pragma omp target device(_chunk.device_id)
#pragma omp parallel for reduction(+:err)
    for(int ii = HALO_PAD; ii < _chunk.z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < _chunk.y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < _chunk.x-HALO_PAD; ++kk)
            {
                const int index = ii*_chunk.x*_chunk.y+jj*_chunk.x+kk;
                u[index] = (u0[index] 
                        + (kx[index+1]*r[index+1] + kx[index]*r[index-1])
                        + (ky[index+_chunk.x]*r[index+_chunk.x] + ky[index]*r[index-_chunk.x])
                        + (kz[index+_chunk.page]*r[index+_chunk.page] + kz[index]*r[index-_chunk.page]))
                    / (1.0 + (kx[index]+kx[index+1])
                            + (ky[index]+ky[index+_chunk.x])
                            + (kz[index]+kz[index+_chunk.page]));

                err += fabs(u[index]-r[index]);
            }
        }
    }

    *error = err;
}

