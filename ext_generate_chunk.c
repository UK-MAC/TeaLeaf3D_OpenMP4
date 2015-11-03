#include "ext_chunk.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


/*
 *		GENERATE CHUNK KERNEL
 *		Sets up the chunk geometry.
 */

// Entry point for the the chunk generation method.
void ext_generate_chunk_(
        const int* chunk,
        double* vertexX,
        double* vertexY,
        double* vertexZ,
        double* cellX,
        double* cellY,
        double* cellZ,
        double* density,
        double* energy0,
        double* u,
        const int* numberOfStates,
        const double* stateDensity,
        const double* stateEnergy,
        const double* stateXMin,
        const double* stateXMax,
        const double* stateYMin,
        const double* stateYMax,
        const double* stateZMin,
        const double* stateZMax,
        const double* stateRadius,
        const int* stateGeometry,
        const int* rectParam,
        const int* circParam,
        const int* pointParam)
{
    for(int ii = 0; ii != _chunk.x*_chunk.y*_chunk.z; ++ii)
    {
        energy0[ii]=stateEnergy[0];
        density[ii]=stateDensity[0];
    }	

    for(int ss = 1; ss < *numberOfStates; ++ss)
    {
        for(int ii = 0; ii != _chunk.z; ++ii) 
        {
            for(int jj = 0; jj != _chunk.y; ++jj) 
            {
                for(int kk = 0; kk != _chunk.x; ++kk) 
                {
                    int applyState = 0;

                    if(stateGeometry[ss] == *rectParam) // Rectangular state
                    {
                        applyState = (
                                vertexX[kk+1] >= stateXMin[ss] && 
                                vertexX[kk] < stateXMax[ss]    &&
                                vertexY[jj+1] >= stateYMin[ss] &&
                                vertexY[jj] < stateYMax[ss] 	 &&
                                vertexZ[ii+1] >= stateZMin[ss] &&
                                vertexZ[ii] < stateZMax[ss]);
                    }
                    else if(stateGeometry[ss] == *circParam) // Circular state
                    {
                        double radius = sqrt(
                                (cellX[kk]-stateXMin[ss])*(cellX[kk]-stateXMin[ss])+
                                (cellY[jj]-stateYMin[ss])*(cellY[jj]-stateYMin[ss])+
                                (cellZ[ii]-stateZMin[ss])*(cellZ[ii]-stateZMin[ss]));

                        applyState = (radius <= stateRadius[ss]);
                    }
                    else if(stateGeometry[ss] == *pointParam) // Point state
                    {
                        applyState = (
                                vertexX[kk] == stateXMin[ss] &&
                                vertexY[jj] == stateYMin[ss] &&
                                vertexZ[ii] == stateZMin[ss]);
                    }

                    // Check if state applies at this vertex, and apply
                    if(applyState)
                    {
                        int index = ii*_chunk.y*_chunk.x+jj*_chunk.x+kk;

                        energy0[index] = stateEnergy[ss];
                        density[index] = stateDensity[ss];
                    }
                }
            }
        }
    }

    for(int ii = 1; ii != _chunk.z-1; ++ii) 
    {
        for(int jj = 1; jj != _chunk.y-1; ++jj) 
        {
            for(int kk = 1; kk != _chunk.x-1; ++kk) 
            {
                int index = ii*_chunk.y*_chunk.x+jj*_chunk.x+kk;
                u[index]=energy0[index]*density[index];
            }
        }
    }
}

