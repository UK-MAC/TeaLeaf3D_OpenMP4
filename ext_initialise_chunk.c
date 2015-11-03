#include "ext_chunk.h"

/*
 * 		INITIALISE CHUNK KERNEL
 * 		Initialises the chunk's mesh data.
 */

// Extended kernel for the chunk initialisation
void ext_initialise_chunk_( 
		int* chunk,
		double* xMin,
		double* yMin,
		double* zMin,
		double* dx,
		double* dy,
		double* dz,
		double* cellX,
		double* cellY,
		double* cellZ,
		double* vertexX,
		double* vertexY,
		double* vertexZ,
		double* volume,
		double* xArea,
		double* yArea,
		double* zArea)
{
	for(int ii = 0; ii != _chunk.x+1; ++ii)
	{
		vertexX[ii]= *xMin+(*dx)*(ii-HALO_PAD);
	}

	for(int ii = 0; ii != _chunk.y+1; ++ii)
	{
		vertexY[ii] = *yMin+(*dy)*(ii-HALO_PAD);
	}

	for(int ii = 0; ii != _chunk.z+1; ++ii)
	{
		vertexZ[ii] = *zMin+(*dz)*(ii-HALO_PAD);
	}

	for(int ii = 0; ii != _chunk.x; ++ii)
	{
		cellX[ii] = 0.5*(vertexX[ii]+vertexX[ii+1]);
	}

	for(int ii = 0; ii != _chunk.y; ++ii)
	{
		cellY[ii] = 0.5*(vertexY[ii]+vertexY[ii+1]);
	}

	for(int ii = 0; ii != _chunk.z; ++ii)
	{
		cellZ[ii] = 0.5*(vertexZ[ii]+vertexZ[ii+1]);
	}

	for(int ii = 0; ii != _chunk.x*_chunk.y*_chunk.z; ++ii)
	{
		volume[ii] = (*dx)*(*dy)*(*dz);
		xArea[ii] = (*dy)*(*dz);
		yArea[ii] = (*dx)*(*dz);
		zArea[ii] = (*dx)*(*dy);
	}
}

