#ifndef __SHARED
#define __SHARED

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "ext_profiler.h"

#define HALO_PAD 2
#define CHUNK_LEFT 1
#define CHUNK_RIGHT 2
#define CHUNK_BOTTOM 3
#define CHUNK_TOP 4
#define CHUNK_BACK 5
#define CHUNK_FRONT 6
#define NUM_FACES 6
#define EXTERNAL_FACE -1

#define FIELD_DENSITY 1
#define FIELD_ENERGY0 2
#define FIELD_ENERGY1 3
#define FIELD_U 4
#define FIELD_P 5
#define FIELD_SD 6
#define NUM_FIELDS 6
#define MAX_DEPTH 2

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

#define SMVP(a) \
	(1.0 + (kx[index+1]+kx[index])\
	 + (ky[index+_chunk.x]+ky[index])\
	 + (kz[index+_chunk.page]+kz[index]))*a[index]\
	 - (kx[index+1]*a[index+1]+kx[index]*a[index-1])\
	 - (ky[index+_chunk.x]*a[index+_chunk.x]+ky[index]*a[index-_chunk.x])\
	 - (kz[index+_chunk.page]*a[index+_chunk.page]+kz[index]*a[index-_chunk.page]);

void plot3d(double* buffer, const char* name);
void panic(int lineNum, const char* file, const char* format, ...);

#endif

