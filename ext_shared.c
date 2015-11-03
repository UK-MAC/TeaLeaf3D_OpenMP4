#include <stdlib.h>
#include <stdio.h>
#include "ext_shared.h"
#include "ext_chunk.h"

// Plots a three-dimensional dat file.
void plot3d(double* buffer, const char* name)
{
	// Open the plot file
	FILE* fp = fopen("plot3d.dat", "wb");
	if(!fp) { printf("Could not open plot file.\n"); }

	double bSum = 0.0;

	for(int ii = 0; ii < _chunk.z; ++ii)
	{
		for(int jj = 0; jj < _chunk.y; ++jj)
		{
			for(int kk = 0; kk < _chunk.x; ++kk)
			{
				double val = buffer[kk+jj*_chunk.x+ii*_chunk.x*_chunk.y];
				fprintf(fp, "%d %d %d %.12E\n", kk, jj, ii, val);
				bSum+=val;
			}
		}
	}

	printf("%s: %.12E\n", name, bSum);
	fclose(fp);
}

// Aborts the application.
void panic(int lineNum, const char* file, const char* format, ...)
{
    printf("\x1b[31m");
    printf("\nError at line %d in %s:", lineNum, file);
    printf("\x1b[0m \n");

    va_list arglist;
    va_start(arglist, format);
    vprintf(format, arglist);
    va_end(arglist);

    exit(1);
}

