#include "ext_chunk.h"
#include <stdlib.h>

#define CELL_DATA 1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4
#define Z_FACE_DATA 5
#define WARP_SIZE 32.0

// Packs left data into buffer.
void pack_left(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(from: buffer[:_chunk.innerZ*_chunk.innerY*depth])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < HALO_PAD+depth; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-HALO_PAD)*depth + 
                    (ii-HALO_PAD)*depth*_chunk.innerY;
                buffer[bufIndex] = field[ii*y*x+jj*x+kk];
            }
        }
    }
}

// Packs right data into buffer.
void pack_right(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(from: buffer[:_chunk.innerZ*_chunk.innerY*depth])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = x-HALO_PAD-depth; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-(x-HALO_PAD-depth)) + (jj-HALO_PAD)*depth + 
                    (ii-HALO_PAD)*depth*_chunk.innerY;
                buffer[bufIndex] = field[ii*y*x+jj*x+kk];
            }
        }
    }
}

// Packs top data into buffer.
void pack_top(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(from: buffer[:_chunk.innerZ*depth*_chunk.innerX])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = y-HALO_PAD-depth; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-(y-HALO_PAD-depth))*_chunk.innerX 
                    + (ii-HALO_PAD)*depth*_chunk.innerX;
                buffer[bufIndex] = field[ii*y*x+jj*x+kk];
            }
        }
    }
}

// Packs bottom data into buffer.
void pack_bottom(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(from: buffer[:_chunk.innerZ*depth*_chunk.innerX])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < HALO_PAD+depth; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-HALO_PAD)*_chunk.innerX + 
                    (ii-HALO_PAD)*depth*_chunk.innerX;
                buffer[bufIndex] = field[ii*y*x+jj*x+kk];
            }
        }
    }
}

// Packs front data into buffer.
void pack_front(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(from: buffer[:depth*_chunk.innerY*_chunk.innerX])
#pragma omp parallel for
    for(int ii = z-HALO_PAD-depth; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-HALO_PAD)*_chunk.innerX 
                    + (ii-(z-HALO_PAD-depth))*_chunk.innerX*_chunk.innerY;

                buffer[bufIndex] = field[ii*y*x+jj*x+kk];
            }
        }
    }
}

// Packs back data into buffer.
void pack_back(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(from: buffer[:depth*_chunk.innerY*_chunk.innerX])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < HALO_PAD+depth; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-HALO_PAD)*_chunk.innerX 
                    + (ii-HALO_PAD)*_chunk.innerX*_chunk.innerY;
                buffer[bufIndex] = field[ii*y*x+jj*x+kk];
            }
        }
    }
}

// Unpacks left data from buffer.
void unpack_left(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(to: buffer[:_chunk.innerZ*_chunk.innerY*depth])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD-depth; kk < HALO_PAD; ++kk)
            {
                int bufIndex = (kk-(HALO_PAD-depth)) + (jj-HALO_PAD)*depth + 
                    (ii-HALO_PAD)*depth*_chunk.innerY;
                field[ii*y*x+jj*x+kk] = buffer[bufIndex];
            }
        }
    }
}

// Unpacks right data from buffer.
void unpack_right(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(to: buffer[:_chunk.innerZ*_chunk.innerY*depth])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = x-HALO_PAD; kk < x-HALO_PAD+depth; ++kk)
            {
                int bufIndex = (kk-(x-HALO_PAD)) + (jj-HALO_PAD)*depth + 
                    (ii-HALO_PAD)*depth*_chunk.innerY;
                field[ii*y*x+jj*x+kk] = buffer[bufIndex];
            }
        }
    }
}

// Unpacks top data from buffer.
void unpack_top(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(to: buffer[:_chunk.innerZ*depth*_chunk.innerX])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = y-HALO_PAD; jj < y-HALO_PAD+depth; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-(y-HALO_PAD))*_chunk.innerX + 
                    (ii-HALO_PAD)*depth*_chunk.innerX;
                field[ii*y*x+jj*x+kk] = buffer[bufIndex];
            }
        }
    }
}

// Unpacks bottom data from buffer.
void unpack_bottom(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(to: buffer[:_chunk.innerZ*depth*_chunk.innerX])
#pragma omp parallel for
    for(int ii = HALO_PAD; ii < z-HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD-depth; jj < HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-(HALO_PAD-depth))*_chunk.innerX + 
                    (ii-HALO_PAD)*depth*_chunk.innerX;
                field[ii*y*x+jj*x+kk] = buffer[bufIndex];
            }
        }
    }
}

// Unpacks front data from buffer.
void unpack_front(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(to: buffer[:depth*_chunk.innerY*_chunk.innerX])
#pragma omp parallel for
    for(int ii = z-HALO_PAD; ii < z-HALO_PAD+depth; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-HALO_PAD)*_chunk.innerX + 
                    (ii-(z-HALO_PAD))*_chunk.innerX*_chunk.innerY;
                field[ii*y*x+jj*x+kk] = buffer[bufIndex];
            }
        }
    }
}

// Unpacks back data from buffer.
void unpack_back(
        const int x,
        const int y,
        const int z,
        double* field,
        double* buffer,
        const int depth,
        const int offload)
{
#pragma omp target device(_chunk.device_id) if(offload) \
    map(to: buffer[:depth*_chunk.innerY*_chunk.innerX])
#pragma omp parallel for
    for(int ii = HALO_PAD-depth; ii < HALO_PAD; ++ii)
    {
        for(int jj = HALO_PAD; jj < y-HALO_PAD; ++jj)
        {
            for(int kk = HALO_PAD; kk < x-HALO_PAD; ++kk)
            {
                int bufIndex = (kk-HALO_PAD) + (jj-HALO_PAD)*_chunk.innerX + 
                    (ii-(HALO_PAD-depth))*_chunk.innerX*_chunk.innerY;
                field[ii*y*x+jj*x+kk] = buffer[bufIndex];
            }
        }
    }
}

typedef void (*PackKernel)(int,int,int,double*,double*,int,int);

// Either packs or unpacks data from/to buffers.
void pack_unpack_kernel(
        double* density,
        double* energy0,
        double* energy,
        double* u,
        double* p,
        double* sd,
        const int* fields,
        const int* offsets,
        const int depth,
        const int face,
        double* buffer,
        const int pack,
        const int offload)
{
    int exchanges = 0;
    for(int ii = 0; ii != NUM_FIELDS; ++ii)
    {
        exchanges += fields[ii];
    }

    if(exchanges < 1) return;

    PackKernel kernel;

    switch(face)
    {
        case CHUNK_LEFT:
            kernel = pack ? pack_left : unpack_left;
            break;
        case CHUNK_RIGHT:
            kernel = pack ? pack_right : unpack_right;
            break;
        case CHUNK_TOP:
            kernel = pack ? pack_top : unpack_top;
            break;
        case CHUNK_BOTTOM:
            kernel = pack ? pack_bottom : unpack_bottom;
            break;
        case CHUNK_FRONT:
            kernel = pack ? pack_front : unpack_front;
            break;
        case CHUNK_BACK:
            kernel = pack ? pack_back : unpack_back;
            break;
        default:
            panic(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
    }

    for(int ii = 0; ii < NUM_FIELDS; ++ii)
    {
        if(fields[ii])
        {
            double* deviceField = NULL;
            switch(ii+1)
            {
                case FIELD_DENSITY:
                    deviceField = density;
                    break;
                case FIELD_ENERGY0:
                    deviceField = energy0;
                    break;
                case FIELD_ENERGY1:
                    deviceField = energy;
                    break;
                case FIELD_U:
                    deviceField = u;
                    break;
                case FIELD_P:
                    deviceField = p;
                    break;
                case FIELD_SD:
                    deviceField = sd;
                    break;
                default:
                    panic(__LINE__,__FILE__, "Incorrect field provided: %d.\n", ii+1);
            }

            kernel(_chunk.x, _chunk.y, _chunk.z, deviceField, buffer+offsets[ii], depth, offload);
        }
    }
}

// Entry point for packing messages.
void ext_pack_message_(
        const int* chunk,
        double* density,
        double* energy0,
        double* energy,
        double* u,
        double* p,
        double* sd,
        const int* fields,
        const int* offsets,
        const int* depth,
        const int* face,
        double* buffer,
        const int* offload)
{
    pack_unpack_kernel(
            density, energy0, energy, u, p, sd,
            fields, offsets, *depth, *face, buffer, 1, *offload);
}

// Entry point for unpacking messages.
void ext_unpack_message_(
        const int* chunk,
        double* density,
        double* energy0,
        double* energy,
        double* u,
        double* p,
        double* sd,
        const int* fields,
        const int* offsets,
        const int* depth,
        const int* face,
        double* buffer,
        const int* offload)
{
    pack_unpack_kernel(
            density, energy0, energy, u, p, sd,
            fields, offsets, *depth, *face, buffer, 0, *offload);
}

