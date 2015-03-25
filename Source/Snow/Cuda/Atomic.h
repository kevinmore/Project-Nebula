#pragma once
#include <Utility/CUDAInclude.h>

#include "CUDAVector.h"
#include "CUDAMatrix.h"

__device__ __forceinline__
void atomicAdd( CUDAVec3 *add, const CUDAVec3 &toAdd )
{
    atomicAdd(&(add->x), toAdd.x);
    atomicAdd(&(add->y), toAdd.y);
    atomicAdd(&(add->z), toAdd.z);
}

__device__ __forceinline__
void atomicAdd( CUDAMat3 *add, const CUDAMat3 &toAdd )
{
    atomicAdd(&(add->data[0]), toAdd[0]);
    atomicAdd(&(add->data[1]), toAdd[1]);
    atomicAdd(&(add->data[2]), toAdd[2]);
    atomicAdd(&(add->data[3]), toAdd[3]);
    atomicAdd(&(add->data[4]), toAdd[4]);
    atomicAdd(&(add->data[5]), toAdd[5]);
    atomicAdd(&(add->data[6]), toAdd[6]);
    atomicAdd(&(add->data[7]), toAdd[7]);
    atomicAdd(&(add->data[8]), toAdd[8]);
}
