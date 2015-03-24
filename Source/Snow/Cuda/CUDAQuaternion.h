#pragma once
#include <Utility/CUDAInclude.h>

struct CUDAQuat
{
    union {
        float data[4];
        struct { float x, y, z, w; };
    };

    __host__ __device__ __forceinline__
    CUDAQuat() { x = 0.f; y = 0.f; z = 0.f; w = 1.f; }

    __host__ __device__ __forceinline__
    CUDAQuat( float ww, float xx, float yy, float zz ) { x = xx; y = yy; z = zz; w = ww; }

    __host__ __device__ __forceinline__
    CUDAQuat( const CUDAQuat &q ) { x = q.x; y = q.y; z = q.z; w = q.w; }

    __host__ __device__ __forceinline__
    CUDAQuat& operator = ( const CUDAQuat &q ) { x = q.x; y = q.y; z = q.z; w = q.w; return *this; }

    __host__ __device__ __forceinline__
    float& operator [] ( int i ) { return data[i]; }

    __host__ __device__ __forceinline__
    float operator [] ( int i ) const { return data[i]; }

    __host__ __device__ __forceinline__
    CUDAQuat& operator *= ( float f ) { x *= f; y *= f; z *= f; w *= f; return *this; }

    __host__ __device__ __forceinline__
    CUDAQuat operator * ( float f ) const { return CUDAQuat( w*f, x*f, y*f, z*f ); }

    __host__ __device__ __forceinline__
    CUDAQuat& operator *= ( const CUDAQuat &q )
    {
        CUDAQuat p(*this);
        w = p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z;
        x = p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y;
        y = p.w*q.y + p.y*q.w + p.z*q.x - p.x*q.z;
        z = p.w*q.z + p.z*q.w + p.x*q.y - p.y*q.x;
        return *this;
    }

    __host__ __device__ __forceinline__
    CUDAQuat operator * ( const CUDAQuat &q ) const
    {
        CUDAQuat p;
        p.w = w*q.w - x*q.x - y*q.y - z*q.z;
        p.x = w*q.x + x*q.w + y*q.z - z*q.y;
        p.y = w*q.y + y*q.w + z*q.x - x*q.z;
        p.z = w*q.z + z*q.w + x*q.y - y*q.x;
        return p;
    }

};
