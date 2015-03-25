#pragma once
#include <Snow/Cuda/CUDAMatrix.h>

struct NodeCache
{
	enum Offset { R, AR, P, AP, V, DF };

	// Data used by Conjugate Residual Method
	CUDAVec3 r;
	CUDAVec3 Ar;
	CUDAVec3 p;
	CUDAVec3 Ap;
	CUDAVec3 v;
	CUDAVec3 df;
	double scratch;
	__host__ __device__ CUDAVec3& operator [] ( Offset i )
	{
		switch ( i ) {
		case R: return r;
		case AR: return Ar;
		case P: return p;
		case AP: return Ap;
		case V: return v;
		case DF: return df;
		}
		return r;
	}

	__host__ __device__ CUDAVec3 operator [] ( Offset i ) const
	{
		switch ( i ) {
		case R: return r;
		case AR: return Ar;
		case P: return p;
		case AP: return Ap;
		case V: return v;
		case DF: return df;
		}
		return r;
	}

};

struct SnowParticleCache
{
	// Data used during initial node computations
	CUDAMat3 *sigmas;

	// Data used during implicit node velocity update
	CUDAMat3 *Aps;
	CUDAMat3 *FeHats;
	CUDAMat3 *ReHats;
	CUDAMat3 *SeHats;
	CUDAMat3 *dFs;
};