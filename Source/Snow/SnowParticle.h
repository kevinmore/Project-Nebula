#pragma once
#include "SnowMaterial.h"
#include <Snow/Cuda/CUDAMatrix.h>

struct SnowParticle
{
	CUDAVec3 position;
	CUDAVec3 velocity;
	float mass;
	float volume;
	CUDAMat3 elasticF;
	CUDAMat3 plasticF;
	SnowMaterial material;

	__host__ __device__ SnowParticle()
	{
		position = CUDAVec3( 0.f, 0.f, 0.f );
		velocity = CUDAVec3( 0.f, 0.f, 0.f );
		mass = 1e-6;
		volume = 1e-9;
		elasticF = CUDAMat3( 1.f );
		plasticF = CUDAMat3( 1.f );
		material = SnowMaterial(); // default
	}
};