#pragma once
#include "Grid.h"
#include <Snow/Cuda/CUDAVector.h>

struct Node
{  
	float mass;
	CUDAVec3 velocity;
	CUDAVec3 velocityChange; // v_n+1 - v_n (store this value through steps 4,5,6)
	CUDAVec3 force;
	Node() : mass(0.f), velocity(0,0,0), force(0,0,0) {}
};