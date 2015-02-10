#pragma once
#include "EPAPolytope.h"

class GJKSimplex;
class CollisionObject;
class NarrowCollisionInfo;
class EPAAlgorithm
{
public:
	EPAAlgorithm(){}
	~EPAAlgorithm(){}

	bool computePenetrationDepthAndContactPoints(const GJKSimplex& simplex, CollisionObject& objA, CollisionObject& objB, vec3& v, NarrowCollisionInfo* pCollisionInfo, int maxIteration = 30);

private:
	EPAPolytope m_Polytope;
};

