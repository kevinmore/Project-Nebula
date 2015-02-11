#pragma once
#include "EPAPolytope.h"

class GJKSimplex;
class CollisionObject;
class NarrowPhaseCollisionFeedback;
class EPASolver
{
public:
	EPASolver(){}
	~EPASolver(){}

	bool computePenetrationDepthAndContactPoints(const GJKSimplex& simplex, CollisionObject& objA, CollisionObject& objB, 
		vec3& v, NarrowPhaseCollisionFeedback& pCollisionInfo, int maxIteration = 30);

private:
	EPAPolytope m_Polytope;
};

