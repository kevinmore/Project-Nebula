#pragma once
#include <Physicis/Geometry/Polytope.h>

class Simplex;
class ICollider;
class NarrowPhaseCollisionFeedback;
class EPASolver
{
public:
	EPASolver(){}
	~EPASolver(){}

	bool computePenetrationDepthAndContactPoints(const Simplex& simplex, ICollider* objA, ICollider* objB, 
		vec3& v, NarrowPhaseCollisionFeedback& pCollisionInfo, int maxIteration = 30);

private:
	Polytope m_Polytope;
};

