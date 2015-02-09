#pragma once
#include "EPAPolytope.h"

class GJKSimplex;
class CollisionObject;
class NarrowCollisionInfo;
class EPAAlgorithm
{
public:
	EPAAlgorithm();
	~EPAAlgorithm();

	bool ComputePenetrationDepthAndContactPoints(const GJKSimplex& simplex, CollisionObject& objA, CollisionObject& objB, vec3& v, NarrowCollisionInfo* pCollisionInfo, int maxIteration = 30);

private:
	EPAPolytope m_Polytope;
	int IsOriginInTetrahedron(const vec3& p1, const vec3& p2, const vec3& p3, const vec3& p4) const;
};

