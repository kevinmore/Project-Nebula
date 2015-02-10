#pragma once
#include "EPAAlgorithm.h"

class EPAEdge;
class GJKSimplex;
class WorldSimulation;
class Transform;
class GJKAlgorithm
{
public:
	GJKAlgorithm(){}
	~GJKAlgorithm(){}

	bool checkCollision(CollisionObject& objA, CollisionObject& objB, 
		NarrowCollisionInfo* pCollisionInfo, bool bProximity = false);

private:
	// helper function to generate CollisionInfo
	bool generateCollisionInfo(const CollisionObject& objA, const CollisionObject& objB, 
		const Transform &transB2A, const GJKSimplex& simplex, 
		vec3 v, float distSqrd, NarrowCollisionInfo* pCollisionInfo) const;

	EPAAlgorithm m_EPAAlgorithm;
};

