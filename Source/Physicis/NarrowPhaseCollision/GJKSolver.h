#pragma once
#include "EPASolver.h"

class EPAEdge;
class GJKSimplex;
class WorldSimulation;
class Transform;
class GJKSolver
{
public:
	GJKSolver(){}
	~GJKSolver(){}

	bool checkCollision(CollisionObject& objA, CollisionObject& objB, 
		NarrowPhaseCollisionFeedback* pCollisionInfo, bool bProximity = false);

private:
	// helper function to generate CollisionInfo
	bool generateCollisionInfo(const CollisionObject& objA, const CollisionObject& objB, 
		const Transform &transB2A, const GJKSimplex& simplex, 
		vec3 v, float distSqrd, NarrowPhaseCollisionFeedback* pCollisionInfo) const;

	EPASolver m_EPAAlgorithm;
};

