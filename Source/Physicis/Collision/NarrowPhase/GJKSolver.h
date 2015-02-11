#pragma once
#include "EPASolver.h"

class Edge;
class Simplex;
class WorldSimulation;
class Transform;
class GJKSolver
{
public:
	GJKSolver(){}
	~GJKSolver(){}

	bool checkCollision(ICollider* objA, ICollider* objB, 
		NarrowPhaseCollisionFeedback& pCollisionInfo, bool bProximity = false);

private:
	// helper function to generate CollisionFeedback
	bool generateCollisionInfo(const ICollider* objA, const ICollider* objB, 
		const Transform &transB2A, const Simplex& simplex, 
		vec3 v, float distSqrd, NarrowPhaseCollisionFeedback& pCollisionInfo) const;

	EPASolver m_EPASolver;
};

