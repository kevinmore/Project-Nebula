#pragma once
#include <Utility/Math.h>
#include "GJKSolver.h"

using namespace Math;

class ICollider;
class GJKSolver;
class NarrowPhaseCollisionFeedback
{
public:
	NarrowPhaseCollisionFeedback() : pObjA(NULL), pObjB(NULL), bIntersect(false), penetrationDepth(0) {}

	NarrowPhaseCollisionFeedback(ICollider* a, ICollider* b, bool intersect, const vec3& pa, const vec3& pb, float depth) 
		: pObjA(a), pObjB(b), bIntersect(intersect), 
		contactPntALocal(pa), contactPntBLocal(pb), 
		penetrationDepth(depth) {}

	NarrowPhaseCollisionFeedback(ICollider* a, ICollider* b) 
		: pObjA(a), pObjB(b), bIntersect(false), 
		contactPntALocal(Vector3::ZERO),
		contactPntBLocal(Vector3::ZERO),
		penetrationDepth(0), proximityDistance(0) {}

	ICollider* pObjA;
	ICollider* pObjB;
	bool bIntersect;
	vec3 contactPntALocal; // closest point in object A in local space of object A
	vec3 contactPntBLocal; // closest point in object B in local space of object B
	vec3 contactPntAWorld; // closest point in object A in world space of object A
	vec3 contactPntBWorld; // closest point in object B in world space of object B

	vec3 impulseDirectionOnALocal; // impulse direction in local space of object A
	vec3 impulseDirectionOnBLocal; // impulse direction in local space of object B

	float proximityDistance; 
	float penetrationDepth; // must be positive in case bIntersect is true.
};

class NarrowPhaseCollisionDetection
{
public:
	NarrowPhaseCollisionDetection();
	~NarrowPhaseCollisionDetection();

	std::vector<NarrowPhaseCollisionFeedback>& getPairs() { return m_CollisionPairs; }
	const std::vector<NarrowPhaseCollisionFeedback>& getPairs() const { return m_CollisionPairs; }
	void addPair(const NarrowPhaseCollisionFeedback pair) { m_CollisionPairs.push_back(pair); }

	int checkCollisions();

protected:
	GJKSolver m_solver;
	std::vector<NarrowPhaseCollisionFeedback> m_CollisionPairs;
};

