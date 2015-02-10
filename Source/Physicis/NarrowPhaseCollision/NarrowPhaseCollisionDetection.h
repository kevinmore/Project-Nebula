#pragma once
#include <Utility/EngineCommon.h>

class CollisionObject;
class GJKAlgorithm;
class NarrowCollisionInfo
{
public:
	NarrowCollisionInfo() : pObjA(NULL), pObjB(NULL), bIntersect(false), penetrationDepth(0) {}

	NarrowCollisionInfo(CollisionObject* a, CollisionObject* b, bool intersect, const vec3& pa, const vec3& pb, float depth) 
		: pObjA(a), pObjB(b), bIntersect(intersect), 
		witnessPntA(pa), witnessPntB(pb), 
		penetrationDepth(depth) {}

	NarrowCollisionInfo(CollisionObject* a, CollisionObject* b) 
		: pObjA(a), pObjB(b), bIntersect(false), 																				  
		penetrationDepth(0), proximityDistance(0) {}

	CollisionObject* pObjA;
	CollisionObject* pObjB;
	bool bIntersect;
	vec3 witnessPntA; // clostest point in object A in local space of object A
	vec3 witnessPntB; // clostest point in object B in local space of object B
	float proximityDistance; 
	float penetrationDepth; // must be positive in case bIntersect is true.
};

class NarrowPhaseCollisionDetection
{
public:
	NarrowPhaseCollisionDetection();
	~NarrowPhaseCollisionDetection();

	std::vector<NarrowCollisionInfo>& getPairs() { return m_CollisionPairs; }
	const std::vector<NarrowCollisionInfo>& getPairs() const { return m_CollisionPairs; }
	void addPair(const NarrowCollisionInfo pair) { m_CollisionPairs.push_back(pair); }

	int checkCollisions();

protected:
	GJKAlgorithm* m_pAlgorithm;
	std::vector<NarrowCollisionInfo> m_CollisionPairs;
};

