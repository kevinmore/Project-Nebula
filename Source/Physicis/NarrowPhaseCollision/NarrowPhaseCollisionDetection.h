#pragma once
#include <Utility/EngineCommon.h>

class CollisionObject;
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
	virtual ~NarrowPhaseCollisionDetection();

	std::vector<NarrowCollisionInfo>& GetPairs() { return m_CollisionPairs; }
	const std::vector<NarrowCollisionInfo>& GetPairs() const { return m_CollisionPairs; }
	void AddPair(const NarrowCollisionInfo pair) { m_CollisionPairs.push_back(pair); }

	int CheckCollisions();

protected:
	std::vector<NarrowCollisionInfo> m_CollisionPairs;

};

