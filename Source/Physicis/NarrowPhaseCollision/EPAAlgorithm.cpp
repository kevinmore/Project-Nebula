#include "EPAAlgorithm.h"
#include <algorithm>
#include "GJKSimplex.h"
#include "EPAEdge.h"
#include "EPATriangle.h"

EPAAlgorithm::EPAAlgorithm(){}
EPAAlgorithm::~EPAAlgorithm(){}

bool EPAAlgorithm::ComputePenetrationDepthAndContactPoints( const GJKSimplex& simplex, CollisionObject& objA, CollisionObject& objB, vec3& v, NarrowCollisionInfo* pCollisionInfo, int maxIteration /*= 30*/ )
{
	return true;
}
