#include "NarrowPhaseCollisionDetection.h"


NarrowPhaseCollisionDetection::NarrowPhaseCollisionDetection(void)
{
}


NarrowPhaseCollisionDetection::~NarrowPhaseCollisionDetection(void)
{
}

int NarrowPhaseCollisionDetection::CheckCollisions()
{
	int numIntersections = 0;

// 	for ( std::vector<NarrowCollisionInfo>::iterator iter = m_CollisionPairs.begin(); iter != m_CollisionPairs.end(); iter++ )
// 	{
// 		if ( m_pAlgorithm->CheckCollision(*(*iter).pObjA, *(*iter).pObjB, &(*iter), true) )
// 			++numIntersections;
// 	}

	return numIntersections;
}
