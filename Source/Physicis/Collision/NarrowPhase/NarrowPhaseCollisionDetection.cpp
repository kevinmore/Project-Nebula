#include "NarrowPhaseCollisionDetection.h"


NarrowPhaseCollisionDetection::NarrowPhaseCollisionDetection()
{
}


NarrowPhaseCollisionDetection::~NarrowPhaseCollisionDetection()
{
}

int NarrowPhaseCollisionDetection::checkCollisions()
{
	int numIntersections = 0;

	for ( std::vector<NarrowPhaseCollisionFeedback>::iterator iter = m_CollisionPairs.begin(); iter != m_CollisionPairs.end(); iter++ )
	{
		if ( m_solver.checkCollision((*iter).pObjA, (*iter).pObjB, *iter, true) )
			++numIntersections;
	}

	return numIntersections;
}
