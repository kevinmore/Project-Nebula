#include "NarrowPhaseCollisionDetection.h"
#include "GJKSolver.h"


NarrowPhaseCollisionDetection::NarrowPhaseCollisionDetection()
{
}


NarrowPhaseCollisionDetection::~NarrowPhaseCollisionDetection()
{
	delete m_pAlgorithm;
}

int NarrowPhaseCollisionDetection::checkCollisions()
{
	int numIntersections = 0;

	for ( std::vector<NarrowPhaseCollisionFeedback>::iterator iter = m_CollisionPairs.begin(); iter != m_CollisionPairs.end(); iter++ )
	{
		if ( m_pAlgorithm->checkCollision((*iter).pObjA, (*iter).pObjB, *iter, true) )
			++numIntersections;
	}

	return numIntersections;
}
