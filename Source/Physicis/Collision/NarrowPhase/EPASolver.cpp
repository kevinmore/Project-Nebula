#include "EPASolver.h"
#include "NarrowPhaseCollisionDetection.h"
#include <algorithm>
#include <Physicis/Geometry/Triangle.h>
#include <Physicis/Geometry/Edge.h>
#include <Physicis/Geometry/Simplex.h>
#include <Utility/Math.h>
#include <Physicis/Collision/Collider/ICollider.h>

using namespace Math;

bool EPASolver::computePenetrationDepthAndContactPoints( const Simplex& simplex, NarrowPhaseCollisionFeedback& pCollisionInfo, int maxIteration /*= 30*/ )
{
	QVector<vec3> suppPointsA;       
	QVector<vec3> suppPointsB;    
	QVector<vec3> points;   

	suppPointsA.reserve(20);
	suppPointsB.reserve(20);
	points.reserve(20);

	// Initialize collision info
	pCollisionInfo.bIntersect = false;
	pCollisionInfo.penetrationDepth = 0;
	pCollisionInfo.proximityDistance = 0;

	ICollider* objA = pCollisionInfo.pObjA;
	ICollider* objB = pCollisionInfo.pObjB;

	// transform a local position in objB space to local position in objA space
	Transform transB2A = objA->getTransform().inversed() * objB->getTransform();

	// transform a local position in objA space to local position in objB space
	Transform transA2B = objB->getTransform().inversed() * objA->getTransform();

	// rotate matrix which transform a local vector in objA space to local vector in objB space
	glm::mat3 rotB = objB->getTransform().getRotationMatrix();
	glm::mat3 rotA = objA->getTransform().getRotationMatrix();
	glm::mat3 rotA2B = glm::transpose(rotB) * rotA;

	int numVertices = simplex.getPoints(suppPointsA, suppPointsB, points);
	m_polytope.clear();

	assert(numVertices == points.size());
	assert(m_polytope.getTriangles().size() == 0);

	switch ( numVertices )
	{
	case 1:
		// Two objects are barely touching.
		return false;

	case 2:
		// The origin lies in a line segment. 
		// We create a hexahedron which is glued two tetrahedrons. It is explained in Geno's book. 
		break;

	case 3:
		{
			// The origin lies in a triangle. 
			// Add two new vertices to create a hexahedron. It is explained in Geno's book. 
			vec3 n = vec3::crossProduct(points[1] - points[0], points[2] - points[0]);
			vec3 w0 =  objA->getLocalSupportPoint(n)  - transB2A * objB->getLocalSupportPoint(Converter::toQtVec3(rotA2B * Converter::toGLMVec3(-n)));
			vec3 w1 =  objA->getLocalSupportPoint(-n) - transB2A * objB->getLocalSupportPoint(Converter::toQtVec3(rotA2B * Converter::toGLMVec3(n)));

			if ( !m_polytope.addHexahedron(points[0], points[1], points[2], w0, w1) )
				return false;
		}
		break;

	case 4:
		{
			// In this case, simplex computed from GJK is a tetrahedron. 
			if ( !m_polytope.addTetrahedron(points[0], points[1], points[2], points[3]) )
				return false;
		}
		break;
	}

	assert(m_polytope.getVertices().size() > 0);

	// Now we can expand the polytope which contains the origin to get the penetration depth and contact points. 
	float upperBound = FLT_MAX;
	float lowerBound = -FLT_MAX;

	int numIter = 0;
	while ( numIter < maxIteration )
	{
		Triangle* pClosestTriangle = m_polytope.popAClosestTriangleToOriginFromHeap();
		if(!pClosestTriangle)  return false;
		vec3 v = pClosestTriangle->getClosestPoint().normalized();

		vec3 supportPointA = objA->getLocalSupportPoint(v, objA->getMargin());
		vec3 supportPointB = transB2A * objB->getLocalSupportPoint(Converter::toQtVec3(rotA2B * Converter::toGLMVec3(-v)), objB->getMargin());

		vec3 w = supportPointA - supportPointB;
		// Compute upper and lower bounds
		upperBound = qMin(upperBound, vec3::dotProduct(w, v));
		lowerBound = qMax(lowerBound, 1.0f); // 1 == v.length()

		if ( upperBound - lowerBound < 1e-4 || numIter == maxIteration - 1 )
		//if ( qFuzzyIsNull(upperBound - lowerBound) || numIter == maxIteration - 1 )
		{
			pCollisionInfo.bIntersect = true;
			pCollisionInfo.contactNormalWorld = v;
			pCollisionInfo.penetrationDepth = 0.5f * (upperBound + lowerBound);

			pCollisionInfo.closestPntALocal = pClosestTriangle->getClosestPointToOriginInSupportPntSpace(suppPointsA);
			pCollisionInfo.closestPntBLocal = transA2B * pClosestTriangle->getClosestPointToOriginInSupportPntSpace(suppPointsB);

			pCollisionInfo.closestPntAWorld = objA->getTransform() * pCollisionInfo.closestPntALocal;
			pCollisionInfo.closestPntBWorld = objB->getTransform() * pCollisionInfo.closestPntBLocal;

			pCollisionInfo.proximityDistance = 0;
			pCollisionInfo.pObjA = objA;
			pCollisionInfo.pObjB = objB;

			break;
		}

		if ( !m_polytope.expandPolytopeWithNewPoint(w, pClosestTriangle) )
		{
			pCollisionInfo.bIntersect = false;
			return false;
		}

		suppPointsA.push_back(supportPointA);
		suppPointsB.push_back(supportPointB);
		points.push_back(w);

		++numIter;
	}

	return pCollisionInfo.bIntersect;
}
