#include "EPASolver.h"
#include "NarrowPhaseCollisionDetection.h"
#include <algorithm>
#include <Physicis/Geometry/Triangle.h>
#include <Physicis/Geometry/Edge.h>
#include <Physicis/Geometry/Simplex.h>
#include <Utility/Math.h>
#include <Physicis/Collision/Collider/ICollider.h>

using namespace Math;

bool EPASolver::computePenetrationDepthAndContactPoints( const Simplex& simplex, ICollider* objA, ICollider* objB, vec3& v, NarrowPhaseCollisionFeedback& pCollisionInfo, int maxIteration /*= 30*/ )
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
	pCollisionInfo.pObjA = objA;
	pCollisionInfo.pObjB = objB;

	// transform a local position in objB space to local position in objA space
	Transform transB2A = objA->getTransform().inversed() * objB->getTransform();

	// transform a local position in objA space to local position in objB space
	Transform transA2B = objB->getTransform().inversed() * objA->getTransform();

	// rotate matrix which transform a local vector in objA space to local vector in objB space
	mat3 rotB = objB->getTransform().getRotationMatrix();
	mat3 rotA = objA->getTransform().getRotationMatrix();
	mat3 rotA2B = rotB.transposed() * rotA;

	int numVertices = simplex.getPoints(suppPointsA, suppPointsB, points);
	m_Polytope.clear();

	Q_ASSERT(numVertices == points.size());
	Q_ASSERT(m_Polytope.getTriangles().size() == 0);

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
			vec3 w0 =  objA->getLocalSupportPoint(n) - transB2A * objB->getLocalSupportPoint(Vector3::setMul(-n, rotA2B));
			vec3 w1 =  objA->getLocalSupportPoint(-n) - transB2A * objB->getLocalSupportPoint(Vector3::setMul(n, rotA2B));

			if ( !m_Polytope.addHexahedron(points[0], points[1], points[2], w0, w1) )
				return false;
		}
		break;

	case 4:
		{
			// In this case, simplex computed from GJK is a tetrahedron. 
			if ( !m_Polytope.addTetrahedron(points[0], points[1], points[2], points[3]) )
				return false;
		}
		break;
	}

	Q_ASSERT(m_Polytope.getVertices().size() > 0);

	// Now we can expand the polytope which contains the origin to get the penetration depth and contact points. 
	float upperBound = FLT_MAX;
	float lowerBound = -FLT_MAX;

	int numIter = 0;
	while ( numIter < maxIteration )
	{
		Triangle* pClosestTriangle = m_Polytope.popAClosestTriangleToOriginFromHeap();
		Q_ASSERT(pClosestTriangle != NULL);

		v = pClosestTriangle->getClosestPoint();

		vec3 supportPointA = objA->getLocalSupportPoint(v, objA->getMargin());
		vec3 supportPointB = transB2A * objB->getLocalSupportPoint(Vector3::setMul(-v, rotA2B), objB->getMargin());

		vec3 w = supportPointA - supportPointB;
		// Compute upper and lower bounds
		upperBound = qMin(upperBound, vec3::dotProduct(w, v.normalized()));
		lowerBound = qMax(lowerBound, v.length());

		if ( upperBound - lowerBound < 1e-4 || numIter == maxIteration - 1 )
		{
			pCollisionInfo.bIntersect = true;
			pCollisionInfo.penetrationDepth = 0.5f * (upperBound + lowerBound);
			pCollisionInfo.contactPntALocal = pClosestTriangle->getClosestPointToOriginInSupportPntSpace(suppPointsA);
			pCollisionInfo.contactPntBLocal = transA2B * pClosestTriangle->getClosestPointToOriginInSupportPntSpace(suppPointsB);

			pCollisionInfo.contactPntAWorld = objA->getTransform() * pCollisionInfo.contactPntALocal;
			pCollisionInfo.contactPntBWorld = objB->getTransform() * pCollisionInfo.contactPntBLocal;

			vec3 impulseDirectionAWorld = pCollisionInfo.contactPntAWorld - pCollisionInfo.contactPntBWorld;
			vec3 impulseDirectionBWorld = - impulseDirectionAWorld;

			pCollisionInfo.impulseDirectionOnALocal = objA->getTransform().inversed() * impulseDirectionAWorld;
			pCollisionInfo.impulseDirectionOnBLocal = objB->getTransform().inversed() * impulseDirectionBWorld;
			pCollisionInfo.impulseDirectionOnALocal.normalize();
			pCollisionInfo.impulseDirectionOnBLocal.normalize();

			pCollisionInfo.proximityDistance = 0;
			pCollisionInfo.pObjA = objA;
			pCollisionInfo.pObjB = objB;

			break;
		}

		if ( !m_Polytope.expandPolytopeWithNewPoint(w, pClosestTriangle) )
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
