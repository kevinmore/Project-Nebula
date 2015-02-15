#include "GJKSolver.h"
#include "EPASolver.h"
#include "NarrowPhaseCollisionDetection.h"
#include <Physicis/Collision/Collider/ICollider.h>
#include <Physicis/Geometry/Edge.h>
#include <Physicis/Geometry/Simplex.h>

bool GJKSolver::generateCollisionInfo( const ICollider* objA, const ICollider* objB, const Transform &transB2A, const Simplex& simplex, vec3 v, float distSqrd, NarrowPhaseCollisionFeedback& pCollisionInfo ) const
{
	vec3 closestPntA;
	vec3 closestPntB;
	simplex.closestPointAandB(closestPntA, closestPntB);

	float dist = sqrt(distSqrd);
	pCollisionInfo.proximityDistance = dist;

	Q_ASSERT(dist > 0.0);

	vec3 n = v.normalized();

	closestPntA = closestPntA + (objA->getMargin() * (-n));
	closestPntB = closestPntB + (objB->getMargin() * (n));
	closestPntB = transB2A.inversed() * closestPntB;

	// normal vector of collision
	vec3 normalCollision = objA->getTransform().getRotation().rotatedVector(-n);
			
	// penetration depth
	float margin = objA->getMargin() + objB->getMargin();
	float penetrationDepth = margin - dist;

	pCollisionInfo.penetrationDepth = penetrationDepth;
	pCollisionInfo.closestPntALocal = closestPntA;
	pCollisionInfo.closestPntBLocal = closestPntB;
	pCollisionInfo.closestPntAWorld = objA->getTransform() * closestPntA;
	pCollisionInfo.closestPntBWorld = objB->getTransform() * closestPntB;

	if ( penetrationDepth <= 0 )
		pCollisionInfo.bIntersect = false;
	else
		pCollisionInfo.bIntersect = true;

	return pCollisionInfo.bIntersect;
}

bool GJKSolver::checkCollision( ICollider* objA, ICollider* objB, NarrowPhaseCollisionFeedback& pCollisionInfo, bool bProximity /*= false*/ )
{
	vec3 suppPntA; // support point from object A
	vec3 suppPntB; // support point from object B
	vec3 closestPntA;
	vec3 closestPntB;
	vec3 w; // support point of Minkowski difference(A-B)
	float vw; // v dot w
	const float margin = objA->getMargin() + objB->getMargin();
	const float marginSqrd = margin * margin;
	Simplex simplex;

	// transform a local position in objB space to local position in objA space
	Transform transA = objA->getTransform();
	Transform transB = objB->getTransform();
	Transform transB2A = transA.inversed() * transB;

	// rotation which transforms a local vector in objA space to local vector in objB space
	quat rotA = objA->getTransform().getRotation();
	quat rotB = objB->getTransform().getRotation();
	quat rotA2B = rotB.conjugate() * rotA;

	// closest point to the origin
	vec3 v(1.0, 0.0, 0.0);

	float distSqrd = FLT_MAX;
	float distSqrdPrev = distSqrd;

	while (true) 
	{
		// support points are in local space of objA
		suppPntA = objA->getLocalSupportPoint(-v);
		suppPntB = transB2A * objB->getLocalSupportPoint(rotA2B.rotatedVector(v)); 

		// w is also in local space of objA
		w = suppPntA - suppPntB;

		vw = vec3::dotProduct(v, w);

		// If v.Dot(w) > 0, it means there is a separating axis and objA and objB do not intersect.
		// If v.Dot(w)^2 / dist^2 > (margin from objA + margin from objB)^2, it means enlarged objects with margins do not intersect.
		// So we just exit this function if we don't check proximity. If we compute proximity(distance), we should not terminate here
		// and keep iterating until w becomes a part of simplex or dist is close enough to be proxmity distance satisfing (dist^2 - v*w <= dist^2 * tolerance). 
		// In this case, penetration depth will be negative and this function will return false since objects don't intersect. 
		bool bFoundSeparatingAxis = false;

		if ( vw > 0.0 && vw*vw > (distSqrd * marginSqrd) )
			bFoundSeparatingAxis = true;

		if ( !bProximity && bFoundSeparatingAxis )
			return false;

		// Check if w is a part of simplex already. If so, it means v == w and v is a closest point to origin in Minkowski differnce and 
		// objects are disjoint. 
		// If dist^2 - v*w <= dist^2 * tolerance, v is close enough to the closest point to origin in Minkowski differnce and 
		// ||v|| is a proximity distance. In this case, objects are disjoint because v * w > 0 and it means
		// v is a separating axis. 
		// 
		// TODO: Techinically, the scond condidion(dist^2 - v*w <= dist^2 * tolerance) will cover the first condition(w is a part of simplex already).
		//       A little bit redundancy. 
		bool isDegenerate = simplex.isDegenerate(w);
		if ( simplex.isDegenerate(w) || distSqrd - vw <= distSqrd * 1e-6 )
		{
			return generateCollisionInfo(objA, objB, transB2A, simplex, v, distSqrd, pCollisionInfo);			
		}

		// Add w into simplex. Determinants will be computed inside addPoint(..). 
		simplex.addPoint(w, suppPntA, suppPntB);

		if ( !simplex.isAffinelyIndependent() ) 
		{
			return generateCollisionInfo(objA, objB, transB2A, simplex, v, distSqrd, pCollisionInfo);
		}

		// Run Johnson's Algorithm
		// Using Johnson's Algorithm, we can calculate a new 'v' as well as the subset simplex which contains 'v'.
		// The subset simplex can be a vertex, edge, triangle or tetrahedron. 
		if ( !simplex.runJohnsonAlgorithm(v) )
		{
			return generateCollisionInfo(objA, objB, transB2A, simplex, v, distSqrd, pCollisionInfo);			
		}

		distSqrdPrev = distSqrd;
		distSqrd = v.lengthSquared();

		// If there is not much improvement since previous iteration
		// TODO: Do I need to check this? 
		if ( abs(distSqrdPrev - distSqrd) <= DBL_EPSILON * distSqrdPrev )
		{
			return generateCollisionInfo(objA, objB, transB2A, simplex, v, distSqrd, pCollisionInfo);	
		}

		// If simplex is full, we found a simplex(tetrahedron) containing origin. 
		// We stop iterating and pass this simplex to EPA Solver to find penetration depth and closest points.
		if ( simplex.isFull() )
		{
			if ( bFoundSeparatingAxis )
				return false;
			else
				break;
		}

		// Instead seeing if distSqrd is zero within tolerance, we use relative tolerance using max squared length from simplex.
		// This is explained in 'Collision Detection in Interactive 3D'.
		// So if distSqrd is less than or equal to EPSILON * (max squared length from simplex), we consider that we've found
		// a simplex containing origin. In this case, the simplex could be a vertex, a line segment, a triangle or a tetrahedron. 
		// We stop iterating and pass this simplex to EPA to find penetration depth and closest points.
		// TODO: FreeSolid is using 'approxZero(v)'. Do I need to use it rather than a line below?
		//if ( distSqrd <= DBL_EPSILON * simplex.MaxLengthSqrd() ) 
		if ( distSqrd <= 1e-6 ) 
			return false; // TODO:break or return false?

	} 

	// TODO: Need to invesgate hybrid method metioned in Geno's book.
	// TODO: Need to use GJK to compute the shallow penetration depth.

	return m_EPASolver.computePenetrationDepthAndContactPoints(simplex, objA, objB, v, pCollisionInfo);
}
