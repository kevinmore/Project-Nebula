/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Util/hkpSphereUtil.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>

// assert to verify the correctness of getVertices()
HK_COMPILE_TIME_ASSERT( sizeof(hkVector4) == sizeof(hkSphere) );

#if !defined(HK_PLATFORM_SPU)

hkpCapsuleShape::hkpCapsuleShape( const hkVector4& vertexA, const hkVector4& vertexB, hkReal radius)
:	hkpConvexShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpCapsuleShape), radius)
{
	hkSimdReal simdRadius;	simdRadius.setFromFloat(radius);
	m_vertexA.setXYZ_W(vertexA, simdRadius);
	m_vertexB.setXYZ_W(vertexB, simdRadius);

#ifdef HK_DEBUG
	hkVector4 diff; diff.setSub( vertexB, vertexA );
	HK_ASSERT2( 0xf010345, diff.length<3>().getReal() != 0.0f, "You are not allowed to create a capsule with identical vertices");
#endif
}

//
//	Serialization constructor

hkpCapsuleShape::hkpCapsuleShape( hkFinishLoadedObjectFlag flag )
:	hkpConvexShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpCapsuleShape));
}

void hkpCapsuleShape::getFirstVertex(hkVector4& v) const
{
	v = getVertex<1>();
}

#endif //HK_PLATFORM_SPU

const hkSphere* hkpCapsuleShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	sphereBuffer[0].setPositionAndRadius( m_vertexA );
	sphereBuffer[1].setPositionAndRadius( m_vertexB );
	return &sphereBuffer[0];
}

void hkpCapsuleShape::closestInfLineSegInfLineSeg( const hkVector4& A, const hkVector4& dA, const hkVector4& B, const hkVector4& dB, hkReal& distSquared, hkReal& t, hkReal &u, hkVector4& p, hkVector4& q )
{

	hkLineSegmentUtil::ClosestPointInfLineInfLineResult result;
	hkLineSegmentUtil::closestPointInfLineInfLine(A, dA, B, dB, result);
	
	p = result.m_closestPointA;
	q = result.m_closestPointB;
	t = result.m_fractionA;
	u = result.m_fractionB;
	distSquared = result.m_distanceSquared;
}

void hkpCapsuleShape::closestPointLineSeg( const hkVector4& A, const hkVector4& B, const hkVector4& B2, hkVector4& pt )
{
	hkVector4 d12; d12.setSub( B, A );
	hkVector4 dB;  dB.setSub( B2, B );

	const hkSimdReal S2 = dB.dot<3>(d12);
	const hkSimdReal D2 = dB.dot<3>(dB);

	HK_ASSERT2(0x58206027,  D2.getReal() != 0.0f, "Length of segment B is zero");

	hkSimdReal u = -S2;

	// If u not in range, modify
	if(u.isLessEqualZero())
	{
		pt = B;
		return;
	}
	else
	{
		if(u.isGreaterEqual(D2))
		{
			pt = B2;
			return;
		}
		else
		{
			u.div(D2);
			pt.setAddMul( B, dB, u );
			return;
		}
	}
}

hkBool hkpCapsuleShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIME_CODE_BLOCK("rcCapsule", HK_NULL);

	{
	    hkVector4 res;
	    closestPointLineSeg( input.m_from, getVertex<0>(), getVertex<1>(), res );

	    // Ray starts inside capsule... reject!
	    hkVector4 join; join.setSub(input.m_from, res);
	    hkReal sToCylDistSq = join.lengthSquared<3>().getReal();
	    if(sToCylDistSq < m_radius * m_radius)
	    {
		    goto returnFalse;
	    }

		// Work out closest points to cylinder
		hkReal infInfDistSquared = HK_REAL_MAX;
		hkReal t, u;
		hkVector4 p,q;

		hkVector4 dA;
		dA.setSub(input.m_to, input.m_from);
		hkVector4 dB;
		dB.setSub(getVertex<1>(), getVertex<0>());

		// Get distance between inf lines + parametric description (t, u) of closest points,
		closestInfLineSegInfLineSeg(input.m_from, dA, getVertex<0>(), dB, infInfDistSquared, t, u, p, q);


		// Is infinite ray within radius of infinite cylinder?
		if(infInfDistSquared > m_radius * m_radius)
		{
			goto returnFalse;
		}

		hkSimdReal axisLength;
		hkVector4 axis;
		hkReal ipT;
		{

			axis = dB;

			// Check for zero axis
			const hkSimdReal axisLengthSqrd = axis.lengthSquared<3>();
			if (axisLengthSqrd > hkSimdReal_Eps)
			{
				axisLength = axis.normalizeWithLength<3>();
			}
			else
			{
				axisLength.setZero();
				axis.setZero();
			}

			hkVector4 dir = dA;
			hkSimdReal component = dir.dot<3>(axis);

			hkVector4 flatDir;
			flatDir.setAddMul(dir, axis, -component);

			// Flatdir is now a ray firing in the "plane" of the cyliner.

						// Convert d to a parameterisation instead of absolute distance along ray.
			// Avoid division by zero in case of ray parallel to infinite cylinder.
			const hkSimdReal flatDirLengthSquared3 = flatDir.lengthSquared<3>();
			if(flatDirLengthSquared3.isNotEqualZero()) // Common case
			{
				hkReal d;
				d = hkMath::sqrt( (m_radius * m_radius - infInfDistSquared) / flatDirLengthSquared3.getReal() );
				// This represents a parameterisation along the ray of the intersection point
				ipT = t - d;
			}
			else // Very rare case
			{
				// We are parallel to cylinder axis, so need to get straight to caps tests
				// To accomplish this, set ipT as (any!) negative to bypass next two if() statements
				// and note that neither ipT not any other results calculated from this
				// (intersectPt, ptProj, ptHeight) between here and the caps tests are used later.
				ipT = -1.0f;
			}
		}

		// Intersection parameterization with infinite cylinder is outside length of ray
		// or is greater than a previous hit fraction
		if( ipT >= results.m_hitFraction )
		{
			goto returnFalse;
		}

		hkSimdReal ptHeight;
		hkSimdReal pointAProj = getVertex<0>().dot<3>(axis);

		// Find intersection point of actual ray with infinite cylinder
		hkVector4 intersectPt;
		intersectPt.setInterpolate( input.m_from, input.m_to, hkSimdReal::fromFloat(ipT) );

		// Test height of intersection point w.r.t. cylinder axis
		// to determine hit against actual cylinder
		// Intersection point projected against cylinder
		const hkSimdReal ptProj = intersectPt.dot<3>(axis);
		ptHeight = ptProj - pointAProj;

		if( ipT >= 0 ) // Actual ray (not infinite ray) must intersect with infinite cylinder
		{
			if(ptHeight.isGreaterZero() && ptHeight < axisLength) // Report hit against cylinder part
			{
				// Calculate normal
				hkVector4 projPtOnAxis;
				projPtOnAxis.setInterpolate( getVertex<0>(), getVertex<1>(), ptHeight / axisLength );
				hkVector4 normal;	normal.setSub( intersectPt, projPtOnAxis );

				normal.normalize<3>();
				results.m_normal = normal;
				results.m_hitFraction = ipT; // This is a parameterization along the ray
				results.m_extraInfo = HIT_BODY;
				results.setKey( HK_INVALID_SHAPE_KEY );
				return true;
			}
		}

		// Cap tests


		{
			// Check whether start point is inside infinite cylinder or not
			hkSimdReal fromLocalProj = input.m_from.dot<3>(axis);
			hkSimdReal projParam = fromLocalProj - pointAProj;

			hkVector4 fromPtProjAxis;
			fromPtProjAxis.setInterpolate( getVertex<0>(), getVertex<1>(), projParam / axisLength );

			hkVector4 axisToRayStart;
			axisToRayStart.setSub(input.m_from, fromPtProjAxis);

			if((ipT < 0) &&  (axisToRayStart.lengthSquared<3>().getReal() > m_radius*m_radius))
			{
 				// Ray starts outside infinite cylinder and points away... must be no hit
				goto returnFalse;
			}

			// Ray can only hit 1 cap... Use intersection point
			// to determine which sphere to test against (NB: this only works because
			// we have discarded cases where ray starts inside)
			hkVector4 extraVertex; extraVertex.setSelect(ptHeight.lessEqualZero(), m_vertexA, m_vertexB);
			hkVector4 from; from.setSub( input.m_from, extraVertex );
			hkVector4 to; to.setSub( input.m_to, extraVertex );

			if( hkpRayCastSphere(from, to, m_radius, results) )
			{
				results.m_extraInfo = ptHeight.isLessEqualZero() ? HIT_CAP0 : HIT_CAP1;
				return true;
			}

			return false;
		}
	}
returnFalse:
	return false;
}


void hkpCapsuleShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkVector4 tol4;
	tol4.setAll(tolerance + m_radius);
	tol4.zeroComponent<3>();

	hkVector4 obj[2];
	hkVector4Util::transformPoints( localToWorld, getVertices(), 2, obj );

	out.m_min.setMin( obj[0], obj[1] );
	out.m_min.sub( tol4 );

	out.m_max.setMax( obj[0], obj[1] );
	out.m_max.add( tol4 );
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
