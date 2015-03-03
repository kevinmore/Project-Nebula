/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Util/hkpSphereUtil.h>
#include <Common/Base/Math/Vector/hkFourTransposedPoints.h>

#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>

#if HK_POINTER_SIZE == 4 && !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT( sizeof(hkpSphereShape) % 16 == 0 );
#endif

#if !defined(HK_PLATFORM_SPU)

hkpSphereShape::hkpSphereShape(hkReal radius)
:	hkpConvexShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpSphereShape), radius)
{}

//
//	Serialization constructor

hkpSphereShape::hkpSphereShape( hkFinishLoadedObjectFlag flag )
:	hkpConvexShape(flag)
{
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpSphereShape));
}

void hkpSphereShape::getFirstVertex(hkVector4& v) const
{
	v.setZero();
}

#endif

hkBool hkpSphereShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIME_CODE_BLOCK("rcSphere", HK_NULL);
	return hkpRayCastSphere(input.m_from, input.m_to, m_radius, results);
}


hkVector4Comparison hkpSphereShape::castRayBundle(const hkpShapeRayBundleCastInput& input, hkpShapeRayBundleCastOutput& results, hkVector4ComparisonParameter mask) const
{
	HK_ASSERT2(0x29d49b87, mask.anyIsSet(), "Calling castRayBundle with no active rays!");

	HK_TIMER_BEGIN("rcSphereBundle", HK_NULL);
	
	//
	//	This functions is a vectorized version of hkpSphereUtil::castRayUtil,
	//	which is itself a modified version of 
	//  http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter1.htm
	//
	//	The algorithm is the same as in hkpSphereUtil::castRayUtil, although some
	//	early-out checks are omitted since it is unlikely that all 4 rays could early-out
	//	These omissions are noted in the code.
	//
	
	hkVector4Comparison activeMask = mask;

	{
		hkSimdReal radius; radius.load<1>(&m_radius);
		hkSimdReal radius2; radius2.setMul(radius, radius);

		const hkSimdReal oneHundred = hkSimdReal::fromFloat(hkReal(100));		

		// 
		// solve quadratic function: ax*x + bx + c = 0
		//

		// Scalar version:
		//	A = dir.lengthSquared3();
		//	B = dir.dot3( input.m_from )
		hkVector4 A;
		hkVector4 B;
		{
			hkFourTransposedPoints vDir;
			
			vDir.setSub(input.m_to, input.m_from);
			vDir.dot3(vDir, A);
			vDir.dot3(input.m_from, B);
		}
		
		// Omitting early-out check:  if ( B >= 0 )


		//
		//	Check for long rays (check for startpoints being 10 times outside the radius
		//
		hkVector4 offset;
		
		hkVector4 AtimeRadius2time100; AtimeRadius2time100.setMul(A, radius2);
		AtimeRadius2time100.mul(oneHundred);
		hkVector4 B2; B2.setMul(B,B);

		hkVector4Comparison longRayCheck = B2.greater(AtimeRadius2time100);

		// Omitting early-out check: if ( B * B > A * radius2 * 100.0f) && if ( A < radius2 )

		offset.setNeg<4>(B);
		offset.zeroIfFalse(longRayCheck);

		B.zeroIfTrue(longRayCheck);
		B2.zeroIfTrue(longRayCheck);

		hkVector4 invA; invA.setReciprocal(A);
		hkVector4 midPointInterp; midPointInterp.setMul(offset, invA);

		// Don't need a select for the midpoints, since the interpolation value is zero for the non-long ray case
		hkVector4 one = hkVector4::getConstant<HK_QUADREAL_1>();
		hkVector4 oneMinusInterp; oneMinusInterp.setSub(one, midPointInterp);

		// Scalar version:  C = midPoint.lengthSquared3() - radius2;
		hkVector4 C;
		hkFourTransposedPoints midPoint;
		midPoint.setMulT(input.m_from, oneMinusInterp);
		midPoint.addMulT(input.m_to, midPointInterp);
		midPoint.dot3(midPoint, C);
		C.setSub(C,radius2);

		// Scalar version: det = B*B - A*C;
		hkVector4 det;
		det.setNeg<4>(A);
		det.mul(C);
		det.add(B2);

		hkVector4Comparison infRayDoesHit = det.greaterZero();
		// We need this modification to the mask: if det<=0, the (infinite) ray doesn't hit.
		activeMask.setAnd(activeMask, infRayDoesHit);


		// Scalar version:
		//	sqDet = hkMath::sqrt( det );
		//	t2 = -B - sqDet;
		//	t = t2 + offset;
		// We're taking the inverse square root of a (possibly) negative number or zero
		// But that's OK, since we mask out the results

		hkVector4 sqDet;
		sqDet.setSqrt(det);
		sqDet.zeroIfFalse(infRayDoesHit); // this makes sure we're OK if det=0

		hkVector4 tV;
		tV.setSub(offset,B);
		tV.sub(sqDet);

		hkVector4 earlyOutFractions;
		earlyOutFractions.set(	results.m_outputs[0].m_hitFraction, results.m_outputs[1].m_hitFraction,
								results.m_outputs[2].m_hitFraction, results.m_outputs[3].m_hitFraction );
		earlyOutFractions.mul(A);

		// Scalar version checks  t >= (A * results.m_hitFraction)  and then computes t/A
		// Since we're going to end up dividing anyway, just compare t/A >= results.m_hitFraction
		//tV.mul4(invA);

		hkVector4Comparison tLessThanHitFrac = tV.less(earlyOutFractions);
		hkVector4Comparison tGreaterThan0 = tV.greaterEqualZero();

		// We have a hit if (t < m_hitFraction) && t>=0
		activeMask.setAnd(activeMask, tLessThanHitFrac);
		activeMask.setAnd(activeMask, tGreaterThan0);

		tV.mul(invA);
		hkVector4 oneMinusT; oneMinusT.setSub(one, tV);

		// Compute all the normals at once instead of doing scalar divides
		hkVector4 normals[4];
		{
			// <ce.todo> Multiply t and oneOverT by 1/radius first
			hkFourTransposedPoints fourNormals;
			fourNormals.setMulT(input.m_from, oneMinusT);
			fourNormals.addMulT(input.m_to, tV);
			hkSimdReal radiusI; radiusI.setReciprocal(radius);
			hkVector4 invRadius; invRadius.setAll(radiusI);
			fourNormals.mulT(invRadius);
			fourNormals.extract(normals[0], normals[1], normals[2], normals[3]);
		}
		

		const hkVector4Comparison::Mask aM = activeMask.getMask();

		if (aM & hkVector4ComparisonMask::MASK_X)
		{
			tV.store<1>((hkReal*)&results.m_outputs[0].m_hitFraction);
			results.m_outputs[0].m_normal = normals[0];
			results.m_outputs[0].setKey(HK_INVALID_SHAPE_KEY);
		}
		if (aM & hkVector4ComparisonMask::MASK_Y)
		{
			tV.getComponent<1>().store<1>((hkReal*)&results.m_outputs[1].m_hitFraction);
			results.m_outputs[1].m_normal = normals[1];
			results.m_outputs[1].setKey(HK_INVALID_SHAPE_KEY);
		}
		if (aM & hkVector4ComparisonMask::MASK_Z)
		{
			tV.getComponent<2>().store<1>((hkReal*)&results.m_outputs[2].m_hitFraction);
			results.m_outputs[2].m_normal = normals[2];
			results.m_outputs[2].setKey(HK_INVALID_SHAPE_KEY);
		}
		if (aM & hkVector4ComparisonMask::MASK_W)
		{
			tV.getComponent<3>().store<1>((hkReal*)&results.m_outputs[3].m_hitFraction);
			results.m_outputs[3].m_normal = normals[3];
			results.m_outputs[3].setKey(HK_INVALID_SHAPE_KEY);
		}
	}

	HK_TIMER_END();
	return activeMask;
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
