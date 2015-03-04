/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/hkcdInternal.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointConvex.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

const static hkSimdReal g_eps_addPlane = hkSimdReal::fromFloat(1e-4f);
const static hkQuadReal g_zeroMax = HK_QUADREAL_CONSTANT(0.0f, 0.0f, 0.0f, HK_REAL_MAX);


// Assumes that the planes are linearly independent
HK_FORCE_INLINE hkVector4 projectPointOntoThreePlanes(hkVector4Parameter x, hkVector4Parameter plane1, hkVector4Parameter plane2, hkVector4Parameter plane3, hkVector4& lambdas)
{
	hkVector4 r0; r0.setCross( plane2, plane3 );
	hkVector4 r1; r1.setCross( plane3, plane1 );
	hkVector4 r2; r2.setCross( plane1, plane2 );

	const hkSimdReal determinant = plane3.dot<3>(r2);
	hkSimdReal dinv; dinv.setReciprocal(determinant);
	hkMatrix3 mInvT; 
	mInvT.getColumn<0>().setMul(dinv, r0);
	mInvT.getColumn<1>().setMul(dinv, r1);
	mInvT.getColumn<2>().setMul(dinv, r2);

	hkMatrix3 mInv; mInv.setTranspose(mInvT);
	hkVector4 rhs; rhs.set(-plane1.getW(), -plane2.getW(), -plane3.getW(), hkSimdReal::getConstant<HK_QUADREAL_0>());
	hkVector4 result; mInvT.multiplyVector(rhs, result);

	hkVector4 diff; diff.setSub(x, result);
	mInv.multiplyVector(diff, lambdas);

	return result;
}

// Assumes that planes are not coplanar
HK_FORCE_INLINE hkVector4 projectPointOntoTwoPlanes(hkVector4Parameter x, hkVector4Parameter plane1, hkVector4Parameter plane2, hkVector4& lambdas)
{
	hkVector4 plane3; plane3.setCross(plane1, plane2);

	// Note: It is possible to solve this as a 2x2 equation system, but the final result will not be accurate enough
#if 1
	hkVector4 r0; r0.setCross( plane2, plane3 );
	hkVector4 r1; r1.setCross( plane3, plane1 );

	const hkSimdReal determinant = plane3.lengthSquared<3>();
	hkSimdReal dinv; dinv.setReciprocal(determinant);
	hkMatrix3 mInvT; 
	mInvT.getColumn<0>().setMul(dinv, r0);
	mInvT.getColumn<1>().setMul(dinv, r1);
	mInvT.getColumn<2>().setMul(dinv, plane3);

	hkMatrix3 mInv; mInv.setTranspose(mInvT);
	hkVector4 rhs; rhs.set(-plane1.getW(), -plane2.getW(), plane3.dot<3>(x), hkSimdReal::getConstant<HK_QUADREAL_0>());
	hkVector4 result; mInvT.multiplyVector(rhs, result);

	hkVector4 diff; diff.setSub(x, result);
	mInv.multiplyVector(diff, lambdas);
	return result;
#else
	return projectPointOntoThreePlanes(x, plane1, plane2, plane3, lambdas);
#endif
}

HK_FORCE_INLINE void findFurthestPlaneAndDistance(hkVector4Parameter x0, const hkVector4* HK_RESTRICT planes, int numPlanes, hkVector4& maxPlane, hkSimdReal& maxDist)
{
	hkVector4 x; x.setXYZ_W(x0, hkSimdReal_1);
	maxPlane.setZero(); maxPlane.setW(hkSimdReal_MinusMax);
	const hkVector4* HK_RESTRICT plane = planes;

	for(int i = (numPlanes >> 2) - 1; i >= 0; --i)
	{
		hkVector4 dots;	hkVector4Util::dot4_1vs4(x, plane[0], plane[1], plane[2], plane[3], dots);
		int ixMax = dots.getIndexOfMaxComponent<4>();
		const hkSimdReal maxSignDist = dots.getComponent(ixMax);
		const hkSimdReal mpW = maxPlane.getW();
		const hkVector4Comparison cmp = maxSignDist.greater(mpW);
		hkVector4 normalDist; normalDist.setXYZ_W(plane[ixMax], maxSignDist);
		maxPlane.setSelect(cmp, normalDist, maxPlane);
		plane += 4;
	}
	for(int i = (numPlanes & 0x3) - 1; i >= 0; --i)
	{
		const hkSimdReal signDist = plane->dot<4>(x);
		const hkSimdReal mpW = maxPlane.getW();
		const hkVector4Comparison cmp = signDist.greater(mpW);
		hkVector4 normalDist; normalDist.setXYZ_W(*plane, signDist);
		maxPlane.setSelect(cmp, normalDist, maxPlane);
		plane++;
	}

	maxDist = maxPlane.getW();
	hkSimdReal d = maxPlane.dot<3>(x);
	maxPlane.setW(maxDist - d);
}


// Use an active set strategy to find the closest point on a convex hull for a given point, which is either outside or inside the convex shape
//   1) Project the point onto the intersection of the active planes.
//   2) Remove those planes that have negative Lagrange multipliers.
//   3) If the projected point is still outside the convex object, then add the most offending plane to the active set and repeat.
// Assumes that x0 has w=1
HK_FORCE_INLINE bool HK_CALL dualGJK_iteration(hkVector4Parameter x0, const hkVector4* HK_RESTRICT planes, int numPlanes, 
					   hkVector4* HK_RESTRICT activePlanesInOut, int& numActiveInOut, hkVector4& lambdasOut, hkVector4& xpOut)
{
	HK_ASSERT2(0x12391239, x0.getW() == hkSimdReal_1, "W component must be 1.");

	hkVector4 dots;	hkVector4Util::dot4_1vs4(x0, activePlanesInOut[0], activePlanesInOut[1], activePlanesInOut[2], (const hkVector4&)g_zeroMax, dots);

	// Project x0 onto the active planes; compute lambda values
	switch(numActiveInOut)
	{
	case 1:
		xpOut.setSubMul(x0, activePlanesInOut[0], dots.getComponent<0>());
		lambdasOut.setSelect<hkVector4ComparisonMask::MASK_YZ>(hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>(), hkVector4::getConstant<HK_QUADREAL_0>());
		break;
	case 2:
		xpOut = projectPointOntoTwoPlanes(x0, activePlanesInOut[0], activePlanesInOut[1], lambdasOut);
		lambdasOut.setComponent<2>(hkSimdReal_MinusMax);
		break;
	case 3:
		xpOut = projectPointOntoThreePlanes(x0, activePlanesInOut[0], activePlanesInOut[1], activePlanesInOut[2], lambdasOut);
		break;
	default:
		HK_ASSERT(0x12838, false);
		break;
	}

	// Remove a plane if necessary
	hkVector4 dists; dists.setSelect(lambdasOut.lessZero(), dots, hkVector4::getConstant<HK_QUADREAL_MAX>());
	hkVector4 negDists; negDists.setNeg<4>(dists);
	int ixMin = negDists.getIndexOfMaxComponent<3>();
	hkVector4Comparison removeOne = dists.getComponent(ixMin).less(hkSimdReal_Max);
	activePlanesInOut[ixMin].setSelect(removeOne, activePlanesInOut[numActiveInOut - 1], activePlanesInOut[ixMin]);
	activePlanesInOut[numActiveInOut - 1].setSelect(removeOne, (const hkVector4&)g_zeroMax, activePlanesInOut[numActiveInOut - 1]);
	numActiveInOut -= (removeOne.getMask() & 1);

	// Add a plane if necessary
	hkSimdReal maxDist;
	hkVector4 maxPlane;
	findFurthestPlaneAndDistance(xpOut, planes, numPlanes, maxPlane, maxDist);
	hkVector4Comparison addOne = maxDist.greater(g_eps_addPlane);
	// The next three lines of code are equivalent to 
	// int ix = lambdas.getIndexOfMinComponentReverseOrder<3>()  
	// (except that no such function exists)
	hkVector4 negLambdas; negLambdas.setNeg<4>(lambdasOut);
	hkVector4 negRevLambdas; negRevLambdas.setPermutation<hkVectorPermutation::WZYX>(negLambdas); // Reverse order, since we want the first component if there are several max values
	int ix = 3 - negLambdas.getIndexOfMaxComponent<3>();
	activePlanesInOut[ix].setSelect(addOne, maxPlane, activePlanesInOut[ix]);
	numActiveInOut = numActiveInOut + ((addOne.getMask() & 1) & (numActiveInOut < 3)); // Same as numActive = hkMath::min2(3, numActive + (addOne.getMask() & 1));

	return !addOne.anyIsSet() && !removeOne.anyIsSet();
}

void HK_CALL dualGJK_init(hkVector4Parameter x, const hkVector4* HK_RESTRICT planes, int numPlanes, 
				  hkVector4* HK_RESTRICT activePlanesOut, int& numActiveOut, hkVector4& lambdasOut, hkVector4& x1Out)
{
	numActiveOut = 1;
	hkSimdReal maxDist;
	findFurthestPlaneAndDistance(x, planes, numPlanes, activePlanesOut[0], maxDist);
	activePlanesOut[1] = (const hkVector4&)g_zeroMax;
	activePlanesOut[2] = (const hkVector4&)g_zeroMax;
	x1Out.setXYZ_W(x, hkSimdReal_1);
}

// dualGJK
bool HK_CALL hkcdDistancePointConvex::_hkcdPointConvex(hkVector4Parameter point, const hkVector4* HK_RESTRICT planes, int numPlanes, hkSimdRealParameter maxDistanceSquared, int maxNumIterations, hkVector4& pointOnSurface, hkVector4& surfaceNormal, hkSimdReal& signedDistanceOut, int& numIterationsOut)
{
	int numActive;
	hkVector4 point1;
	hkVector4 activePlanes[3];
	hkVector4 lambdas; lambdas.setZero();
	dualGJK_init(point, planes, numPlanes, activePlanes, numActive, lambdas, point1);

	hkSimdReal distance = activePlanes[0].dot<4>(point1);
	bool distanceTooBig = distance*distance > maxDistanceSquared && distance.isGreaterZero();

	hkVector4 xp;
	xp.setZero();
	numIterationsOut = 1;
	if(maxNumIterations <= 1 || distanceTooBig)
	{
		xp.setSubMul(point, activePlanes[0], distance);
	}
	else
	{
		for(numIterationsOut = 1; numIterationsOut <= maxNumIterations; numIterationsOut++) 
		{
			bool done = dualGJK_iteration(point1, planes, numPlanes, activePlanes, numActive, lambdas, xp);
			distanceTooBig = point.distanceToSquared(xp) > maxDistanceSquared;
			if(done || distanceTooBig) {
				break;
			}
		}
	}

	pointOnSurface = xp;
	surfaceNormal.setSub(point, pointOnSurface);
	signedDistanceOut = surfaceNormal.normalizeWithLength<3>();
	// If the point was inside the convex object, then invert the sign and the normal.
	if(numIterationsOut <= 1 && activePlanes[0].dot4xyz1(point).isLessZero())
	{
		signedDistanceOut = -signedDistanceOut;
		surfaceNormal.setNeg<3>(surfaceNormal);
	}
	return !distanceTooBig;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
