
/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics/Internal/Collide/Gsk/hknpGskUtil.h>
#include <Physics/Physics/Collide/hknpCdBody.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

#include <Physics/Physics/Collide/hknpCollideSharedData.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>


HK_FORCE_INLINE int hknpCvxCollisionProcessHelper::calcAngTim( const int minAngleA, const int minAngleB, hkVector4Parameter planeAinA, hkVector4Parameter planeBinA )
{
	int currentAngle	=	hknpMotionUtil::convertAngleToAngularTIM( planeAinA, planeBinA );
	int	minTIM			=	hkMath::min2(minAngleA, minAngleB);
	int finTIM			=	hkMath::max2(0, minTIM-currentAngle);
	return finTIM;
}

// TODO: Optimize this entire function
HK_FORCE_INLINE int hknpCvxCollisionProcessHelper::calcAngTim_2(
	const hknpConvexPolytopeShape* HK_RESTRICT shapeA, const hknpConvexPolytopeShape* HK_RESTRICT shapeB, int faceIndexA, int faceIndexB, hkVector4Parameter planeAinA, hkVector4Parameter planeBinA,
	hkVector4Parameter normalInA, const hkRotation& aTbRotation)
{
	// Compute more accurate angularTim
	hkVector4 planeA2inA; // Second-closest plane in A
	{
		const hkVector4* HK_RESTRICT planesA = shapeA->getPlanes();
		hkSimdReal maxA = -hkSimdReal_Max;
		for(int i=0; i<shapeA->getNumFaces(); i++)
		{
			hkVector4 planeA_2 = planesA[i];
			hkSimdReal dot = planeAinA.dot<3>(planeA_2);
			if(i != faceIndexA && dot > maxA) // Possible optimization: Instead of i != faceIndexA use dot < dotFaceA
			{
				planeA2inA = planeA_2;
				maxA = dot;
			}
		}
	}

	hkVector4 planeB2inA; // Second-closest plane in B
	{
		hkVector4 planeB2inB;
		hkVector4 dirBinB; dirBinB._setRotatedInverseDir(aTbRotation, planeBinA);
		const hkVector4* HK_RESTRICT planesB = shapeB->getPlanes();
		hkSimdReal maxB = -hkSimdReal_Max;
		for(int i=0; i<shapeB->getNumFaces(); i++)
		{
			hkSimdReal dot = dirBinB.dot<3>(planesB[i]);
			if(i != faceIndexB && dot > maxB) // Possible optimization: Instead of i != faceIndexB use dot < dotFaceB
			{
				planeB2inB = planesB[i];
				maxB = dot;
			}
		}
		planeB2inA._setRotatedDir( aTbRotation, planeB2inB);
	}

	hkVector4 meanA; meanA.setAdd(planeAinA, planeA2inA);
	meanA.normalizeIfNotZero<3>();
	meanA.setNeg<3>(meanA);
	int diffA = hknpMotionUtil::convertAngleToAngularTIM( normalInA, meanA ); // This is the angle that shape A can move freely

	hkVector4 meanB; meanB.setAdd(planeBinA, planeB2inA);
	meanB.normalizeIfNotZero<3>();
	int diffB = hknpMotionUtil::convertAngleToAngularTIM( normalInA, meanB ); // This is the angle that shape B can move freely

	int minDiff = hkMath::min2(diffA, diffB); // This is the angle that shapes A and B can move freely
	return minDiff;
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
