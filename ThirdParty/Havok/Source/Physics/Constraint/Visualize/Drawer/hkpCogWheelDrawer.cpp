/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpCogWheelDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

////////////////////////////////////////////////////////////////////

void hkpCogWheelDrawer::drawConstraint(const hkpCogWheelConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	const hkpSetLocalTransformsConstraintAtom& transforms = constraintData->m_atoms.m_transforms;
	const hkpCogWheelConstraintAtom& cogWheels = constraintData->m_atoms.m_cogWheels;

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	const hkTransform& refLocalToWorld = localToWorldB;
	const hkTransform& attLocalToWorld = localToWorldA;
	updateCommonParameters(localToWorldA, localToWorldB);
	m_bodyBWPivot.setTransformedPos(refLocalToWorld, transforms.m_transformB.getTranslation());
	m_bodyAWPivot.setTransformedPos(attLocalToWorld, transforms.m_transformA.getTranslation());

	//drawPivots(id, tag);

	hkVector4 angAxisA = transforms.m_transformA.getRotation().getColumn<0>();
	hkVector4 angAxisPerpA = transforms.m_transformA.getRotation().getColumn<1>();
	hkVector4 angAxisB = transforms.m_transformB.getRotation().getColumn<0>();
	hkVector4 angAxisPerpB = transforms.m_transformB.getRotation().getColumn<1>();

	angAxisA.setRotatedDir(attLocalToWorld.getRotation(), angAxisA);
	angAxisPerpA.setRotatedDir(attLocalToWorld.getRotation(), angAxisPerpA);
	angAxisB.setRotatedDir(refLocalToWorld.getRotation(), angAxisB);
	angAxisPerpB.setRotatedDir(refLocalToWorld.getRotation(), angAxisPerpB);

	// Draw pinion circle
	m_cogWheels[0].setParameters(cogWheels.m_cogWheelRadiusA, 0, 2.0f * HK_REAL_PI, 18, m_bodyAWPivot, angAxisA, angAxisPerpA);
	m_cogWheels[1].setParameters(cogWheels.m_cogWheelRadiusB, 0, 2.0f * HK_REAL_PI, 18, m_bodyBWPivot, angAxisB, angAxisPerpB);
	hkArray<hkDisplayGeometry*> geometry;
	geometry.setSize(2);
	geometry[0] = &(m_cogWheels[0]);
	geometry[1] = &(m_cogWheels[1]);
	displayHandler->displayGeometry(geometry,hkColor::WHITE, id, tag);

	hkVector4 tmp;
	tmp.setAddMul(m_bodyAWPivot, angAxisPerpA, hkSimdReal::fromFloat(cogWheels.m_cogWheelRadiusA));
	displayHandler->displayLine(m_bodyAWPivot, tmp, hkColor::WHITE, id, tag);
	tmp.setAddMul(m_bodyBWPivot, angAxisPerpB, hkSimdReal::fromFloat(cogWheels.m_cogWheelRadiusB));
	displayHandler->displayLine(m_bodyBWPivot, tmp, hkColor::WHITE, id, tag);


	{
		// Calculate direction from pivots to actual contact points
		hkVector4 posDiff; posDiff.setSub(m_bodyBWPivot, m_bodyAWPivot);
		const hkSimdReal posDiffLength2 = posDiff.lengthSquared<3>();
		const hkSimdReal eps2 = hkSimdReal_EpsSqrd;

		if(posDiffLength2 >= eps2)
		{
			hkVector4 radiusVecA;
			hkVector4 radiusVecB;
			{
				hkVector4 crossA; crossA.setCross(posDiff, angAxisA);
				hkVector4 crossB; crossB.setCross(angAxisB, posDiff);
				radiusVecA.setCross(angAxisA, crossA);
				radiusVecB.setCross(angAxisB, crossB);

				const hkSimdReal lenA2 = radiusVecA.lengthSquared<3>();
				const hkSimdReal lenB2 = radiusVecB.lengthSquared<3>();

				if(lenA2 >= eps2 && lenB2 >= eps2)
				{
					const hkSimdReal invLenA = lenA2.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
					const hkSimdReal invLenB = lenB2.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
					radiusVecA.mul(invLenA);
					radiusVecB.mul(invLenB);

					// Calculate actual contact positions
					m_bodyAWPivot.addMul(hkSimdReal::fromFloat(cogWheels.m_cogWheelRadiusA), radiusVecA);
					m_bodyBWPivot.addMul(hkSimdReal::fromFloat(cogWheels.m_cogWheelRadiusB), radiusVecB);
				}

			}

		}


	}

	drawPivots(id, tag);

	//drawBodyFrames(id, tag);

	// not a full depiction of the error
	// draw a red error line between the pivots
	displayHandler->displayLine(m_bodyAWPivot, m_bodyBWPivot, hkColor::RED, id, tag);



}

////////////////////////////////////////////////////////////////////

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
