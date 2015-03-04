/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpRackAndPinionDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

////////////////////////////////////////////////////////////////////

void hkpRackAndPinionDrawer::drawConstraint(const hkpRackAndPinionConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	//const hkpRackAndPinionConstraintData* data = static_cast<const hkpRackAndPinionConstraintData*>(constraint->getData());
	const hkpSetLocalTransformsConstraintAtom& transforms = constraintData->m_atoms.m_transforms;
	const hkpRackAndPinionConstraintAtom& rackAndPinion = constraintData->m_atoms.m_rackAndPinion;

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	const hkTransform& refLocalToWorld = localToWorldB;
	const hkTransform& attLocalToWorld = localToWorldA;
	updateCommonParameters(localToWorldA, localToWorldB);
	m_bodyBWPivot.setTransformedPos(refLocalToWorld, transforms.m_transformB.getTranslation());
	m_bodyAWPivot.setTransformedPos(attLocalToWorld, transforms.m_transformA.getTranslation());

	//drawPivots(id, tag);

	hkVector4 angAxis = transforms.m_transformA.getRotation().getColumn<0>();
	hkVector4 angAxisPerp = transforms.m_transformA.getRotation().getColumn<1>();
	hkVector4 shiftAxis = transforms.m_transformB.getRotation().getColumn<0>();

	angAxis.setRotatedDir(attLocalToWorld.getRotation(), angAxis);
	angAxisPerp.setRotatedDir(attLocalToWorld.getRotation(), angAxisPerp);
	shiftAxis.setRotatedDir(refLocalToWorld.getRotation(), shiftAxis);

	// Draw pinion circle
	m_cogWheel.setParameters(rackAndPinion.m_pinionRadiusOrScrewPitch, 0, 2.0f * HK_REAL_PI, 18, m_bodyAWPivot, angAxis, angAxisPerp);
	hkArray<hkDisplayGeometry*> geometry;
	geometry.setSize(1);
	geometry[0] = &(m_cogWheel);
	displayHandler->displayGeometry(geometry,hkColor::WHITE, id, tag);

	hkVector4 tmp; tmp.setAddMul(m_bodyAWPivot, angAxisPerp, hkSimdReal::fromFloat(rackAndPinion.m_pinionRadiusOrScrewPitch));
	displayHandler->displayLine(m_bodyAWPivot, tmp, hkColor::WHITE, id, tag);

	if (!rackAndPinion.m_isScrew)
	{
		hkVector4 cross; cross.setCross(shiftAxis, angAxis);
		if (cross.lengthSquared<3>() > hkSimdReal_EpsSqrd)
		{
			cross.normalize<3>();
			const hkSimdReal pinionRadius = hkSimdReal::fromFloat(rackAndPinion.m_pinionRadiusOrScrewPitch);
			m_bodyBWPivot.addMul(pinionRadius, cross);
			m_bodyAWPivot.addMul(pinionRadius, cross);
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
