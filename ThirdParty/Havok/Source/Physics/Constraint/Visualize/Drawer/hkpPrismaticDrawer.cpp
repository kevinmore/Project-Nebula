/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrismaticDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

////////////////////////////////////////////////////////////////////

void hkpPrismaticDrawer::drawConstraint(const hkpPrismaticConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	//const hkpPrismaticConstraintData* prismatic = static_cast<const hkpPrismaticConstraintData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, constraintData->m_atoms.m_transforms.m_transformB.getTranslation());
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, constraintData->m_atoms.m_transforms.m_transformA.getTranslation());
	}

	drawPivots(id, tag);

	drawBodyFrames(id, tag);

	hkVector4 axisInWorld;
	axisInWorld.setRotatedDir(m_RB, constraintData->m_atoms.m_transforms.m_transformB.getColumn<hkpPrismaticConstraintData::Atoms::AXIS_SHAFT>());
	hkVector4 axisPerpInWorld;
	axisPerpInWorld.setRotatedDir(m_RB, constraintData->m_atoms.m_transforms.m_transformB.getColumn<hkpPrismaticConstraintData::Atoms::AXIS_PERP_TO_SHAFT>());

	// not a full depiction of the error
	// draw a red error line between the pivots
	displayHandler->displayLine(m_bodyAWPivot, m_bodyBWPivot, hkColor::RED, id, tag);


	// draw the free DOF
	{
		hkVector4 startAxis,endAxis;
		if(constraintData->getMaxLinearLimit() != HK_REAL_MAX)
		{
			endAxis.setAddMul(m_bodyBWPivot, axisInWorld, hkSimdReal::fromFloat(constraintData->getMaxLinearLimit()));
			startAxis.setAddMul(m_bodyBWPivot, axisInWorld, hkSimdReal::fromFloat(constraintData->getMinLinearLimit()));
			displayHandler->displayLine(startAxis, endAxis, hkColor::rgbFromFloats(0.f, .5f, 1.f), id, tag);
		}
		else
		{
			endAxis.setAddMul(m_bodyBWPivot, axisInWorld, hkSimdReal::fromFloat(getLineLengthForDraw()));
			startAxis.setAddMul(m_bodyBWPivot, axisInWorld, -hkSimdReal::fromFloat(getLineLengthForDraw()));
			displayHandler->displayLine(startAxis, endAxis, hkColor::rgbFromFloats(0.f, .5f, 1.f), id, tag);
		}
	}

	// draw limits in white
	{
		hkVector4 startLine;
		hkVector4 endLine;
		hkVector4 center;
		hkVector4 temp;

		hkReal minLimit = constraintData->getMinLinearLimit();
		hkReal maxLimit = constraintData->getMaxLinearLimit();

		temp.setMul(hkSimdReal::fromFloat(minLimit), axisInWorld);
		center.setAdd(m_bodyBWPivot,temp);
		temp.setMul(hkSimdReal::fromFloat(.5f * getLineLengthForDraw()), axisPerpInWorld);
		startLine.setAdd(center,temp);
		temp.setMul(hkSimdReal::fromFloat(-.5f * getLineLengthForDraw()), axisPerpInWorld);
		endLine.setAdd(center, temp);
		displayHandler->displayLine(startLine, endLine, hkColor::WHITE, id, tag);

		temp.setMul(hkSimdReal::fromFloat(maxLimit), axisInWorld);
		center.setAdd(m_bodyBWPivot,temp);
		temp.setMul(hkSimdReal::fromFloat(.5f * getLineLengthForDraw()), axisPerpInWorld);
		startLine.setAdd(center,temp);
		temp.setMul(hkSimdReal::fromFloat(-.5f * getLineLengthForDraw()), axisPerpInWorld);
		endLine.setAdd(center, temp);
		displayHandler->displayLine(startLine, endLine, hkColor::WHITE, id, tag);
	}
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
