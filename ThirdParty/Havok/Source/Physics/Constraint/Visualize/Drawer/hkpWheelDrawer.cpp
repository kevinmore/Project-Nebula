/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics/Constraint/Visualize/Drawer/hkpWheelDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>


void hkpWheelDrawer::drawConstraint(const hkpWheelConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	//const hkpWheelConstraintData* wheel = static_cast<const hkpWheelConstraintData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot._setTransformedPos(refLocalToWorld, constraintData->m_atoms.m_suspensionBase.m_transformB.getTranslation());
		m_bodyAWPivot._setTransformedPos(attLocalToWorld, constraintData->m_atoms.m_suspensionBase.m_transformA.getTranslation());
	}

	drawBodyFrames(id, tag);

	drawPivots(id, tag);

	// draws red line between pivot on A and B in world space
	displayHandler->displayLine(m_bodyAWPivot, m_bodyBWPivot, hkColor::RED, id, tag);

	hkSimdReal shortDrawLength; shortDrawLength.setFromFloat(.1f * getLineLengthForDraw());
	hkSimdReal longDrawLength;  longDrawLength.setFromFloat(.5f * getLineLengthForDraw());

	hkVector4 suspensionAxis;
	suspensionAxis._setRotatedDir(m_RB, constraintData->m_atoms.m_suspensionBase.m_transformB.getColumn<hkpWheelConstraintData::Atoms::AXIS_SUSPENSION>());// m_tempWheelAtom.m_basisB.m_suspensionAxis);

	// draw suspension axis
	{
		hkVector4 p1;
		hkVector4 p2;
		p1.setAddMul(m_bodyBWPivot, suspensionAxis, -longDrawLength);
		p2.setAddMul(m_bodyBWPivot, suspensionAxis,  longDrawLength);
		displayHandler->displayLine(p1, p2, hkColor::GREEN, id, tag);
	}

	// draw suspension limits
	{
		hkVector4 p1;
		hkVector4 start;
		hkVector4 end;

		hkReal suspensionMin = constraintData->m_atoms.m_lin0Limit.m_min;
		hkReal suspensionMax = constraintData->m_atoms.m_lin0Limit.m_max;

		hkVector4 perpSuspension;
		perpSuspension._setRotatedDir(m_RB, constraintData->m_atoms.m_suspensionBase.m_transformB.getColumn<hkpWheelConstraintData::Atoms::AXIS_PERP_SUSPENSION>());// m_tempWheelAtom.m_basisB.m_perpToSuspensionAxis);

		p1.setAddMul(m_bodyBWPivot, suspensionAxis, hkSimdReal::fromFloat(suspensionMax));

		start.setAddMul(p1, perpSuspension,  shortDrawLength);
		end.setAddMul(p1,   perpSuspension, -shortDrawLength);
		displayHandler->displayLine(start,end, hkColor::WHITE, id, tag);

		p1.setAddMul(m_bodyBWPivot, suspensionAxis, hkSimdReal::fromFloat(suspensionMin));

		start.setAddMul(p1, perpSuspension,  shortDrawLength);
		end.setAddMul(p1,   perpSuspension, -shortDrawLength);
		displayHandler->displayLine(start, end, hkColor::WHITE, id, tag);
	}

	// draw steering axis
	{
		hkVector4 p1;
		hkVector4 p2;
		hkVector4 steeringAxis;
		steeringAxis._setRotatedDir(m_RB, constraintData->m_atoms.m_steeringBase.m_rotationB.getColumn<hkpWheelConstraintData::Atoms::AXIS_STEERING>());//  m_tempWheelAtom.m_basisB.m_steeringAxis);
		p1.setAddMul(m_bodyBWPivot, steeringAxis, -longDrawLength);
		p2.setAddMul(m_bodyBWPivot, steeringAxis,  longDrawLength);
		displayHandler->displayLine(p1, p2, hkColor::YELLOW, id, tag);
	}

	// draw axle
	{
		hkVector4 p1;
		hkVector4 p2;
		hkVector4 axle;
		axle._setRotatedDir(m_RA, constraintData->m_atoms.m_steeringBase.m_rotationA.getColumn<hkpWheelConstraintData::Atoms::AXIS_AXLE>()); //m_atoms.m_steeringBase.m_rotationA.getColumn(hkpWheelConstraintData::Atoms::AXIS_AXLE));

		p1.setAddMul(m_bodyAWPivot, axle, -longDrawLength);
		p2.setAddMul(m_bodyAWPivot, axle,  longDrawLength);
		displayHandler->displayLine(p1, p2, hkColor::BLUE, id, tag);
	}
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
