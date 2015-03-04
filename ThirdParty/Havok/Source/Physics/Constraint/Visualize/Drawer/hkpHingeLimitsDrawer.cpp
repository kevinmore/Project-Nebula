/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpHingeLimitsDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>


////////////////////////////////////////////////////////////////////

// Nearly exact copy of the code from hkpLimitedHingeDrawer. (With exception of the pivot vectors being zeroed.)
void hkpHingeLimitsDrawer::drawConstraint(const hkpHingeLimitsData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, hkVector4::getZero());
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, hkVector4::getZero());
	}

	drawBodyFrames(id, tag);

	drawPivots(id, tag);

	const hkVector4* baseA = &constraintData->m_atoms.m_rotations.m_rotationA.getColumn(0);
	const hkVector4* baseB = &constraintData->m_atoms.m_rotations.m_rotationB.getColumn(0);

	hkVector4 axisInWorld;
	axisInWorld.setRotatedDir(m_RB, baseB[hkpHingeLimitsData::Atoms::AXIS_AXLE]);

	hkVector4 axisPerpInWorld;
	axisPerpInWorld.setRotatedDir(m_RB, baseB[hkpHingeLimitsData::Atoms::AXIS_PERP_TO_AXLE_2]);


	// draw a red error line between the pivots
	displayHandler->displayLine(m_bodyAWPivot, m_bodyBWPivot, hkColor::RED, id, tag);

	// draw the free DOF
	{
		hkVector4 startAxis,endAxis;
		endAxis.setMul(hkSimdReal::fromFloat(.75f * getLineLengthForDraw()), axisInWorld);
		endAxis.add(m_bodyBWPivot);
		startAxis.setMul(hkSimdReal::fromFloat(-.75f * getLineLengthForDraw()), axisInWorld);
		startAxis.add(m_bodyBWPivot);
		displayHandler->displayLine(startAxis, endAxis, hkColor::rgbFromFloats(0.f, .5f, 1.f), id, tag);
	}

	// draw the limits
	{
		hkReal thetaMax = constraintData->getMaxAngularLimit();
		hkReal thetaMin = constraintData->getMinAngularLimit();

		m_angularLimit.setParameters(getArcRadiusForDraw(), thetaMin, thetaMax, getNumArcSegmentsForDraw(), m_bodyBWPivot, axisInWorld, axisPerpInWorld);
		hkArray<hkDisplayGeometry*> geometry;
		geometry.setSize(1);
		geometry[0] = &(m_angularLimit);
		displayHandler->displayGeometry(geometry,hkColor::WHITE, id, tag);
	}

	// draw a line representing m_axisPerpInWorld to which the angle
	// is with respect to.
	{
		hkVector4 start;
		hkVector4 end;
		start = m_bodyBWPivot;
		end = start;
		end.addMul(hkSimdReal::fromFloat(0.75f * getLineLengthForDraw()), axisPerpInWorld);
		displayHandler->displayLine(start, end, hkColor::YELLOW, id, tag);

		hkVector4 axisPerpA;
		axisPerpA.setRotatedDir(m_RA, baseA[hkpHingeLimitsData::Atoms::AXIS_PERP_TO_AXLE_2]);
		axisPerpA.normalize<3>();

		start = m_bodyAWPivot;
		end = start;
		end.addMul(hkSimdReal::fromFloat(0.5f * getLineLengthForDraw()), axisPerpA);
		displayHandler->displayLine(start, end, hkColor::YELLOW, id, tag);
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
