/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpHingeDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>


////////////////////////////////////////////////////////////////////

void hkpHingeDrawer::drawConstraint(const hkpHingeConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, constraintData->m_atoms.m_transforms.m_transformB.getTranslation());
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, constraintData->m_atoms.m_transforms.m_transformA.getTranslation());
	}

	drawBodyFrames(id, tag);

	drawPivots(id, tag);

	// not a complete representation of error
	// but some feedback
	// draw a red error line between the pivots
	displayHandler->displayLine(m_bodyAWPivot,m_bodyBWPivot,hkColor::RED, id, tag);

	// Draw the free axis of rotation
	{
		hkVector4 axisInWorld;
		axisInWorld._setRotatedDir(m_RB, constraintData->m_atoms.m_transforms.m_transformB.getColumn<hkpHingeConstraintData::Atoms::AXIS_AXLE>());// m_basisB.m_axle);

		hkVector4 startAxis,endAxis;
		endAxis.setMul(hkSimdReal::fromFloat(.75f * getLineLengthForDraw()), axisInWorld);
		endAxis.add(m_bodyBWPivot);
		startAxis.setMul(hkSimdReal::fromFloat(-.75f * getLineLengthForDraw()), axisInWorld);
		startAxis.add(m_bodyBWPivot);
		displayHandler->displayLine(startAxis, endAxis, hkColor::rgbFromFloats(0.f, .5f, 1.f), id, tag);
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
