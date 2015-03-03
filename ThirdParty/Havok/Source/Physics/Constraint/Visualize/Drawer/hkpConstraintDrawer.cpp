/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics/Constraint/Visualize/Drawer/hkpConstraintDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>



hkpConstraintDrawer::hkpConstraintDrawer()
{
	m_numArcSegments = 24;
	m_pivotSize = .25f;
	m_arcRadius = 0.75f;
	m_lineLength = 1.f;
	m_arrowSize = 1.0f;
	m_coneSize = 0.5f;
	m_planeSize = 0.25f;
	m_scale = 1.0f;
}


void hkpConstraintDrawer::drawPivots(int id, int tag)
{
	// Draw pivot points
	m_primitiveDrawer.displayOrientedPoint(m_bodyAWPivot, m_RA, getPivotSizeForDraw(), hkColor::RED, id, tag);
	m_primitiveDrawer.displayOrientedPoint(m_bodyBWPivot, m_RB, getPivotSizeForDraw(), hkColor::WHITE, id, tag);
}


void hkpConstraintDrawer::drawBodyFrames(int id, int tag)
{
	hkUint32 yellow,orange,purple;

	yellow = hkColor::rgbFromFloats(1.f,1.f,0.f);
	orange = hkColor::rgbFromFloats(1.f,.5f,0.f);
	purple = hkColor::rgbFromFloats(1.f,0.f,1.f);


	hkVector4 Ax = m_RA.getColumn<0>();
	hkVector4 Ay = m_RA.getColumn<1>();
	hkVector4 Az = m_RA.getColumn<2>();

	hkVector4 Bx = m_RB.getColumn<0>();
	hkVector4 By = m_RB.getColumn<1>();
	hkVector4 Bz = m_RB.getColumn<2>();

	// Draw coordinate frames
	m_primitiveDrawer.displayArrow(m_bodyAWPos, Ax, Ay, hkColor::RED, getArrowSizeForDraw() * 0.2f, id, tag);
	m_primitiveDrawer.displayArrow(m_bodyAWPos, Ay, Az, hkColor::GREEN, getArrowSizeForDraw() * 0.2f, id, tag);
	m_primitiveDrawer.displayArrow(m_bodyAWPos, Az, Ax, hkColor::BLUE, getArrowSizeForDraw() * 0.2f, id, tag);
	m_primitiveDrawer.displayArrow(m_bodyBWPos, Bx, By, yellow, getArrowSizeForDraw() * 0.2f, id, tag);
	m_primitiveDrawer.displayArrow(m_bodyBWPos, By, Bz, orange, getArrowSizeForDraw() * 0.2f, id, tag);
	m_primitiveDrawer.displayArrow(m_bodyBWPos, Bz, Bx, purple, getArrowSizeForDraw() * 0.2f, id, tag);
}



void hkpConstraintDrawer::setLineLength(hkReal lineLength)
{
	m_lineLength = lineLength;
}

void hkpConstraintDrawer::setArcRadius(hkReal arcRadius)
{
	m_arcRadius = arcRadius;
}

void hkpConstraintDrawer::setNumArcSegments(int numSegments)
{
	m_numArcSegments = numSegments;
}

void hkpConstraintDrawer::setPivotSize(hkReal size)
{
	m_pivotSize = size;
}

void hkpConstraintDrawer::setArrowSize(hkReal arrrowSize)
{
	m_arrowSize = arrrowSize;
}

void hkpConstraintDrawer::setScale(hkReal scale)
{
	m_scale = scale;
}


void hkpConstraintDrawer::updateCommonParameters(const hkTransform& attLocalToWorld, const hkTransform& refLocalToWorld)
{
	m_RB = refLocalToWorld.getRotation();
	m_RA = attLocalToWorld.getRotation();

	m_bodyBWPos = refLocalToWorld.getTranslation();
	m_bodyAWPos = attLocalToWorld.getTranslation();
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
