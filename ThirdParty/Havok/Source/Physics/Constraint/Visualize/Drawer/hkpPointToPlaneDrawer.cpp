/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpPointToPlaneDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>

////////////////////////////////////////////////////////////////////

void hkpPointToPlaneDrawer::drawConstraint(const hkpPointToPlaneConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{

	//const hkpPointToPlaneConstraintData* plane = static_cast<const hkpPointToPlaneConstraintData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, constraintData->m_atoms.m_transforms.m_transformB.getTranslation() );
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, constraintData->m_atoms.m_transforms.m_transformA.getTranslation());
	}

	drawPivots(id, tag);

	drawBodyFrames(id, tag);

		// illustrate the constraint plane attached to body A
	    // by drawing a set of lines in the plane
	{
		hkVector4 perpVec0; perpVec0.setRotatedDir(m_RB, constraintData->m_atoms.m_transforms.m_transformB.getColumn((constraintData->m_atoms.m_lin.m_axisIndex+1)%3) );
		hkVector4 perpVec1; perpVec1.setRotatedDir(m_RB, constraintData->m_atoms.m_transforms.m_transformB.getColumn((constraintData->m_atoms.m_lin.m_axisIndex+2)%3) );

		hkVector4 lines[6][2];
		for (int coord = 0; coord < 2; coord++)
		{
			for (int i = -1; i <= 1; i++)
			{
				hkVector4 v0 = coord ? perpVec0 : perpVec1;
				hkVector4 v1 = coord ? perpVec1 : perpVec0;
				lines[coord*3 + i + 1][0] = m_bodyBWPivot;
				lines[coord*3 + i + 1][1] = m_bodyBWPivot;
				lines[coord*3 + i + 1][0].addMul(hkSimdReal::fromFloat( 2.0f * getLineLengthForDraw()), v0);
				lines[coord*3 + i + 1][1].addMul(hkSimdReal::fromFloat(-2.0f * getLineLengthForDraw()), v0);
				lines[coord*3 + i + 1][0].addMul(hkSimdReal::fromFloat(hkReal(i) * 1.4f * getLineLengthForDraw()), v1);
				lines[coord*3 + i + 1][1].addMul(hkSimdReal::fromFloat(hkReal(i) * 1.4f * getLineLengthForDraw()), v1);
			}
		}

		for (int i = 0; i < 6; i++)
		{
			displayHandler->displayLine(lines[i][0],lines[i][1],hkColor::rgbFromFloats(0.8f,.8f,.8f), id, tag);
		}
	}

		// draw the pivot arm for body A
	displayHandler->displayLine(m_bodyAWPivot,m_bodyAWPos,hkColor::rgbFromFloats(0.3f,.3f,.8f), id, tag);
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
