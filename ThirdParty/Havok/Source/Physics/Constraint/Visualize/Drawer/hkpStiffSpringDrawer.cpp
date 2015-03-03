/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpStiffSpringDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>


///////////////////////////////////////////////////////////////////

void hkpStiffSpringDrawer::drawConstraint(const hkpStiffSpringConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	//const hkpStiffSpringConstraintData* spring = static_cast<const hkpStiffSpringConstraintData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, constraintData->m_atoms.m_pivots.m_translationB);
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, constraintData->m_atoms.m_pivots.m_translationA);
	}

	drawPivots(id, tag);

	drawBodyFrames(id, tag);

	// draw the error
	// first draw the spring segment at its proper length in blue
	// draw the spring segment between the min and max rest length in green
	// then draw error..if any in red
	{
		hkVector4 dir;
		hkSimdReal dist;
		dir.setSub(m_bodyBWPivot,m_bodyAWPivot);
		dist = dir.length<3>();
		dir.normalize<3>();

		hkSimdReal springLength;
		springLength.setMin(dist, hkSimdReal::fromFloat(constraintData->getSpringMaxLength()));
		springLength.setMax(dist, hkSimdReal::fromFloat(constraintData->getSpringMinLength()));

		hkVector4 minDir = dir;
		minDir.setMul(hkSimdReal::fromFloat(constraintData->getSpringMinLength()),dir);
		minDir.add(m_bodyAWPivot);

		dir.setMul(springLength,dir);
		dir.add(m_bodyAWPivot);

		displayHandler->displayLine(m_bodyAWPivot,minDir,hkColor::rgbFromFloats(0,.5f,1), id, tag);
		displayHandler->displayLine(minDir,dir,hkColor::rgbFromFloats(0,1.0f,0.4f), id, tag);
		displayHandler->displayLine(dir,m_bodyBWPivot,hkColor::RED, id, tag);
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
