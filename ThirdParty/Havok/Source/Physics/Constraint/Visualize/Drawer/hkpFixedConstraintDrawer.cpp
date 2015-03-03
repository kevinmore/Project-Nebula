/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/Visualize/Drawer/hkpFixedConstraintDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>

#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Base/Types/Color/hkColor.h>


void hkpFixedConstraintDrawer::drawConstraint(const hkpFixedConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	const hkpSetLocalTransformsConstraintAtom& pivots = constraintData->m_atoms.m_transforms;

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, pivots.m_transformB.getTranslation());
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, pivots.m_transformA.getTranslation());
	}

	drawPivots(id, tag);

	drawBodyFrames(id, tag);

	// Draw Error
	displayHandler->displayLine(m_bodyAWPivot, m_bodyBWPivot, hkColor::RED, id, tag);
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
