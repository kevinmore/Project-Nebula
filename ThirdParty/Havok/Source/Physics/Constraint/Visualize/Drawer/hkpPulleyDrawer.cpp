/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPulleyDrawer.h>
#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>


void hkpPulleyDrawer::drawConstraint(const hkpPulleyConstraintData* constraintData, const hkTransform& localToWorldA, const hkTransform& localToWorldB, hkDebugDisplayHandler* displayHandler, int id, int tag)
{
	//const hkpPulleyConstraintData* pulley = static_cast<const hkpPulleyConstraintData*>(constraint->getData());

	m_primitiveDrawer.setDisplayHandler(displayHandler);

	{
		const hkTransform& refLocalToWorld = localToWorldB;
		const hkTransform& attLocalToWorld = localToWorldA;
		updateCommonParameters(localToWorldA, localToWorldB);
		m_bodyBWPivot.setTransformedPos(refLocalToWorld, constraintData->m_atoms.m_translations.m_translationB);
		m_bodyAWPivot.setTransformedPos(attLocalToWorld, constraintData->m_atoms.m_translations.m_translationA);
	}

	drawPivots(id, tag);

	drawBodyFrames(id, tag);

	const hkpPulleyConstraintAtom& pulleyAtom = constraintData->m_atoms.m_pulley;

	displayHandler->displayLine(pulleyAtom.m_fixedPivotAinWorld, pulleyAtom.m_fixedPivotBinWorld, hkColor::GREY75, id, tag);
	displayHandler->displayLine(m_bodyAWPivot, pulleyAtom.m_fixedPivotAinWorld, hkColor::WHITE, id, tag);

	int numLines = hkMath::hkFloatToInt(hkMath::max2( pulleyAtom.m_leverageOnBodyB, hkReal(1.0f) ));
	hkVector4 shift; shift.set(0.07f, 0, 0);
	shift.setRotatedDir(localToWorldB.getRotation(), shift);

	hkVector4 base; base.setMul( hkSimdReal::fromInt32( numLines-1 ) * -hkSimdReal_Inv2, shift );

	for (int i = 0; i < numLines; i++)
	{
		hkVector4 thisShift; thisShift.setAddMul( base, shift, hkSimdReal::fromInt32(i) );
		hkVector4 a; a.setAdd(m_bodyBWPivot, thisShift);
		hkVector4 b; b.setAdd(pulleyAtom.m_fixedPivotBinWorld, thisShift);
		displayHandler->displayLine(a, b, hkColor::WHITE, id, tag);
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
