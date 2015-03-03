/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Pulley/hkpPulleyConstraintData.h>


hkpPulleyConstraintData::hkpPulleyConstraintData()
{
	m_atoms.m_translations.m_translationA.setZero();
	m_atoms.m_translations.m_translationB.setZero();
	m_atoms.m_pulley.m_fixedPivotAinWorld.setZero();
	m_atoms.m_pulley.m_fixedPivotBinWorld.setZero();
	m_atoms.m_pulley.m_ropeLength = hkReal(0);
	m_atoms.m_pulley.m_leverageOnBodyB = hkReal(1);
}

void hkpPulleyConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpPulleyConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	if ( wantRuntime )
	{
		infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
		infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
	}
	else
	{
		infoOut.m_numSolverResults = 0;
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}

hkBool hkpPulleyConstraintData::isValid() const
{
	return m_atoms.m_pulley.m_ropeLength > hkReal(0) && m_atoms.m_pulley.m_leverageOnBodyB > hkReal(0);
}

int hkpPulleyConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_PULLEY;
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
