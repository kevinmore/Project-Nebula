/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Rotational/hkpRotationalConstraintData.h>


hkpRotationalConstraintData::hkpRotationalConstraintData()
{
	m_atoms.m_rotations.m_rotationA.setIdentity();
	m_atoms.m_rotations.m_rotationB.setIdentity();

	m_atoms.m_ang.m_firstConstrainedAxis = 0;
	m_atoms.m_ang.m_numConstrainedAxes   = 3;
}

void hkpRotationalConstraintData::setInWorldSpace(const hkQuaternion& bodyARotation, const hkQuaternion& bodyBRotation)
{
	m_atoms.m_rotations.m_rotationA.set(bodyARotation);
	m_atoms.m_rotations.m_rotationB.set(bodyBRotation);
}

void hkpRotationalConstraintData::setInBodySpace(const hkQuaternion& aTb)
{
	m_atoms.m_rotations.m_rotationA.setIdentity();
	m_atoms.m_rotations.m_rotationB.set(aTb);
}

void hkpRotationalConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpRotationalConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
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

hkBool hkpRotationalConstraintData::isValid() const
{
	return m_atoms.m_rotations.m_rotationA.isOrthonormal() && m_atoms.m_rotations.m_rotationB.isOrthonormal();
}

int hkpRotationalConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL;
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
