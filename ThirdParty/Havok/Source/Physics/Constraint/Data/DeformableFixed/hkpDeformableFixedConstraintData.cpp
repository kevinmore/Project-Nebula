/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>


hkpDeformableFixedConstraintData::hkpDeformableFixedConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	// Set an infinite strength
	hkSymmetricMatrix3 t;
	t.setDiagonal(hkSimdReal::getConstant<HK_QUADREAL_MAX>());

	setLinearStrength(t, t);
	setAngularStrength(t, t);
}

void hkpDeformableFixedConstraintData::setInWorldSpace(const hkTransform& bodyATransform, const hkTransform& bodyBTransform, const hkTransform& pivotA, const hkTransform& pivotB)
{
	m_atoms.m_transforms.m_transformA._setMulInverseMul(bodyATransform, pivotA);
	m_atoms.m_transforms.m_transformB._setMulInverseMul(bodyBTransform, pivotB);

	// Compute initial offsets
	hkVector4 vLinOffset;
	vLinOffset.setSub(pivotB.getTranslation(), pivotA.getTranslation());
	vLinOffset._setRotatedInverseDir(bodyATransform.getRotation(), vLinOffset);

	hkRotation relRotAB;		relRotAB.setMulInverseMul(pivotA.getRotation(), pivotB.getRotation());
	hkQuaternion qAngOffset;	qAngOffset.setAndNormalize(relRotAB);

	// Set-up the Destruction constraint atom
	m_atoms.m_lin.setOffset(vLinOffset);
	m_atoms.m_ang.setOffset(qAngOffset);
}

void hkpDeformableFixedConstraintData::setInBodySpace(const hkTransform& pivotA, const hkTransform& pivotB)
{
	m_atoms.m_transforms.m_transformA = pivotA;
	m_atoms.m_transforms.m_transformB = pivotB;
}

void hkpDeformableFixedConstraintData::getConstraintInfo(hkpConstraintData::ConstraintInfo& infoOut) const
{
	getConstraintInfoUtil(m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut);
}

void hkpDeformableFixedConstraintData::getRuntimeInfo(hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut) const
{
	if ( wantRuntime )
	{
		infoOut.m_numSolverResults		= SOLVER_RESULT_MAX;
		infoOut.m_sizeOfExternalRuntime = sizeof(Runtime);
	}
	else
	{
		infoOut.m_numSolverResults		= 0;
		infoOut.m_sizeOfExternalRuntime	= 0;
	}
}

hkBool hkpDeformableFixedConstraintData::isValid() const
{
	return true;
}

int hkpDeformableFixedConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED;
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
