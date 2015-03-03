/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/RagdollLimits/hkpRagdollLimitsData.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

hkpRagdollLimitsData::hkpRagdollLimitsData()
{
	m_atoms.m_rotations.m_rotationA.setIdentity();
	m_atoms.m_rotations.m_rotationB.setIdentity();

	//m_atoms.m_twistLimit.m_limitAxis = Atoms::AXIS_TWIST;
	m_atoms.m_twistLimit.m_twistAxis = Atoms::AXIS_TWIST;
	m_atoms.m_twistLimit.m_refAxis   = Atoms::AXIS_PLANES;
	m_atoms.m_twistLimit.m_minAngle = hkReal(-0.5236f);
	m_atoms.m_twistLimit.m_maxAngle = hkReal( 0.5236f);
	m_atoms.m_twistLimit.m_angularLimitsTauFactor = hkReal(0.8f);

	m_atoms.m_coneLimit.m_twistAxisInA = Atoms::AXIS_TWIST;
	m_atoms.m_coneLimit.m_refAxisInB   = Atoms::AXIS_TWIST;
	m_atoms.m_coneLimit.m_angleMeasurementMode = hkpConeLimitConstraintAtom::ZERO_WHEN_VECTORS_ALIGNED;
	m_atoms.m_coneLimit.m_minAngle = hkReal(-100); // unlimited
	m_atoms.m_coneLimit.m_maxAngle = hkReal(1);
	m_atoms.m_coneLimit.m_angularLimitsTauFactor = hkReal(0.8f);
	setConeLimitStabilization(true);

	m_atoms.m_planesLimit.m_twistAxisInA = Atoms::AXIS_TWIST;
	m_atoms.m_planesLimit.m_refAxisInB   = Atoms::AXIS_PLANES;
	m_atoms.m_planesLimit.m_angleMeasurementMode = hkpConeLimitConstraintAtom::ZERO_WHEN_VECTORS_PERPENDICULAR;
	m_atoms.m_planesLimit.m_minAngle = hkReal(-0.025f);
	m_atoms.m_planesLimit.m_maxAngle = hkReal( 0.025f);
	m_atoms.m_planesLimit.m_angularLimitsTauFactor = hkReal(0.8f);

}

/// \param bodyA			The attached body transform
/// \param bodyB			The reference body transform
/// \param pivot			The pivot point, specified in world space.
/// \param twistAxisW		The twist axis, specified in world space.
/// \param planeAxisW		The plane axis, specified in world space.
void hkpRagdollLimitsData::setInWorldSpace(const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
										  const hkVector4& twistAxisW, const hkVector4& planeAxisW)
{
	HK_ASSERT2(0x51e44d43, hkMath::equal( twistAxisW.dot<3>(planeAxisW).getReal(), 0.0f), "twistAxisW && planeAxisW shoudl be perpendicular");

	// Set relative orientation
	{
		hkVector4 constraintBaseInW[3];

		constraintBaseInW[Atoms::AXIS_TWIST] = twistAxisW; constraintBaseInW[Atoms::AXIS_TWIST].normalize<3>();
		constraintBaseInW[Atoms::AXIS_PLANES] = planeAxisW; constraintBaseInW[Atoms::AXIS_PLANES].normalize<3>();
		constraintBaseInW[Atoms::AXIS_CROSS_PRODUCT].setCross( constraintBaseInW[Atoms::AXIS_TWIST], constraintBaseInW[Atoms::AXIS_PLANES] );

		hkVector4Util::rotateInversePoints( bodyATransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_rotations.m_rotationA.getColumn(0) );
		hkVector4Util::rotateInversePoints( bodyBTransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_rotations.m_rotationB.getColumn(0) );
	}

	HK_ASSERT2(0xadea65ff, isValid(), "Members of ragdoll constraint inconsistent.");
}

////////////////////////////////////////////////////////////////////////////////////////////////

/// \param twistAxisA		The twist axis, specified in bodyA space.
/// \param twistAxisB		The twist axis, specified in bodyB space.
/// \param planeAxisA		The plane axis, specified in bodyA space.
/// \param planeAxisB		The plane axis, specified in bodyB space.
void hkpRagdollLimitsData::setInBodySpace( const hkVector4& planeAxisA,const hkVector4& planeAxisB,
										const hkVector4& twistAxisA, const hkVector4& twistAxisB)
{
	hkVector4* HK_RESTRICT baseA = &m_atoms.m_rotations.m_rotationA.getColumn(0);
	baseA[Atoms::AXIS_TWIST] = twistAxisA; baseA[Atoms::AXIS_TWIST].normalize<3>();
	baseA[Atoms::AXIS_PLANES] = planeAxisA; baseA[Atoms::AXIS_PLANES].normalize<3>();
	baseA[Atoms::AXIS_CROSS_PRODUCT].setCross( baseA[Atoms::AXIS_TWIST], baseA[Atoms::AXIS_PLANES] );

	hkVector4* HK_RESTRICT baseB = &m_atoms.m_rotations.m_rotationB.getColumn(0);
	baseB[Atoms::AXIS_TWIST] =  twistAxisB; baseB[Atoms::AXIS_TWIST].normalize<3>();
	baseB[Atoms::AXIS_PLANES] = planeAxisB; baseB[Atoms::AXIS_PLANES].normalize<3>();
	baseB[Atoms::AXIS_CROSS_PRODUCT].setCross( baseB[Atoms::AXIS_TWIST], baseB[Atoms::AXIS_PLANES] );

	HK_ASSERT2(0xadea65fa, isValid(), "Members of ragdoll constraint inconsistent.");
}



void hkpRagdollLimitsData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}
/////////////////////////////////////////////////////////////////////////////////
void hkpRagdollLimitsData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// we need runtime data to be able to support lastAngle and friction
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
}


///////////////////////////////////////////////////////

hkBool hkpRagdollLimitsData::isValid() const
{
	hkBool valid = true;
	valid = valid && m_atoms.m_rotations.m_rotationA.isOrthonormal();
	valid = valid && m_atoms.m_rotations.m_rotationB.isOrthonormal();
	valid = valid && ( m_atoms.m_coneLimit.m_minAngle == -100.0f );
	valid = valid && ( m_atoms.m_coneLimit.m_maxAngle >= 0.0f );
	valid = valid && ( m_atoms.m_coneLimit.m_maxAngle < HK_REAL_PI );
	valid = valid && ( m_atoms.m_planesLimit.m_minAngle <= m_atoms.m_planesLimit.m_maxAngle );
	valid = valid && ( m_atoms.m_twistLimit.m_minAngle <= m_atoms.m_twistLimit.m_maxAngle );

	// Check ranges on other angular limits ??
	// No wacky behaviour seems to result

	return valid;
}

/////////////////////////////////////////////////////////////////////////////////

void hkpRagdollLimitsData::setConeLimitStabilization(hkBool enable)
{
	const int memOffset = HK_OFFSET_OF(Runtime, m_coneAngleOffset) - HK_OFFSET_OF( Runtime, m_solverResults[SOLVER_RESULT_CONE] );
	HK_ASSERT2(0XAD8755AA, memOffset == (memOffset & 0xff), "Offset doesn't fit into hkUint8.");
	m_atoms.m_coneLimit.m_memOffsetToAngleOffset = hkUint8( int(enable) * memOffset );
}

/////////////////////////////////////////////////////////////////////////////////

void hkpRagdollLimitsData::getConstraintFrameA( hkMatrix3& constraintFrameA ) const
{
	constraintFrameA = m_atoms.m_rotations.m_rotationA;
}

/////////////////////////////////////////////////////////////////////////////////

void hkpRagdollLimitsData::getConstraintFrameB( hkMatrix3& constraintFrameB ) const
{
	constraintFrameB = m_atoms.m_rotations.m_rotationB;
}

/////////////////////////////////////////////////////////////////////////////////

int hkpRagdollLimitsData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS;
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
