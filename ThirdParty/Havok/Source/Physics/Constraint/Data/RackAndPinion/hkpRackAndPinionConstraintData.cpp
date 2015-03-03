/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/RackAndPinion/hkpRackAndPinionConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpRackAndPinionConstraintData::hkpRackAndPinionConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_rackAndPinion.m_pinionRadiusOrScrewPitch = hkReal(1);
	m_atoms.m_rackAndPinion.m_isScrew = false;
	m_atoms.m_rackAndPinion.m_memOffsetToInitialAngleOffset = HK_OFFSET_OF(Runtime, m_initialAngleOffset);
	m_atoms.m_rackAndPinion.m_memOffsetToPrevAngle = HK_OFFSET_OF(Runtime, m_prevAngle);
	m_atoms.m_rackAndPinion.m_memOffsetToRevolutionCounter = HK_OFFSET_OF(Runtime, m_revolutionCounter);
}


void hkpRackAndPinionConstraintData::setInWorldSpace(
											const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
											const hkVector4& pinionARotationPivot, const hkVector4& pinionARotationAxis,
											const hkVector4& rackBShiftAxis, hkReal pinionRadiusOrScrewPitch,
											Type type )
{
	HK_ASSERT2(0xad234232,    hkMath::equal(pinionARotationAxis.lengthSquared<3>().getReal(), 1.0f)
						   && hkMath::equal(rackBShiftAxis.lengthSquared<3>().getReal(), 1.0f),
						   "Input axes must be normalized.");

	// Set bases: baseA.getColumn(0) is the axis of rotation, baseB.getColumn(0) is the axis of shift.
	// The other vectors of the base are used to determine the angles of relative orientation.
	{
		hkVector4 baseA[3]; // Base A in world.
		hkVector4 baseB[3]; // Base B in world, note that the base of the two bodies are not aligned in this constraint.

		// Initialize constraint bases in world space.
		baseA[0] = pinionARotationAxis;
		baseB[0] = rackBShiftAxis;
		hkVector4Util::calculatePerpendicularVector( baseA[0], baseA[1] );
		hkVector4Util::calculatePerpendicularVector( baseB[0], baseB[1] );
		baseA[1].normalize<3>();
		baseB[1].normalize<3>();
		baseA[2].setCross(baseA[0], baseA[1]);
		baseB[2].setCross(baseB[0], baseB[1]);

		// Rotate constraint bases to local space of the bodies.
		hkVector4* dstRotationA = &m_atoms.m_transforms.m_transformA.getRotation().getColumn(0);
		hkVector4* dstRotationB = &m_atoms.m_transforms.m_transformB.getRotation().getColumn(0);
		hkVector4Util::rotateInversePoints( bodyATransform.getRotation(), baseA, 3, dstRotationA );
		hkVector4Util::rotateInversePoints( bodyBTransform.getRotation(), baseB, 3, dstRotationB );
	}

	// Set pivot points
	m_atoms.m_transforms.m_transformA.getTranslation().setTransformedInversePos( bodyATransform, pinionARotationPivot );
	m_atoms.m_transforms.m_transformB.getTranslation().setTransformedInversePos( bodyBTransform, pinionARotationPivot );

	m_atoms.m_rackAndPinion.m_pinionRadiusOrScrewPitch = pinionRadiusOrScrewPitch;
	m_atoms.m_rackAndPinion.m_isScrew = (type == TYPE_SCREW);

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of rack-and-pinion constraint inconsistent.");
}

void hkpRackAndPinionConstraintData::setInBodySpace(
									const hkVector4& pinionARotationPivotInA, const hkVector4& pinionARotationPivotInB,
									const hkVector4& pinionARotationAxisInA, const hkVector4& rackBShiftAxisInB,
									hkReal pinionRadiusOrScrewPitch, Type type)
{
	HK_ASSERT2(0xad234231,    hkMath::equal(pinionARotationAxisInA.lengthSquared<3>().getReal(), 1.0f)
						   && hkMath::equal(rackBShiftAxisInB.lengthSquared<3>().getReal(), 1.0f),
						   "Input axes must be normalized.");

	// Set bases: baseA.getColumn(0) is the axis of rotation, baseB.getColumn(0) is the axis of shift.
	// The other vectors of the base are used to determine the angles of relative orientation.
	{
		hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getRotation().getColumn(0); // Base A in bodyA space.
		hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getRotation().getColumn(0); // Base B in bodyB space,
										//note that the base of the two bodies are not aligned in this constraint.

		// Initialize constraint bases in world space.
		baseA[0] = pinionARotationAxisInA;
		baseB[0] = rackBShiftAxisInB;
		hkVector4Util::calculatePerpendicularVector( baseA[0], baseA[1] );
		hkVector4Util::calculatePerpendicularVector( baseB[0], baseB[1] );
		baseA[1].normalize<3>();
		baseB[1].normalize<3>();
		baseA[2].setCross(baseA[0], baseA[1]);
		baseB[2].setCross(baseB[0], baseB[1]);
	}

	// Set pivot points
	m_atoms.m_transforms.m_transformA.setTranslation(pinionARotationPivotInA);
	m_atoms.m_transforms.m_transformB.setTranslation(pinionARotationPivotInB);

	m_atoms.m_rackAndPinion.m_pinionRadiusOrScrewPitch = pinionRadiusOrScrewPitch;
	m_atoms.m_rackAndPinion.m_isScrew = (type == TYPE_SCREW);

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of rack-and-pinion constraint inconsistent.");
}

hkBool hkpRackAndPinionConstraintData::isValid() const
{
	return m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal()
		&& m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal()
		&& !hkMath::equal(hkReal(0), m_atoms.m_rackAndPinion.m_pinionRadiusOrScrewPitch, HK_REAL_EPSILON);
}

int hkpRackAndPinionConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_RACK_AND_PINION;
}

void hkpRackAndPinionConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpRackAndPinionConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
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
