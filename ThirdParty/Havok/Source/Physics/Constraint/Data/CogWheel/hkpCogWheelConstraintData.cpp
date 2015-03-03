/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/CogWheel/hkpCogWheelConstraintData.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpCogWheelConstraintData::hkpCogWheelConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_cogWheels.m_cogWheelRadiusA = 1.0f;
	m_atoms.m_cogWheels.m_cogWheelRadiusB = 1.0f;
	m_atoms.m_cogWheels.m_isScrew = false;
	m_atoms.m_cogWheels.m_memOffsetToInitialAngleOffset = HK_OFFSET_OF(Runtime, m_initialAngleOffset);
	m_atoms.m_cogWheels.m_memOffsetToPrevAngle = HK_OFFSET_OF(Runtime, m_prevAngle);
	m_atoms.m_cogWheels.m_memOffsetToRevolutionCounter = HK_OFFSET_OF(Runtime, m_revolutionCounter);
}


void hkpCogWheelConstraintData::setInWorldSpace(const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
										const hkVector4& rotationPivotA, const hkVector4& rotationAxisA, hkReal radiusA,
										const hkVector4& rotationPivotB, const hkVector4& rotationAxisB, hkReal radiusB)
{
	HK_ASSERT2(0xad234232,    hkMath::equal(rotationAxisA.lengthSquared<3>().getReal(), 1.0f)
		&& hkMath::equal(rotationAxisB.lengthSquared<3>().getReal(), 1.0f),
		"Input axes must be normalized.");

	// Set bases: baseA.getColumn(0) and baseB.getColumn(0) are axes of rotation for each body.
	// The other vectors of the base are used to determine the angles of relative orientation.
	{
		hkVector4 baseA[3]; // Base A in world.
		hkVector4 baseB[3]; // Base B in world, note that the base of the two bodies are not aligned in this constraint.

		// Initialize constraint bases in world space.
		baseA[0] = rotationAxisA;
		baseB[0] = rotationAxisB;
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
	m_atoms.m_transforms.m_transformA.getTranslation()._setTransformedInversePos( bodyATransform, rotationPivotA );
	m_atoms.m_transforms.m_transformB.getTranslation()._setTransformedInversePos( bodyBTransform, rotationPivotB );

	m_atoms.m_cogWheels.m_cogWheelRadiusA = radiusA;
	m_atoms.m_cogWheels.m_cogWheelRadiusB = radiusB;

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of rack-and-pinion constraint inconsistent.");
}

void hkpCogWheelConstraintData::setInBodySpace(
							const hkVector4& rotationPivotAInA, const hkVector4& rotationAxisAInA, hkReal radiusA,
							const hkVector4& rotationPivotBInB, const hkVector4& rotationAxisBInB, hkReal radiusB)
{
	HK_ASSERT2(0xad234233,    hkMath::equal(rotationAxisAInA.lengthSquared<3>().getReal(), 1.0f)
		&& hkMath::equal(rotationAxisBInB.lengthSquared<3>().getReal(), 1.0f),
		"Input axes must be normalized.");

	// Set bases: baseA.getColumn(0) and baseB.getColumn(0) are axes of rotation for each body.
	// The other vectors of the base are used to determine the angles of relative orientation.
	{
		hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getRotation().getColumn(0); // Base A in body A space.
		hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getRotation().getColumn(0); // Base B in body B space,
										  // note that the base of the two bodies are not aligned in this constraint.

		// Initialize constraint bases in world space.
		baseA[0] = rotationAxisAInA;
		baseB[0] = rotationAxisBInB;
		hkVector4Util::calculatePerpendicularVector( baseA[0], baseA[1] );
		hkVector4Util::calculatePerpendicularVector( baseB[0], baseB[1] );
		baseA[1].normalize<3>();
		baseB[1].normalize<3>();
		baseA[2].setCross(baseA[0], baseA[1]);
		baseB[2].setCross(baseB[0], baseB[1]);
	}

	// Set pivot points
	m_atoms.m_transforms.m_transformA.setTranslation(rotationPivotAInA);
	m_atoms.m_transforms.m_transformB.setTranslation(rotationPivotBInB);

	m_atoms.m_cogWheels.m_cogWheelRadiusA = radiusA;
	m_atoms.m_cogWheels.m_cogWheelRadiusB = radiusB;

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of cog wheel constraint inconsistent.");
}


hkBool hkpCogWheelConstraintData::isValid() const
{
	return m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal()
		&& m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal()
		&& !hkMath::equal(hkReal(0), m_atoms.m_cogWheels.m_cogWheelRadiusA, HK_REAL_EPSILON)
		&& !hkMath::equal(hkReal(0), m_atoms.m_cogWheels.m_cogWheelRadiusB, HK_REAL_EPSILON);
}

int hkpCogWheelConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_COG_WHEEL;
}

void hkpCogWheelConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpCogWheelConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
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
