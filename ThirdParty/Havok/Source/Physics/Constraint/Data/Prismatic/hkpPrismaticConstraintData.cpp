/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpPrismaticConstraintData::hkpPrismaticConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_linLimit.m_min = -HK_REAL_MAX;
	m_atoms.m_linLimit.m_max =  HK_REAL_MAX;
	m_atoms.m_linLimit.m_axisIndex = 0;

	m_atoms.m_friction.m_maxFrictionForce = hkReal(0);
	m_atoms.m_friction.m_frictionAxis = 0;
	m_atoms.m_motor.m_motorAxis = 0;

	m_atoms.m_motor.m_isEnabled = false;
	m_atoms.m_motor.m_targetPosition = hkReal(0);
	m_atoms.m_motor.m_motor = HK_NULL;

	m_atoms.m_motor.m_motorAxis = hkpPrismaticConstraintData::Atoms::AXIS_SHAFT;

	// Set motor offsets
	{
		HK_ASSERT2(0xad873422, SOLVER_RESULT_MOTOR == 0, "Motor's runtime assumed to be at zero-offset.");
		m_atoms.m_motor.m_initializedOffset = HK_OFFSET_OF(Runtime, m_initialized);
		m_atoms.m_motor.m_previousTargetPositionOffset = HK_OFFSET_OF(Runtime, m_previousTargetPosition);
	}

	m_atoms.m_ang.m_firstConstrainedAxis = 0;
	m_atoms.m_ang.m_numConstrainedAxes   = 3;
	m_atoms.m_lin0.m_axisIndex = 1;
	m_atoms.m_lin1.m_axisIndex = 2;
}

hkpPrismaticConstraintData::Atoms::Atoms(hkFinishLoadedObjectFlag f)
:	m_transforms(f)
,	m_motor(f)
,	m_friction(f)
,	m_ang(f)
,	m_lin0(f)
,	m_lin1(f)
,	m_linLimit(f)
{
	if( f.m_finishing )
	{
		// Set motor offsets
		m_motor.m_initializedOffset				= HK_OFFSET_OF(Runtime, m_initialized);
		m_motor.m_previousTargetPositionOffset	= HK_OFFSET_OF(Runtime, m_previousTargetPosition);
	}
}

hkpPrismaticConstraintData::~hkpPrismaticConstraintData()
{
	if( m_atoms.m_motor.m_motor )
	{
		m_atoms.m_motor.m_motor->removeReference();
	}
}


void hkpPrismaticConstraintData::setInWorldSpace( const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
											 const hkVector4& pivot,const hkVector4& axis)
{
		// Set relative orientation
	{
		hkVector4 constraintBaseInW[3];

		constraintBaseInW[0] = axis; constraintBaseInW[0].normalize<3>();
		hkVector4Util::calculatePerpendicularVector( constraintBaseInW[0], constraintBaseInW[1] ); constraintBaseInW[1].normalize<3>();
		HK_ASSERT2(0xadbbc333, hkMath::equal( 1.0f, constraintBaseInW[1].lengthSquared<3>().getReal() ), "Vector perpendicular to axis is not normalized");
		constraintBaseInW[2].setCross( constraintBaseInW[0], constraintBaseInW[1] );

		hkVector4Util::rotateInversePoints( bodyATransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_transforms.m_transformA.getRotation().getColumn(0) );
		hkVector4Util::rotateInversePoints( bodyBTransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_transforms.m_transformB.getRotation().getColumn(0) );
	}

		// Set pivot points
	m_atoms.m_transforms.m_transformA.getTranslation().setTransformedInversePos( bodyATransform, pivot );
	m_atoms.m_transforms.m_transformB.getTranslation().setTransformedInversePos( bodyBTransform, pivot );

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of prismatic constraint inconsistent.");

}

void hkpPrismaticConstraintData::setInBodySpace( const hkVector4& pivotA, const hkVector4& pivotB,
											  const hkVector4& axisA, const hkVector4& axisB,
											  const hkVector4& axisAPerp, const hkVector4& axisBPerp)
{
	m_atoms.m_transforms.m_transformA.getTranslation() = pivotA;
	m_atoms.m_transforms.m_transformB.getTranslation() = pivotB;

	hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getColumn(0);
	baseA[0] = axisA; baseA[0].normalize<3>();
	baseA[1] = axisAPerp; baseA[1].normalize<3>();
	baseA[2].setCross( baseA[0], baseA[1] );

	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[0] = axisB; baseB[0].normalize<3>();
	baseB[1] = axisBPerp; baseB[1].normalize<3>();
	baseB[2].setCross( baseB[0], baseB[1] );

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of prismatic constraint inconsistent.");
}

void hkpPrismaticConstraintData::setMotor( hkpConstraintMotor* motor )
{
	if( motor )
	{
		motor->addReference();
	}

	if( m_atoms.m_motor.m_motor )
	{
		m_atoms.m_motor.m_motor->removeReference();
	}

	m_atoms.m_motor.m_motor = motor;
}

void hkpPrismaticConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpPrismaticConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// always request a runtime
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
}


hkBool hkpPrismaticConstraintData::isValid() const
{
	return m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal() && m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal() && m_atoms.m_linLimit.m_min <= m_atoms.m_linLimit.m_max;
}

int hkpPrismaticConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC;
}

void hkpPrismaticConstraintData::setMotorEnabled( hkpConstraintRuntime* runtimeIn, hkBool isEnabled )
{
	m_atoms.m_motor.m_isEnabled = isEnabled;
	m_atoms.m_friction.m_isEnabled = !isEnabled;

	Runtime* runtime = getRuntime( runtimeIn );
	if (runtime)
	{
		runtime->m_solverResults[SOLVER_RESULT_MOTOR].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_INTERNAL].init();
	}
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
