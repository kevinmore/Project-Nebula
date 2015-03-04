/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Clearance/hkpLinearClearanceConstraintData.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>


hkpLinearClearanceConstraintData::hkpLinearClearanceConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_linLimit0.m_min = 0;
	m_atoms.m_linLimit0.m_max =  0;
	m_atoms.m_linLimit0.m_axisIndex = 0;

	m_atoms.m_linLimit1.m_min = 0;
	m_atoms.m_linLimit1.m_max = 0;
	m_atoms.m_linLimit1.m_axisIndex = 1;

	m_atoms.m_linLimit2.m_min = 0;
	m_atoms.m_linLimit2.m_max = 0;
	m_atoms.m_linLimit2.m_axisIndex = 2;

	m_atoms.m_friction0.m_maxFrictionForce = hkReal(0);
	m_atoms.m_friction0.m_frictionAxis = 0;
	m_atoms.m_friction1.m_maxFrictionForce = hkReal(0);
	m_atoms.m_friction1.m_frictionAxis = 1;
	m_atoms.m_friction2.m_maxFrictionForce = hkReal(0);
	m_atoms.m_friction2.m_frictionAxis = 2;

	m_atoms.m_motor.m_motorAxis = 0;

	m_atoms.m_motor.m_isEnabled = false;
	m_atoms.m_motor.m_targetPosition = hkReal(0);
	m_atoms.m_motor.m_motor = HK_NULL;

	m_atoms.m_motor.m_motorAxis = hkpLinearClearanceConstraintData::Atoms::AXIS_SHAFT;

	// Set motor offsets
	{
		HK_ASSERT2(0xad873422, SOLVER_RESULT_MOTOR == 0, "Motor's runtime assumed to be at zero-offset.");
		m_atoms.m_motor.m_initializedOffset = HK_OFFSET_OF(Runtime, m_initialized);
		m_atoms.m_motor.m_previousTargetPositionOffset = HK_OFFSET_OF(Runtime, m_previousTargetPosition);
	}

	m_atoms.m_ang.m_firstConstrainedAxis = 0;
	m_atoms.m_ang.m_numConstrainedAxes   = 3;
}

hkpLinearClearanceConstraintData::Atoms::Atoms(hkFinishLoadedObjectFlag f)
:	m_transforms(f)
,	m_motor(f)
,	m_friction0(f)
,	m_friction1(f)
,	m_friction2(f)
,	m_ang(f)
,	m_linLimit0(f)
,	m_linLimit1(f)
,	m_linLimit2(f)
{
	if( f.m_finishing )
	{
		// Set motor offsets
		m_motor.m_initializedOffset				= HK_OFFSET_OF(Runtime, m_initialized);
		m_motor.m_previousTargetPositionOffset	= HK_OFFSET_OF(Runtime, m_previousTargetPosition);
	}
}

hkpLinearClearanceConstraintData::~hkpLinearClearanceConstraintData()
{
	if( m_atoms.m_motor.m_motor )
	{
		m_atoms.m_motor.m_motor->removeReference();
	}
}


void hkpLinearClearanceConstraintData::setInWorldSpace(hkpLinearClearanceConstraintData::Type type, const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
											 const hkVector4& pivot,const hkVector4& axis1, const hkVector4& axis2)
{
	hkVector4 l1A; l1A.setRotatedInverseDir( bodyATransform.getRotation(), axis1 ); l1A.normalize<3>();
	hkVector4 l2A; l2A.setRotatedInverseDir( bodyATransform.getRotation(), axis2 ); l2A.normalize<3>();
	hkVector4 lpA; lpA.setTransformedInversePos( bodyATransform, pivot );
	hkVector4 l1B; l1B.setRotatedInverseDir( bodyBTransform.getRotation(), axis1 ); l1B.normalize<3>();
	hkVector4 l2B; l2B.setRotatedInverseDir( bodyBTransform.getRotation(), axis2 ); l2B.normalize<3>();
	hkVector4 lpB; lpB.setTransformedInversePos( bodyBTransform, pivot );

	setInBodySpace(type, lpA, lpB, l1A, l1B, l2A, l2B);
}

void hkpLinearClearanceConstraintData::setInBodySpace(hkpLinearClearanceConstraintData::Type type, const hkVector4& pivotA, const hkVector4& pivotB,
											  const hkVector4& axis1A, const hkVector4& axis1B,
											  const hkVector4& axis2A, const hkVector4& axis2B)
{
	m_atoms.m_transforms.m_transformA.getTranslation() = pivotA;
	m_atoms.m_transforms.m_transformB.getTranslation() = pivotB;

	hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getColumn(0);
	baseA[0] = axis1A; baseA[0].normalize<3>();
	baseA[1] = axis2A; baseA[1].normalize<3>();
	baseA[2].setCross( baseA[0], baseA[1] );

	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[0] = axis1B; baseB[0].normalize<3>();
	baseB[1] = axis2B; baseB[1].normalize<3>();
	baseB[2].setCross( baseB[0], baseB[1] );

	hkUint8 ConstraintAxes[] = { 3, 2, 0 };
	m_atoms.m_ang.m_numConstrainedAxes = ConstraintAxes[type];

	HK_ASSERT2(0x4b2bf185, isValid(), "Members of linear slack constraint inconsistent.");
}

void hkpLinearClearanceConstraintData::setMotor( hkpConstraintMotor* motor )
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

void hkpLinearClearanceConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpLinearClearanceConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// always request a runtime
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
}


hkBool hkpLinearClearanceConstraintData::isValid() const
{
	return m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal() && m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal()
		&& m_atoms.m_linLimit0.m_min <= m_atoms.m_linLimit0.m_max
		&& m_atoms.m_linLimit1.m_min <= m_atoms.m_linLimit1.m_max
		&& m_atoms.m_linLimit2.m_min <= m_atoms.m_linLimit2.m_max
		&& m_atoms.m_ang.m_numConstrainedAxes <= 3;
}

int hkpLinearClearanceConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_LINEAR_SLACK;
}

void hkpLinearClearanceConstraintData::setMotorEnabled( hkpConstraintRuntime* runtimeIn, hkBool isEnabled )
{
	m_atoms.m_motor.m_isEnabled = isEnabled;
	m_atoms.m_friction0.m_isEnabled = !isEnabled;

	Runtime* runtime = getRuntime( runtimeIn );
	if (runtime)
	{
		runtime->m_solverResults[SOLVER_RESULT_MOTOR].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_0].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_0_INTERNAL].init();
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
