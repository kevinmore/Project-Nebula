/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpLimitedHingeConstraintData::hkpLimitedHingeConstraintData()
{
	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_2dAng.m_freeRotationAxis = 0;

	m_atoms.m_angLimit.m_limitAxis = 0;
	m_atoms.m_angLimit.m_minAngle = -HK_REAL_PI;
	m_atoms.m_angLimit.m_maxAngle =  HK_REAL_PI;
	m_atoms.m_angLimit.m_angularLimitsTauFactor = hkReal(1);

	m_atoms.m_angFriction.m_firstFrictionAxis = 0;
	m_atoms.m_angFriction.m_numFrictionAxes = 1;
	m_atoms.m_angFriction.m_maxFrictionTorque = hkReal(0);

	// Set motor offsets
	{
		HK_ASSERT2(0xad873422, SOLVER_RESULT_MOTOR == 0, "Motor's runtime assumed to be at zero-offset.");

		// the initialized variable is placed after all the solver result
		m_atoms.m_angMotor.m_initializedOffset = HK_OFFSET_OF( Runtime, m_initialized );

		// the previous target angle is stored right after it
		m_atoms.m_angMotor.m_previousTargetAngleOffset = HK_OFFSET_OF( Runtime, m_previousTargetAngle );

		m_atoms.m_angMotor.m_correspondingAngLimitSolverResultOffset = SOLVER_RESULT_LIMIT * sizeof(hkpSolverResults);
	}

	m_atoms.m_angMotor.m_isEnabled = false;
	m_atoms.m_angMotor.m_targetAngle = hkReal(0);
	m_atoms.m_angMotor.m_motor = HK_NULL;
	m_atoms.m_angMotor.m_motorAxis = 0;
}

hkpLimitedHingeConstraintData::Atoms::Atoms(hkFinishLoadedObjectFlag f)
:	m_transforms(f)
,	m_setupStabilization(f)
,	m_angMotor(f)
,	m_angFriction(f)
,	m_angLimit(f)
,	m_2dAng(f)
,	m_ballSocket(f)
{
	if( f.m_finishing )
	{
		// Set motor offsets
		m_angMotor.m_initializedOffset							= HK_OFFSET_OF( hkpLimitedHingeConstraintData::Runtime, m_initialized );
		m_angMotor.m_previousTargetAngleOffset					= HK_OFFSET_OF( hkpLimitedHingeConstraintData::Runtime, m_previousTargetAngle );
		m_angMotor.m_correspondingAngLimitSolverResultOffset	= SOLVER_RESULT_LIMIT * sizeof(hkpSolverResults);
	}
}

hkpLimitedHingeConstraintData::~hkpLimitedHingeConstraintData()
{
	if( m_atoms.m_angMotor.m_motor )
	{
		m_atoms.m_angMotor.m_motor->removeReference();
	}
}


void hkpLimitedHingeConstraintData::setInWorldSpace(const hkTransform& bodyATransform,
													const hkTransform& bodyBTransform,
													const hkVector4& pivot,
													const hkVector4& axisIn)
{
	hkVector4 axis = axisIn; axis.normalize<3>();
	hkVector4 perpToAxle1;
	hkVector4 perpToAxle2;
	hkVector4Util::calculatePerpendicularVector( axis, perpToAxle1 ); perpToAxle1.normalize<3>();
	perpToAxle2.setCross(axis, perpToAxle1);

	m_atoms.m_transforms.m_transformA.getColumn(0).setRotatedInverseDir(bodyATransform.getRotation(), axis);
	m_atoms.m_transforms.m_transformA.getColumn(1).setRotatedInverseDir(bodyATransform.getRotation(), perpToAxle1);
	m_atoms.m_transforms.m_transformA.getColumn(2).setRotatedInverseDir(bodyATransform.getRotation(), perpToAxle2);
	m_atoms.m_transforms.m_transformA.getColumn(3).setTransformedInversePos(bodyATransform, pivot);

	m_atoms.m_transforms.m_transformB.getColumn(0).setRotatedInverseDir(bodyBTransform.getRotation(), axis);
	m_atoms.m_transforms.m_transformB.getColumn(1).setRotatedInverseDir(bodyBTransform.getRotation(), perpToAxle1);
	m_atoms.m_transforms.m_transformB.getColumn(2).setRotatedInverseDir(bodyBTransform.getRotation(), perpToAxle2);
	m_atoms.m_transforms.m_transformB.getColumn(3).setTransformedInversePos(bodyBTransform, pivot);

	HK_ASSERT2(0x3a0a5294, isValid(), "Members of LimitedHinge constraint inconsistent after World Space constructor..");
}


void hkpLimitedHingeConstraintData::setInBodySpace(const hkVector4& pivotA,
												   const hkVector4& pivotB,
												   const hkVector4& axisA,
												   const hkVector4& axisB,
												   const hkVector4& axisAPerp,
												   const hkVector4& axisBPerp)
{
	hkTransform& ta = m_atoms.m_transforms.m_transformA;
	ta.getColumn(0) = axisA;
	ta.getColumn(0).normalize<3>();
	ta.getColumn(1) = axisAPerp;
	ta.getColumn(1).normalize<3>();
	ta.getColumn(2).setCross(ta.getColumn<0>(), ta.getColumn<1>());
	ta.getColumn(3) = pivotA;

	hkTransform& tb = m_atoms.m_transforms.m_transformB;
	tb.getColumn(0) = axisB;
	tb.getColumn(0).normalize<3>();
	tb.getColumn(1) = axisBPerp;
	tb.getColumn(1).normalize<3>();
	tb.getColumn(2).setCross(tb.getColumn<0>(), tb.getColumn<1>());
	tb.getColumn(3) = pivotB;

	HK_ASSERT2(0x3a0a5394, isValid(), "Members of LimitedHinge constraint inconsistent after Body Space constructor..");
}

void hkpLimitedHingeConstraintData::setBreachImpulse(hkReal breachImpulse)
{
	m_atoms.m_ballSocket.setBreachImpulse(breachImpulse);
}

hkReal hkpLimitedHingeConstraintData::getBreachImpulse() const
{
	return m_atoms.m_ballSocket.getBreachImpulse();
}

void hkpLimitedHingeConstraintData::setBodyToNotify(int bodyIdx)
{
	//HK_ASSERT2(0xad808071, notifyBodyA >= 0 && notifyBodyA <= 1 && notifyBodyB >= 0 && notifyBodyB <= 1, "Notify parameters must be 0 or 1.");
	//m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 * notifyBodyA + 0x02 * notifyBodyB);
	HK_ASSERT2(0xad808071, bodyIdx >= 0 && bodyIdx <= 1, "Notify body index must be 0 or 1.");
	m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 << bodyIdx);
}

hkUint8 hkpLimitedHingeConstraintData::getNotifiedBodyIndex() const
{
	//return m_atoms.m_ballSocket.m_bodiesToNotify;
	HK_ASSERT2(0xad809032, m_atoms.m_ballSocket.m_bodiesToNotify & 0x3, "Body to be notified not set.");
	return m_atoms.m_ballSocket.m_bodiesToNotify >> 1;
}

void hkpLimitedHingeConstraintData::setMotorEnabled( hkpConstraintRuntime* runtimeIn, hkBool isEnabled )
{
	m_atoms.m_angMotor.m_isEnabled = isEnabled;
	m_atoms.m_angFriction.m_isEnabled = !isEnabled;

	Runtime* runtime = getRuntime( runtimeIn );
	if ( runtime )
	{
		runtime->m_solverResults[SOLVER_RESULT_MOTOR].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_INTERNAL].init();
	}
}

void hkpLimitedHingeConstraintData::setMotor( hkpConstraintMotor* motor )
{
	if( motor )
	{
		motor->addReference();
	}

	if( m_atoms.m_angMotor.m_motor )
	{
		m_atoms.m_angMotor.m_motor->removeReference();
	}
	m_atoms.m_angMotor.m_motor = motor;
}

void hkpLimitedHingeConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpLimitedHingeConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// we need runtime data to be able to support lastAngle and friction
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
}


hkBool hkpLimitedHingeConstraintData::isValid() const
{
	// In stable mode, we need the setupStabilization atom enabled!
	if ( (m_atoms.m_ballSocket.m_solvingMethod == hkpConstraintAtom::METHOD_STABILIZED) &&
		!m_atoms.m_setupStabilization.m_enabled )
	{
		return false;
	}

	return		m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal() &&
				m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal()
		// limits solverResults are used as a reference for motors!
		&& (!m_atoms.m_angMotor.m_isEnabled || m_atoms.m_angLimit.m_isEnabled);
	// Hinge now allows for ranges > 2 * HK_REAL_PI
}


int hkpLimitedHingeConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE;
}

//
//	Sets the solving method for this constraint. Use one of the hkpConstraintAtom::SolvingMethod as a value for method.

void hkpLimitedHingeConstraintData::setSolvingMethod(hkpConstraintAtom::SolvingMethod method)
{
	switch ( method )
	{
	case hkpConstraintAtom::METHOD_STABILIZED:
		{
			m_atoms.m_setupStabilization.m_enabled	= true;
			m_atoms.m_ballSocket.m_solvingMethod	= hkpConstraintAtom::METHOD_STABILIZED;
		}
		break;

	case hkpConstraintAtom::METHOD_OLD:
		{
			m_atoms.m_setupStabilization.m_enabled	= false;
			m_atoms.m_ballSocket.m_solvingMethod	= hkpConstraintAtom::METHOD_OLD;
		}
		break;

	default:
		{
			HK_ASSERT2(0x3ce72932, false, "Unknown solving method! Please use one of the values in hkpConstraintAtom::SolvingMethod!");
		}
		break;
	}
}

//
//	Gets the inertia stabilization factor, returns HK_FAILURE if the factor is not defined for the given constraint.

hkResult hkpLimitedHingeConstraintData::getInertiaStabilizationFactor(hkReal& inertiaStabilizationFactorOut) const
{
	inertiaStabilizationFactorOut = m_atoms.m_ballSocket.getInertiaStabilizationFactor();
	return HK_SUCCESS;
}

//
//	Sets the inertia stabilization factor, return HK_FAILURE if the factor is not defined for the given constraint.

hkResult hkpLimitedHingeConstraintData::setInertiaStabilizationFactor(const hkReal inertiaStabilizationFactorIn)
{
	m_atoms.m_ballSocket.setInertiaStabilizationFactor(inertiaStabilizationFactorIn);
	return HK_SUCCESS;
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
