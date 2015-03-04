/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpRagdollConstraintData::hkpRagdollConstraintData()
{

	m_atoms.m_transforms.m_transformA.setIdentity();
	m_atoms.m_transforms.m_transformB.setIdentity();

	m_atoms.m_angFriction.m_firstFrictionAxis = 0;
	m_atoms.m_angFriction.m_numFrictionAxes = 3;
	m_atoms.m_angFriction.m_maxFrictionTorque = hkReal(0);

	// this is the unpowered version of the constraint
	m_atoms.m_ragdollMotors.m_isEnabled = false;

	// Set motor offsets
	{
		// the initialized variable is placed after all the solver result
		m_atoms.m_ragdollMotors.m_initializedOffset = HK_OFFSET_OF( Runtime, m_initialized );

		// the previous target angle is stored right after it
		m_atoms.m_ragdollMotors.m_previousTargetAnglesOffset = HK_OFFSET_OF( Runtime, m_previousTargetAngles );
	}

	m_atoms.m_ragdollMotors.m_target_bRca.setIdentity();
	m_atoms.m_ragdollMotors.m_motors[0] = HK_NULL;
	m_atoms.m_ragdollMotors.m_motors[1] = HK_NULL;
	m_atoms.m_ragdollMotors.m_motors[2] = HK_NULL;


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

hkpRagdollConstraintData::Atoms::Atoms(hkFinishLoadedObjectFlag f)
:	m_transforms(f)
,	m_setupStabilization(f)
,	m_ragdollMotors(f)
,	m_angFriction(f)
,	m_twistLimit(f)
,	m_coneLimit(f)
,	m_planesLimit(f)
,	m_ballSocket(f)
{
	if( f.m_finishing )
	{
		// Set motor offsets
		m_ragdollMotors.m_initializedOffset				= HK_OFFSET_OF( Runtime, m_initialized );
		m_ragdollMotors.m_previousTargetAnglesOffset	= HK_OFFSET_OF( Runtime, m_previousTargetAngles );
	}
}

void hkpRagdollConstraintData::setInWorldSpace(const hkTransform& bodyATransform, const hkTransform& bodyBTransform,
										  const hkVector4& pivot, const hkVector4& twistAxisW,
										  const hkVector4& planeAxisW)
{
	HK_ASSERT2(0x77ad7e93, hkMath::equal( twistAxisW.dot<3>(planeAxisW).getReal(), 0.0f), "twistAxisW && planeAxisW should be perpendicular");

		// Set relative orientation
	{
		hkVector4 constraintBaseInW[3];

		constraintBaseInW[Atoms::AXIS_TWIST] = twistAxisW; constraintBaseInW[Atoms::AXIS_TWIST].normalize<3>();
		constraintBaseInW[Atoms::AXIS_PLANES] = planeAxisW; constraintBaseInW[Atoms::AXIS_PLANES].normalize<3>();
		constraintBaseInW[Atoms::AXIS_CROSS_PRODUCT].setCross( constraintBaseInW[Atoms::AXIS_TWIST], constraintBaseInW[Atoms::AXIS_PLANES] );

		hkVector4Util::rotateInversePoints( bodyATransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_transforms.m_transformA.getRotation().getColumn(0) );
		hkVector4Util::rotateInversePoints( bodyBTransform.getRotation(), constraintBaseInW, 3, &m_atoms.m_transforms.m_transformB.getRotation().getColumn(0) );
	}

		// Set pivot points
	m_atoms.m_transforms.m_transformA.getTranslation().setTransformedInversePos( bodyATransform, pivot );
	m_atoms.m_transforms.m_transformB.getTranslation().setTransformedInversePos( bodyBTransform, pivot );

	m_atoms.m_ragdollMotors.m_target_bRca = m_atoms.m_transforms.m_transformB.getRotation();

	HK_ASSERT2(0xadea65ee, isValid(), "Members of ragdoll constraint inconsistent.");
}


void hkpRagdollConstraintData::setInBodySpace( const hkVector4& pivotA,const hkVector4& pivotB,
										const hkVector4& planeAxisA,const hkVector4& planeAxisB,
										const hkVector4& twistAxisA, const hkVector4& twistAxisB)
{
	hkVector4* baseA = &m_atoms.m_transforms.m_transformA.getColumn(0);
	baseA[Atoms::AXIS_TWIST] = twistAxisA; baseA[Atoms::AXIS_TWIST].normalize<3>();
	baseA[Atoms::AXIS_PLANES] = planeAxisA; baseA[Atoms::AXIS_PLANES].normalize<3>();
	baseA[Atoms::AXIS_CROSS_PRODUCT].setCross( baseA[Atoms::AXIS_TWIST], baseA[Atoms::AXIS_PLANES] );
	m_atoms.m_transforms.m_transformA.getTranslation() = pivotA;

	hkVector4* baseB = &m_atoms.m_transforms.m_transformB.getColumn(0);
	baseB[Atoms::AXIS_TWIST] = twistAxisB; baseB[Atoms::AXIS_TWIST].normalize<3>();
	baseB[Atoms::AXIS_PLANES] = planeAxisB; baseB[Atoms::AXIS_PLANES].normalize<3>();
	baseB[Atoms::AXIS_CROSS_PRODUCT].setCross( baseB[Atoms::AXIS_TWIST], baseB[Atoms::AXIS_PLANES] );
	m_atoms.m_transforms.m_transformB.getTranslation() = pivotB;

	m_atoms.m_ragdollMotors.m_target_bRca = m_atoms.m_transforms.m_transformB.getRotation();

	HK_ASSERT2(0xadea65ef, isValid(), "Members of ragdoll constraint inconsistent.");
}

void hkpRagdollConstraintData::setBreachImpulse(hkReal breachImpulse)
{
	m_atoms.m_ballSocket.setBreachImpulse(breachImpulse);
}

hkReal hkpRagdollConstraintData::getBreachImpulse() const
{
	return m_atoms.m_ballSocket.getBreachImpulse();
}

void hkpRagdollConstraintData::setBodyToNotify(int bodyIdx)
{
	//HK_ASSERT2(0xad808071, notifyBodyA >= 0 && notifyBodyA <= 1 && notifyBodyB >= 0 && notifyBodyB <= 1, "Notify parameters must be 0 or 1.");
	//m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 * notifyBodyA + 0x02 * notifyBodyB);
	HK_ASSERT2(0xad808071, bodyIdx >= 0 && bodyIdx <= 1, "Notify body index must be 0 or 1.");
	m_atoms.m_ballSocket.m_bodiesToNotify = hkUint8(0x01 << bodyIdx);
}

hkUint8 hkpRagdollConstraintData::getNotifiedBodyIndex() const
{
	//return m_atoms.m_ballSocket.m_bodiesToNotify;
	HK_ASSERT2(0xad809032, m_atoms.m_ballSocket.m_bodiesToNotify & 0x3, "Body to be notified not set.");
	return m_atoms.m_ballSocket.m_bodiesToNotify >> 1;
}

void hkpRagdollConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& infoOut ) const
{
	getConstraintInfoUtil( m_atoms.getAtoms(), m_atoms.getSizeOfAllAtoms(), infoOut );
}

void hkpRagdollConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const
{
	// we need runtime data to be able to support lastAngle and friction
	infoOut.m_numSolverResults = SOLVER_RESULT_MAX;
	infoOut.m_sizeOfExternalRuntime = sizeof( Runtime );
}



void hkpRagdollConstraintData::setMaxFrictionTorque(hkReal tmag)
{
	m_atoms.m_angFriction.m_maxFrictionTorque = tmag;
}

void hkpRagdollConstraintData::setConeLimitStabilization(hkBool enable)
{
	const int memOffset = HK_OFFSET_OF(Runtime, m_coneAngleOffset) - HK_OFFSET_OF( Runtime, m_solverResults[SOLVER_RESULT_CONE] );
	HK_ASSERT2(0XAD8755AA, memOffset == (memOffset & 0xff), "Offset doesn't fit into hkUint8.");
	m_atoms.m_coneLimit.m_memOffsetToAngleOffset = hkUint8( int(enable) * memOffset);
}

void hkpRagdollConstraintData::setAsymmetricConeAngle( hkReal cone_min, hkReal cone_max )
{
	hkReal center_cone = (cone_max + cone_min) * hkReal(0.5f);
	hkReal diff_cone   = (cone_max - cone_min) * hkReal(0.5f);

	hkTransform& baseB = m_atoms.m_transforms.m_transformB;
	hkQuaternion baseModificationB; baseModificationB.setAxisAngle( baseB.getColumn<Atoms::AXIS_PLANES>(), -center_cone );
	baseB.getColumn(Atoms::AXIS_TWIST).setRotatedDir( baseModificationB, baseB.getColumn<Atoms::AXIS_TWIST>() );
	baseB.getColumn(Atoms::AXIS_CROSS_PRODUCT).setCross( baseB.getColumn<Atoms::AXIS_TWIST>(), baseB.getColumn<Atoms::AXIS_PLANES>() );

	HK_ASSERT2(0xad67ddaa, baseB.getRotation().isOrthonormal(), "Base B is not orthonormal");

	setConeAngularLimit( diff_cone );
}


hkBool hkpRagdollConstraintData::isValid() const
{
	// In stable mode, we need the setupStabilization atom enabled!
	if ( (m_atoms.m_ballSocket.m_solvingMethod == hkpConstraintAtom::METHOD_STABILIZED) &&
		!m_atoms.m_setupStabilization.m_enabled )
	{
		return false;
	}

	hkBool valid = true;
	valid = valid && m_atoms.m_transforms.m_transformA.getRotation().isOrthonormal();
	valid = valid && m_atoms.m_transforms.m_transformB.getRotation().isOrthonormal();
	valid = valid && ( m_atoms.m_coneLimit.m_minAngle == -100.0f );
	valid = valid && ( m_atoms.m_coneLimit.m_maxAngle >= 0.0f );
	valid = valid && ( m_atoms.m_coneLimit.m_maxAngle <= HK_REAL_PI );
	valid = valid && ( m_atoms.m_planesLimit.m_minAngle <= m_atoms.m_planesLimit.m_maxAngle );
	valid = valid && ( m_atoms.m_twistLimit.m_minAngle <= m_atoms.m_twistLimit.m_maxAngle );

	// Check ranges on other angular limits ??
	// No wacky behavior seems to result

	return valid;
}


void hkpRagdollConstraintData::getConstraintFrameA( hkMatrix3& constraintFrameA ) const
{
	constraintFrameA = m_atoms.m_transforms.m_transformA.getRotation();
}


void hkpRagdollConstraintData::getConstraintFrameB( hkMatrix3& constraintFrameB ) const
{
	constraintFrameB = m_atoms.m_transforms.m_transformB.getRotation();
}



//////////////////////////////////////////////////////////////////////////
//
//  Motor - related methods
//
//////////////////////////////////////////////////////////////////////////


hkpRagdollConstraintData::~hkpRagdollConstraintData()
{
	for (int m = 0; m < 3; m++)
	{
		if( m_atoms.m_ragdollMotors.m_motors[m] != HK_NULL)
		{
			m_atoms.m_ragdollMotors.m_motors[m]->removeReference();
		}
	}
}

void hkpRagdollConstraintData::setTarget( const hkMatrix3& target_cbRca )
{
	// we assign :
	//  m_target_bRca = (bRcb * cbRca) * (caRa * aRca);
	// where :
	//  bRcb -- is rotation from bodyB's constrainst space to bodyB's space // it's stored in m_atoms.m_transforms.m_transformA.getRotation()
	//  aRca -- is rotation from bodyA's constrainst space to bodyA's space // it's stored in m_atoms.m_transforms.m_transformA.getRotation()
	//  caRa == aRca^-1
	//  cbRcb -- is the target relative orientation of constraint refernce frames attached to bodies A & B
	//

	m_atoms.m_ragdollMotors.m_target_bRca.setMul( m_atoms.m_transforms.m_transformB.getRotation(), target_cbRca );
}


void hkpRagdollConstraintData::setTargetRelativeOrientationOfBodies( const hkRotation& target_bRa )
{
	// we assign :
	//  m_target_bRca = (bRcb * cbRca * caRa) * aRca;
	// for explanation of the indiviual rotation variables see info in setTarget().

	HK_ASSERT2(0xad67dd88, hkpRagdollConstraintData::Atoms::AXIS_TWIST == 0, "Assuming twist axis has index 0 in base");
	m_atoms.m_ragdollMotors.m_target_bRca.setMul( target_bRa, m_atoms.m_transforms.m_transformA.getRotation() );
}


void hkpRagdollConstraintData::getTarget(hkMatrix3& target_cbRca )
{
	target_cbRca.setMulInverseMul(m_atoms.m_transforms.m_transformB.getRotation(), m_atoms.m_ragdollMotors.m_target_bRca);
}


hkpConstraintMotor* hkpRagdollConstraintData::getTwistMotor() const
{
	return m_atoms.m_ragdollMotors.m_motors[MOTOR_TWIST];
}

void hkpRagdollConstraintData::setMotor( MotorIndex index, hkpConstraintMotor* newMotor)
{
	if( newMotor )
	{
		newMotor->addReference();
	}

	if( m_atoms.m_ragdollMotors.m_motors[index] )
	{
		m_atoms.m_ragdollMotors.m_motors[index]->removeReference();
	}

	m_atoms.m_ragdollMotors.m_motors[index] = newMotor;

}

void hkpRagdollConstraintData::setTwistMotor( hkpConstraintMotor* motor )
{
	setMotor(MOTOR_TWIST, motor);
}


hkpConstraintMotor* hkpRagdollConstraintData::getConeMotor() const
{
	return m_atoms.m_ragdollMotors.m_motors[MOTOR_CONE];
}


void hkpRagdollConstraintData::setConeMotor( hkpConstraintMotor* motor )
{
	setMotor(MOTOR_CONE, motor);
}


hkpConstraintMotor* hkpRagdollConstraintData::getPlaneMotor() const
{
	return m_atoms.m_ragdollMotors.m_motors[MOTOR_PLANE];
}

void hkpRagdollConstraintData::setPlaneMotor( hkpConstraintMotor* motor )
{
	setMotor(MOTOR_PLANE, motor);
}

void hkpRagdollConstraintData::setMotorsEnabled( hkpConstraintRuntime* runtimeIn, hkBool areEnabled )
{
	m_atoms.m_ragdollMotors.m_isEnabled = areEnabled;
	m_atoms.m_angFriction.m_isEnabled = !areEnabled;

	Runtime* runtime = getRuntime( runtimeIn );
	if ( runtime )
	{
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_0].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_0_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_1].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_1_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_2].init();
		runtime->m_solverResults[SOLVER_RESULT_FRICTION_2_INTERNAL].init();

		runtime->m_solverResults[SOLVER_RESULT_MOTOR_0].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_0_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_1].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_1_INTERNAL].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_2].init();
		runtime->m_solverResults[SOLVER_RESULT_MOTOR_2_INTERNAL].init();
	}
}

int hkpRagdollConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL;
}

//
//	Sets the solving method for this constraint. Use one of the hkpConstraintAtom::SolvingMethod as a value for method.

void hkpRagdollConstraintData::setSolvingMethod(hkpConstraintAtom::SolvingMethod method)
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

hkResult hkpRagdollConstraintData::getInertiaStabilizationFactor(hkReal& inertiaStabilizationFactorOut) const
{
	inertiaStabilizationFactorOut = m_atoms.m_ballSocket.getInertiaStabilizationFactor();
	return HK_SUCCESS;
}

//
//	Sets the inertia stabilization factor, return HK_FAILURE if the factor is not defined for the given constraint.

hkResult hkpRagdollConstraintData::setInertiaStabilizationFactor(const hkReal inertiaStabilizationFactorIn)
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
