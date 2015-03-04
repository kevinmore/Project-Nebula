/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Motion/hkpMotion.h>
#include <Physics/ConstraintSolver/Solve/hkpSolverInfo.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>


hkpMotion::hkpMotion() : hkReferencedObject()
{
	hkVector4 position;
	position.setZero();
	hkQuaternion rotation;
	rotation.setIdentity();
	init(position, rotation, false);
}

hkpMotion::hkpMotion(const hkVector4& position, const hkQuaternion& rotation, bool wantDeactivation)
{
	init(position, rotation, wantDeactivation);
}

HK_FORCE_INLINE void hkpMotion::init(const hkVector4& position, const hkQuaternion& rotation, bool wantDeactivation)
{
	m_linearVelocity.setZero();
	m_angularVelocity.setZero();

	hkCheckDeterminismUtil::checkMt(0xf0000050, position);
	hkCheckDeterminismUtil::checkMt(0xf0000051, rotation);

	m_motionState.initMotionState( position, rotation );

	m_motionState.m_linearDamping.setZero();
	m_motionState.m_angularDamping.setZero();
	m_type = MOTION_INVALID;

	// deactivation data
	{
		if ( wantDeactivation)
		{
			m_deactivationIntegrateCounter = hkInt8(0xf & int(position(0)));
		}
		else
		{
			m_deactivationIntegrateCounter = 0xff;
		}
		m_deactivationNumInactiveFrames[0] = 0;
		m_deactivationNumInactiveFrames[1] = 0;
		m_deactivationRefPosition[0].setZero();
		m_deactivationRefPosition[1].setZero();
		m_deactivationRefOrientation[0] = 0;
		m_deactivationRefOrientation[1] = 0;
	}

	// Gravity factor - leave gravity as it is by default
	m_gravityFactor.setOne();
}

// Set the mass of the rigid body.
void hkpMotion::setMass(hkReal mass)
{
	hkReal massInv;
	if (mass == 0.0f)
	{
		massInv = 0.0f;
	}
	else
	{
		massInv = 1.0f / mass;
	}
	setMassInv( massInv );
}
void hkpMotion::setMass(hkSimdRealParameter mass)
{
	hkSimdReal massInv; massInv.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(mass);
	setMassInv( massInv );
}

	// Get the mass of the rigid body.
hkReal hkpMotion::getMass() const
{
	const hkSimdReal massInv = getMassInv();
	hkSimdReal invM; invM.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(massInv);
	return invM.getReal();
}

// Set the mass of the rigid body.
void hkpMotion::setMassInv(hkReal massInv)
{
	m_inertiaAndMassInv(3) = massInv;
}
void hkpMotion::setMassInv(hkSimdRealParameter massInv)
{
	m_inertiaAndMassInv.setW(massInv);
}

// Explicit center of mass in local space.
void hkpMotion::setCenterOfMassInLocal(const hkVector4& centerOfMass)
{	
	hkSweptTransformUtil::setCentreOfRotationLocal( centerOfMass, m_motionState );
}


void hkpMotion::setPosition(const hkVector4& position)
{
	hkSweptTransformUtil::warpToPosition( position, m_motionState );
}

void hkpMotion::setRotation(const hkQuaternion& rotation)
{
	hkSweptTransformUtil::warpToRotation( rotation, m_motionState);
}

void hkpMotion::setPositionAndRotation(const hkVector4& position, const hkQuaternion& rotation)
{
	hkSweptTransformUtil::warpTo( position, rotation, m_motionState );
}

void hkpMotion::setTransform(const hkTransform& transform)
{
	hkSweptTransformUtil::warpTo( transform, m_motionState );
}

void hkpMotion::approxTransformAt( hkTime time, hkTransform& transformOut )
{
	getMotionState()->getSweptTransform().approxTransformAt( time, transformOut );
}

void hkpMotion::setLinearVelocity(const hkVector4& newVel)
{
	HK_ASSERT2(0xf093fe57, newVel.isOk<3>(), "Invalid Linear Velocity");
	m_linearVelocity = newVel;
}

void hkpMotion::setAngularVelocity(const hkVector4& newVel)
{
	HK_ASSERT2(0xf093fe56, newVel.isOk<3>(), "Invalid Angular Velocity");
	m_angularVelocity = newVel;
}

void hkpMotion::applyLinearImpulse(const hkVector4& imp)
{
	// PSEUDOCODE IS m_linearVelocity += m_massInv * imp;
	m_linearVelocity.addMul( getMassInv(), imp);
}

void hkpMotion::getMotionStateAndVelocitiesAndDeactivationType(hkpMotion* motionOut)
{
	motionOut->m_motionState = m_motionState;
	motionOut->m_linearVelocity = m_linearVelocity;	// Copy over linear velocity
	motionOut->m_angularVelocity = m_angularVelocity;	// Copy over angular velocity
	motionOut->m_deactivationIntegrateCounter = m_deactivationIntegrateCounter;
}

void hkpMotion::setDeactivationClass(int deactivationClass)
{
	HK_ASSERT2( 0xf0230234, deactivationClass > 0 && deactivationClass < hkpSolverInfo::DEACTIVATION_CLASSES_END, "Your deactivation class is out of range");
	m_motionState.m_deactivationClass = hkUint8(deactivationClass);
}

void hkpMotion::requestDeactivation()
{
	// See hkRigidMotionUtilCheckDeactivation(), hkRigidMotionUtilCanDeactivateFinal() for details

	m_deactivationRefPosition[0] = getPosition();
	m_deactivationRefPosition[0].setW(hkSimdReal_Max);	// speed
	m_deactivationRefPosition[1] = m_deactivationRefPosition[0];

	m_deactivationRefOrientation[0] = hkVector4Util::packQuaternionIntoInt32( getRotation().m_vec );
	m_deactivationRefOrientation[1] = m_deactivationRefOrientation[0];

	(m_deactivationNumInactiveFrames[0] &= ~0x7f) |= NUM_INACTIVE_FRAMES_TO_DEACTIVATE+1;
	(m_deactivationNumInactiveFrames[1] &= ~0x7f) |= NUM_INACTIVE_FRAMES_TO_DEACTIVATE+1;
}


//HK_COMPILE_TIME_ASSERT( sizeof(hkpMotion) == 0xd0 );

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
