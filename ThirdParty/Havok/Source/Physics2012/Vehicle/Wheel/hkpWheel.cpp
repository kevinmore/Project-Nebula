/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Wheel/hkpWheel.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>
#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>
#include <Common/Base/Types/Physics/hkStepInfo.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Visualize/hkDebugDisplay.h>

void hkpWheel::init(hkpRigidBody* chassis,
					hkpWheelFrictionConstraintAtom::Axle* axle,
					hkReal radius, 
					hkReal width,
					hkVector4Parameter spinAxisCs, 
					hkVector4Parameter suspensionHardpointCs, 
					hkVector4Parameter suspensionDirectionCs, 
					hkReal suspensionLength)
{
	//
	// Copy basic parameters
	//

	m_chassis = chassis;
	m_axle = axle;
	m_radius = radius;
	m_width = width;
	m_suspensionHardpointCs = suspensionHardpointCs;
	m_suspensionDirectionCs = suspensionDirectionCs;
	m_suspensionLength = suspensionLength;
	m_spinAxisCs = spinAxisCs;

	// Initialize the friction constraint data
	m_frictionData.init( axle, radius );
	m_frictionConstraint = HK_NULL;

	//
	// Set default values for wheel parameters
	//

	m_friction = 3.0f;
	m_normalClippingAngleCos = 0.2f;
	m_suspensionStrength = 50.0f;
	m_suspensionDampingCompression = 3.0f;
	m_suspensionDampingRelaxation = 3.0f;
	m_slipDamping = 1.5f;

	m_torque = 0.0f;
	m_steeringAngle = 0.0f;

	//
	// Initialize data that will be updated by the wheel each step
	//

	hkVector4 up; up = hkVector4::getConstant<HK_QUADREAL_0100>();
	m_contactPoint.set( hkVector4::getZero(), up, 1);

	m_contactFriction = 0.0f;
	m_contactBody = HK_NULL;
	m_contactShapeKey[0] = hkpShapeKey(-1);

	m_hardPointWs.setZero();
	m_rayEndPointWs.setZero();
	m_currentSuspensionLength = 0.0f;
	m_suspensionDirectionWs.setZero();

	m_spinAxisWs.setZero();
	m_steeringOrientationCs.setIdentity();

	m_noSlipIdealSpinVelocity = 0.0f;
	m_spinAngle = 0.0f;
	m_skidEnergyDensity = 0.0f;
	m_sideForce = 0.0f;
	m_forwardSlipVelocity = 0.0f;

	//
	// Set up the collision phantom
	//

	hkAabb aabb;
	getAabb( aabb );
	m_phantom = new hkpAabbPhantom( aabb, m_collisionFilterInfo );
	m_chassis->getWorld()->addPhantom(m_phantom);

	m_rejectChassisListener.m_chassis = m_chassis->getCollidable();
	m_phantom->addPhantomOverlapListener( &m_rejectChassisListener );
}

hkpWheel::~hkpWheel()
{
	if ( m_frictionConstraint != HK_NULL )
	{
		removeFrictionConstraint();
	}

	m_chassis->getWorld()->removePhantom(m_phantom);
	m_chassis = HK_NULL;
	m_phantom->removePhantomOverlapListener(&m_rejectChassisListener);
	m_phantom->removeReference();
}

void hkpWheel::removeFrictionConstraint()
{
	HK_ON_DEBUG(bool constraintRemoved = ) m_chassis->getWorld()->removeConstraint( m_frictionConstraint );
	HK_ASSERT2(0x20ad547, constraintRemoved, "Expected constraint to be removed immediately");
	m_frictionConstraint->removeReference();
	m_frictionConstraint = HK_NULL;
	m_axle->m_numWheelsOnGround--;
	HK_ASSERT2(0x1ab74609, m_frictionData.m_atoms.m_friction.m_axle == m_axle, "Axle mismatch");
	HK_ASSERT2(0xa3165b70, m_axle->m_numWheelsOnGround >= 0, "Axle got negative wheels on ground");
}

void hkpWheel::update(const hkStepInfo& stepInfo)
{
	const hkTransform& chassisTransform = m_chassis->getTransform();

	HK_ASSERT2( 0x155bffe2, m_chassis->isAddedToWorld(), "Vehicle chassis is not added to world.");

	m_suspensionDirectionWs.setRotatedDir( chassisTransform.getRotation(), m_suspensionDirectionCs );
	m_hardPointWs.setTransformedPos( chassisTransform, m_suspensionHardpointCs );
	m_rayEndPointWs.setAddMul( m_hardPointWs, m_suspensionDirectionWs, hkSimdReal::fromFloat( m_suspensionLength + m_radius ) );

	// setAxisAngle version optimized for small angles
	hkQuaternion steeringRotation;
	{
		hkReal halfAngle = 0.5f * m_steeringAngle;
		hkSimdReal sinHalf = hkSimdReal::fromFloat( halfAngle );
		steeringRotation.m_vec.setMul(sinHalf, m_suspensionDirectionCs);
		steeringRotation.m_vec(3) = 1;
		steeringRotation.m_vec.normalize<4>();
	}
	m_steeringOrientationCs = steeringRotation;

	hkVector4 spinAxisCs;
	spinAxisCs.setRotatedDir(m_steeringOrientationCs, m_spinAxisCs);
	m_spinAxisWs.setRotatedDir( chassisTransform.getRotation(), spinAxisCs);

	hkAabb aabb;
	getAabb( aabb );
	m_phantom->markForWrite();
	m_phantom->setAabb( aabb );
	m_phantom->unmarkForWrite();

	collide();

	m_forwardDirectionWs.setCross( m_contactPoint.getNormal(), m_spinAxisWs );
	m_forwardDirectionWs.normalize<3>();

	m_sideDirectionWs.setCross( m_forwardDirectionWs, m_contactPoint.getNormal() );
	m_sideDirectionWs.normalize<3>();

	m_contactLocal.setSub(m_contactPoint.getPosition(), m_chassis->getCenterOfMassInWorld());
}

void hkpWheel::preStep( const hkStepInfo& stepInfo )
{
	// Clear impulses
	m_frictionImpulse.setZero();
	m_suspensionImpulse.setZero();

	// Remove the friction constraint if it is no longer valid
	if ( m_frictionConstraint != HK_NULL && m_frictionConstraint->getRigidBodyB() != m_contactBody )
	{
		removeFrictionConstraint();
	}

	// Remove velocity from fixed wheels
	if ( m_axle->m_isFixed )
	{
		m_axle->m_spinVelocity = 0;
	}

	if ( m_contactBody == HK_NULL )
	{			
		// The wheel is in the air
		m_forwardSlipVelocity = 0.0f;
		m_skidEnergyDensity = 0.0f;
		m_sideForce = 0.0f;

		// Integrate spin velocity now, since there will be no friction constraint to integrate it for us during the step
		m_axle->m_spinVelocity += stepInfo.m_deltaTime * m_torque * m_axle->m_invInertia;

		m_noSlipIdealSpinVelocity = m_axle->m_spinVelocity;
		m_forwardSlipVelocity = 0;
	}
	else
	{
		// Apply suspension and use the result to determine the maximum friction force
		applySuspension( stepInfo );
		
		hkVector4 suspensionForce; suspensionForce.setMul( m_suspensionDirectionWs, hkSimdReal::fromFloat( m_suspensionForce ) );
		hkReal suspensionForceOnNormal = m_contactPoint.getNormal().dot<3>( suspensionForce ).getReal();
		hkReal maximumFrictionForce = hkMath::fabs( suspensionForceOnNormal ) * m_friction * m_contactFriction;

		// Update the friction constraint data
		m_frictionData.setInWorldSpace( m_chassis->getTransform(), m_contactBody->getTransform(), m_contactPoint.getPosition(), m_forwardDirectionWs, m_sideDirectionWs );
		m_frictionData.setMaxFrictionForce( maximumFrictionForce );
		m_frictionData.setTorque( m_torque );

		// If this is our first frame in contact with this body, we need to create a new constraint
		if ( m_frictionConstraint == HK_NULL )
		{
			m_frictionConstraint = new hkpConstraintInstance( m_chassis, m_contactBody, &m_frictionData );
			m_chassis->getWorld()->addConstraint( m_frictionConstraint );
			m_axle->m_numWheelsOnGround++;
			HK_ASSERT2(0x577049e0, m_frictionData.m_atoms.m_friction.m_axle == m_axle, "Axle mismatch");
			HK_ASSERT2(0xd5d927f0, m_axle->m_numWheelsOnGround > 0, "Axle got negative wheels on ground");
		}
	}
}

void hkpWheel::postStep( const hkStepInfo& stepInfo )
{
	// Integrate spin
	m_spinAngle += stepInfo.m_deltaTime * m_axle->m_spinVelocity;

	// Keep spin angle in range
	if ( m_spinAngle > HK_REAL_PI * 1000.0f )
	{
		m_spinAngle -= HK_REAL_PI * 1000.0f;
	}
	else if ( m_spinAngle < -HK_REAL_PI * 1000.0f )
	{
		m_spinAngle += HK_REAL_PI * 1000.0f;
	}

	// Get the applied friction impulse
	if (m_frictionConstraint != HK_NULL)
	{
		m_frictionImpulse.setMul( hkSimdReal::fromFloat( m_frictionData.getForwardFrictionImpulse() ), m_forwardDirectionWs );
		m_frictionImpulse.addMul( hkSimdReal::fromFloat( m_frictionData.getSideFrictionImpulse() ), m_sideDirectionWs );

		// Compute skid energy
		hkVector4 slipImpulse;
		slipImpulse.setMul( hkSimdReal::fromFloat( m_frictionData.getForwardSlipImpulse() ), m_forwardDirectionWs );
		slipImpulse.addMul( hkSimdReal::fromFloat( m_frictionData.getSideSlipImpulse() ), m_sideDirectionWs );

		hkVector4 slipAngular;
		computeAngularChassisAcceleration( slipImpulse, slipAngular );

		hkVector4 slipVelocity;
		slipVelocity.setCross( slipAngular, m_contactLocal );
		slipVelocity.addMul( slipImpulse, hkSimdReal::fromFloat( m_chassis->getMassInv() ) );

		// Translational kinetic energy = 1/2 * m * v^2
		hkReal translationalEnergy = 0.5f * m_chassis->getMass() * slipVelocity.lengthSquared<3>().getReal(); // Joules
		m_skidEnergyDensity = translationalEnergy * 0.001f; // to kilojoules

		// Now that we've extracted the solver results in m_frictionImpulse, we can clear them from the constraint data
		m_frictionData.resetSolverData();

		// Calculate forward slip
		hkVector4 contactVelocity;
		contactVelocity.setCross(m_contactLocal, m_chassis->getAngularVelocity());
		contactVelocity.add(m_chassis->getLinearVelocity());

		m_noSlipIdealSpinVelocity = contactVelocity.dot<3>(m_forwardDirectionWs).getReal() / m_radius;
		m_forwardSlipVelocity = (m_axle->m_spinVelocity - m_noSlipIdealSpinVelocity) * m_radius;
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// Suspension forces are integrated into linear/angular velocities and accumulated	for chassis
// --------------------------------------------------------------------------------------------------------------------------------
void hkpWheel::applySuspension( const hkStepInfo& stepInfo )
{
	// Calculate suspension force
	if ( m_contactBody )
	{
		hkReal force;

		// Spring constant component
		{
			const hkReal lengthDiff = ( m_suspensionLength - m_currentSuspensionLength );
			force = m_suspensionStrength * lengthDiff * m_suspensionScalingFactor;
		}

		// Damping 
		{
			hkReal damping;
			if ( m_suspensionClosingSpeed < 0.0f )
			{
				damping = m_suspensionDampingCompression;
			}
			else
			{
				damping = m_suspensionDampingRelaxation;
			}
			force -= damping * m_suspensionClosingSpeed;
		}

		m_suspensionForce = force * m_chassis->getMass();
	}
	else
	{
		m_suspensionForce = 0.0f;
	}

	// Calculate the impulse and apply it
	if ( m_suspensionForce > 0 )
	{
		hkReal impulse = stepInfo.m_deltaTime * m_suspensionForce;
		m_suspensionImpulse.setMul( hkSimdReal::fromFloat( impulse ), m_contactPoint.getNormal() );
		m_chassis->applyPointImpulse( m_suspensionImpulse, m_hardPointWs );
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// With the final clipped impulse, computes the angular acceleration contribution from this wheel/step for chassis.
// --------------------------------------------------------------------------------------------------------------------------------
void hkpWheel::computeAngularChassisAcceleration( const hkVector4& worldLinearFrictionImpulse, hkVector4& worldAngularVelocityDeltaOut )
{
	// values caching
	const hkRotation& worldFromChassis = m_chassis->getMotion()->getTransform().getRotation();
	const hkVector4& chassisInertiaMassInv = m_chassis->getMotion()->m_inertiaAndMassInv;

	// using worldLinearFrictionImpulse, applies torque (from hardpoint to chassis Center-of-mass) and integrates to get angular acceleration
	hkVector4 worldFrictionAngularImpulse; 
	worldFrictionAngularImpulse.setCross( m_contactLocal, worldLinearFrictionImpulse );
	hkVector4 localFrictionAngularImpulse; 
	localFrictionAngularImpulse._setRotatedInverseDir( worldFromChassis, worldFrictionAngularImpulse ); // to chassis space to get accel dividing by I
	hkVector4 localAngularVelocityDelta; 
	localAngularVelocityDelta.setMul(chassisInertiaMassInv, localFrictionAngularImpulse);
	worldAngularVelocityDeltaOut.setRotatedDir(worldFromChassis,localAngularVelocityDelta); // back to world space
}

// Calculate the current position and rotation of a wheel for the graphics engine
void hkpWheel::calcCurrentPositionAndRotation( hkVector4& posOut, hkQuaternion& rotOut )
{
	//
	//	concatenate the matrices for the wheels
	//
	{
		hkQuaternion spinRotation; spinRotation.setAxisAngle( m_spinAxisCs, -m_spinAngle);
		HK_ASSERT(0x3ea913d9, spinRotation.isOk());

		const hkQuaternion& chassisOrientation = m_chassis->getRotation();
		HK_ASSERT(0x4a5e595c, chassisOrientation.isOk());
		hkQuaternion tmp;
		tmp.setMul( chassisOrientation, m_steeringOrientationCs );
		HK_ASSERT(0x22388644, tmp.isOk());
		rotOut.setMul(tmp, spinRotation);
		HK_ASSERT(0x337dce91, rotOut.isOk());
	}

	const hkReal suspLen = hkMath::max2( hkReal(0.0f), m_currentSuspensionLength );
	posOut._setTransformedPos( m_chassis->getTransform(), m_suspensionHardpointCs );
	posOut.addMul( hkSimdReal::fromFloat( suspLen ), m_suspensionDirectionWs );
}

void hkpWheel::getAabb( hkAabb& aabb )
{
	// Get half extents of wheel in local space where spin axis is (0, 0, 1)
	hkVector4 halfExtents;
	halfExtents.set( m_radius + HK_REAL_EPSILON, m_radius + HK_REAL_EPSILON, 0.5f * m_width + HK_REAL_EPSILON );

	hkQuaternion rotation;
	rotation.setMul( m_chassis->getRotation(), m_steeringOrientationCs );

	// Get rotation to spin axis
	hkReal angle = hkMath::acos( m_spinAxisCs( 2 ) );
	if (angle > HK_REAL_EPSILON)
	{
		hkVector4 sideAxisCs; sideAxisCs.set( 0, 0, 1 );
		hkVector4 rotationAxis; rotationAxis.setCross( sideAxisCs, m_spinAxisCs );
		rotationAxis.normalize<3>();
		hkQuaternion rotationToSpinAxis; rotationToSpinAxis.setAxisAngle( rotationAxis, hkSimdReal::fromFloat( angle ) );
		rotation.setMul( rotation, rotationToSpinAxis );
	}

	hkVector4 wheelEnd;
	wheelEnd.setAddMul( m_hardPointWs, m_suspensionDirectionWs, hkSimdReal::fromFloat( m_suspensionLength ) );

	hkTransform transform;
	transform.set( rotation, wheelEnd );

	hkAabbUtil::calcAabb( transform, halfExtents, hkSimdReal_0, aabb );
}

// Returns the position of the contact point in world space
const hkVector4& hkpWheel::getContactPosition()
{
	return m_contactPoint.getPosition();
}

// Returns the normal of the contact point in world space
const hkVector4& hkpWheel::getContactNormal()
{
	return m_contactPoint.getNormal();
}

// Returns the wheel's hardpoint in world space
const hkVector4& hkpWheel::getHardPoint()
{
	return m_hardPointWs;
}

hkReal hkpWheel::getSpinVelocity()
{
	return m_axle->m_spinVelocity;
}

void hkpWheel::setSpinVelocity( hkReal spinVelocity )
{
	m_axle->m_spinVelocity = spinVelocity;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
