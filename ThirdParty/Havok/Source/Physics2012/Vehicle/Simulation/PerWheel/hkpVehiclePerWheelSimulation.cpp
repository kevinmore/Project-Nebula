/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/hkpVehicleInstance.h>
#include <Physics2012/Vehicle/Simulation/PerWheel/hkpVehiclePerWheelSimulation.h>
#include <Common/Base/Ext/hkBaseExt.h>

// --------------------------------------------------------------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------------------------------------------------------------
hkpVehiclePerWheelSimulation::hkpVehiclePerWheelSimulation()
{
	m_slipDamping = 0.3f;
	m_impulseScaling = 0.05f;
	m_maxImpulse = 2.f;
	m_takeDynamicVelocity = true;
	m_curbDamping = 10.0f;

	m_world = HK_NULL;
}

hkpVehiclePerWheelSimulation::~hkpVehiclePerWheelSimulation()
{
	if (m_world != HK_NULL)
	{
		removeFromWorld();
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// Initialize with vehicle data
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehiclePerWheelSimulation::init( hkpVehicleInstance* instance )
{
	m_instance = instance;

	const hkpVehicleData& data = *instance->m_data;
	m_wheelData.setSize( data.m_numWheels );

	// Initialize the wheel data
	for (int w_it = 0; w_it < m_wheelData.getSize(); w_it++)
	{
		// Calculate the inertia of the wheel
		const hkpVehicleData::WheelComponentParams& wheelParams = data.m_wheelParams[w_it];
		hkReal invInertia;
		if (wheelParams.m_mass == 0.0f)
		{
			invInertia = 0.0f;
		}
		else
		{
			// Use cylinder inertia. This isn't totally realistic because the wheel inertia should include drive train inertia,
			// but the wheel mass can be set higher to scale it up (wheel mass isn't used for anything else).
			invInertia = 2.0f / (wheelParams.m_mass * wheelParams.m_radius * wheelParams.m_radius);
		}

		// Initialize the friction constraint data
		m_wheelData[w_it].m_axle.init(1, invInertia);
		m_wheelData[w_it].m_frictionData.init( &m_wheelData[w_it].m_axle, wheelParams.m_radius );
	}
}

hkpVehicleSimulation* hkpVehiclePerWheelSimulation::clone( hkpVehicleInstance* instance )
{
	hkpVehiclePerWheelSimulation* newSimulation = new hkpVehiclePerWheelSimulation();

	newSimulation->m_slipDamping = m_slipDamping;
	newSimulation->m_impulseScaling = m_impulseScaling;
	newSimulation->m_maxImpulse = m_maxImpulse;
	newSimulation->m_takeDynamicVelocity = m_takeDynamicVelocity;
	newSimulation->m_curbDamping = m_curbDamping;

	newSimulation->init( instance );

	return newSimulation;
}

void hkpVehiclePerWheelSimulation::addToWorld( hkpWorld* world )
{
	m_world = world;
	world->addWorldPostSimulationListener( this );
}

void hkpVehiclePerWheelSimulation::removeFromWorld()
{
	m_world->removeWorldPostSimulationListener( this );
	m_world = HK_NULL;

	for (int w_it = 0; w_it < m_wheelData.getSize(); w_it++)
	{
		if (m_wheelData[w_it].m_frictionConstraint != HK_NULL)
		{
			removeFrictionConstraint(w_it);
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// High level simulation step
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehiclePerWheelSimulation::simulateVehicle( hkpVehicleInstance* instance, const hkStepInfo& stepInfo, const SimulationInput& simulInput, hkpVehicleJobResults* vehicleJobResults )
{
	HK_ASSERT2( 0x7fe7d20e, vehicleJobResults == HK_NULL, "hpkVehicleSimplePerWheelSimulation cannot be used with the hkpMultithreadedVehicleManager or the hkpVehicleIntegrateJob." );
	HK_ASSERT2( 0x7fe7d20f, instance == m_instance, "Instance mismatch." );

	// Apply suspension and aerodynamics to chassis
	applySuspensionForces( stepInfo, simulInput );
	applyAerodynamicForces( stepInfo, simulInput );

	// Set up constraints to apply friction during the physics step
	setupFriction( stepInfo, simulInput );
}

// --------------------------------------------------------------------------------------------------------------------------------
// Suspension forces are integrated into linear/angular velocities and accumulated	for chassis
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehiclePerWheelSimulation::applySuspensionForces( const hkStepInfo& stepInfo, const SimulationInput& simulInput )
{
	// some chassis values
	hkpRigidBody* chassis = m_instance->getChassis();

	// computing the suspension forces/torques contribution to chassis per wheel
	int numWheels = m_instance->getNumWheels();
	for ( int w_it = 0; w_it < numWheels; ++w_it )
	{
		const hkpVehicleInstance::WheelInfo& wheel_info = m_instance->m_wheelsInfo[w_it];

		if ( simulInput.suspensionForceAtWheel[w_it] > 0 )
		{
			hkSimdReal impulse = hkSimdReal::fromFloat( stepInfo.m_deltaTime * simulInput.suspensionForceAtWheel[w_it] );
			hkVector4 impulseOnNormalWs; impulseOnNormalWs.setMul( impulse, wheel_info.m_contactPoint.getNormal() );

			chassis->applyPointImpulse( impulseOnNormalWs, wheel_info.m_hardPointWs );
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// Aerodynamic forces are integrated into linear/angular velocities and accumulated for chassis
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehiclePerWheelSimulation::applyAerodynamicForces( const hkStepInfo& stepInfo, const SimulationInput& simulInput )
{
	hkpRigidBody* chassis = m_instance->getChassis();
	const hkSimdReal deltaTime = hkSimdReal::fromFloat(stepInfo.m_deltaTime);

	// Apply linear force
	hkVector4 linearVelocity = chassis->getLinearVelocity();
	hkSimdReal invMassDt; invMassDt.setMul(hkSimdReal::fromFloat(chassis->getMassInv()), deltaTime);
	linearVelocity.addMul( invMassDt, simulInput.aerodynamicsDragInfo.m_aerodynamicsForce );
	chassis->setLinearVelocity( linearVelocity );

	// Apply angular force
	hkRotation chassisRot = m_instance->getChassis()->getMotion()->getTransform().getRotation();
	hkVector4 angularImpulseWs; angularImpulseWs.setMul( deltaTime, simulInput.aerodynamicsDragInfo.m_aerodynamicsTorque );
	hkVector4 angularImpulseLs; angularImpulseLs._setRotatedInverseDir( chassisRot, angularImpulseWs ); // to chassis space	
	hkVector4 deltaAngularVelocityLs; deltaAngularVelocityLs.setMul( chassis->getMotion()->m_inertiaAndMassInv, angularImpulseLs );
	hkVector4 deltaAngularVelocityWS; deltaAngularVelocityWS._setRotatedDir( chassisRot, deltaAngularVelocityLs ); // back to world space

	hkVector4 angularVelocity;
	angularVelocity.setAdd(chassis->getAngularVelocity(), deltaAngularVelocityWS);
	chassis->setAngularVelocity(angularVelocity);
}

void hkpVehiclePerWheelSimulation::removeFrictionConstraint(int w_it)
{
	WheelData& wheelData = m_wheelData[w_it];
	hkpRigidBody* chassis = m_instance->getChassis();

	HK_ON_DEBUG( bool constraintRemoved = ) chassis->getWorld()->removeConstraint( wheelData.m_frictionConstraint );
	HK_ASSERT2(0x194c2438, constraintRemoved, "Expected constraint to be removed immediately. Is the simulation being run in an action?");
	wheelData.m_frictionConstraint->removeReference();
	wheelData.m_frictionConstraint = HK_NULL;
	wheelData.m_axle.m_numWheelsOnGround--;
	HK_ASSERT2(0x76f59f8a, wheelData.m_frictionData.m_atoms.m_friction.m_axle == &wheelData.m_axle, "Axle mismatch");
	HK_ASSERT2(0x466c390a, wheelData.m_axle.m_numWheelsOnGround >= 0, "Axle got negative wheels on ground");
}

// --------------------------------------------------------------------------------------------------------------------------------
// Update friction constraints at each wheel
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehiclePerWheelSimulation::setupFriction( const hkStepInfo& stepInfo, const SimulationInput& simulInput )
{
	HK_ASSERT2(0x7fe7d20d, m_instance->getNumWheels() == m_wheelData.getSize(), "Wheel count changed");

	hkpRigidBody* chassis = m_instance->getChassis();

	for ( int w_it = 0; w_it < m_wheelData.getSize(); ++w_it )
	{
		WheelData& wheelData = m_wheelData[w_it];
		hkpVehicleInstance::WheelInfo& wheel_info = m_instance->m_wheelsInfo[w_it];
		const hkpVehicleData::WheelComponentParams& wheelParams = m_instance->m_data->m_wheelParams[w_it];
		hkpRigidBody* contactBody = wheel_info.m_contactBody;
		hkReal torque = simulInput.totalLinearForceAtWheel[w_it] * wheelParams.m_radius;

		// Remove the friction constraint if it is no longer valid
		if ( wheelData.m_frictionConstraint != HK_NULL && wheelData.m_frictionConstraint->getRigidBodyB() != contactBody )
		{
			removeFrictionConstraint(w_it);
		}

		// Remove velocity from fixed wheels
		wheelData.m_axle.m_isFixed = m_instance->m_isFixed[w_it];
		if ( wheelData.m_axle.m_isFixed )
		{
			wheelData.m_axle.m_spinVelocity = 0;
		}
		else
		{
			wheelData.m_axle.m_spinVelocity = wheel_info.m_spinVelocity;
		}

		if ( contactBody == HK_NULL )
		{
			// The wheel is in the air
			wheel_info.m_forwardSlipVelocity = 0.0f;
			wheel_info.m_skidEnergyDensity = 0.0f;
			wheel_info.m_sideForce = 0.0f;

			if (!wheelData.m_axle.m_isFixed)
			{
				// Integrate spin velocity now, since there will be no friction constraint to integrate it for us during the step
				wheelData.m_axle.m_spinVelocity += stepInfo.m_deltaTime * torque * wheelData.m_axle.m_invInertia;
			}

			wheel_info.m_noSlipIdealSpinVelocity = wheelData.m_axle.m_spinVelocity;
			wheel_info.m_forwardSlipVelocity = 0;
		}
		else
		{
			// Calculate the maximum friction force
			hkVector4 suspensionForce; suspensionForce.setMul( wheel_info.m_suspensionDirectionWs, hkSimdReal::fromFloat( simulInput.suspensionForceAtWheel[w_it] ) );
			hkReal suspensionForceOnNormal = wheel_info.m_contactPoint.getNormal().dot<3>( suspensionForce ).getReal();
			hkReal maximumFrictionForce = hkMath::fabs( suspensionForceOnNormal ) * wheelParams.m_friction * wheel_info.m_contactFriction;

			// Get forward and side directions for the wheel perpendicular to the contact
			wheelData.m_forwardDirectionWs.setCross(wheel_info.m_contactPoint.getNormal(), wheel_info.m_spinAxisWs);
			wheelData.m_forwardDirectionWs.normalize<3>();
			wheelData.m_sideDirectionWs.setCross(wheelData.m_forwardDirectionWs, wheel_info.m_contactPoint.getNormal());
			wheelData.m_sideDirectionWs.normalize<3>();

			// Save the relative position of the contact point
			wheelData.m_contactLocal.setSub(wheel_info.m_contactPoint.getPosition(), chassis->getCenterOfMassInWorld());

			// Update the friction constraint data
			wheelData.m_frictionData.setInWorldSpace( chassis->getTransform(), contactBody->getTransform(), wheel_info.m_contactPoint.getPosition(), wheelData.m_forwardDirectionWs, wheelData.m_sideDirectionWs );
			wheelData.m_frictionData.setMaxFrictionForce( maximumFrictionForce );
			wheelData.m_frictionData.setTorque( torque );
			wheelData.m_frictionData.setImpulseScaling( m_impulseScaling, m_maxImpulse );

			// If this is our first frame in contact with this body, we need to create a new constraint
			if ( wheelData.m_frictionConstraint == HK_NULL )
			{
				wheelData.m_frictionConstraint = new hkpConstraintInstance( chassis, contactBody, &wheelData.m_frictionData );
				chassis->getWorld()->addConstraint( wheelData.m_frictionConstraint );
				wheelData.m_axle.m_numWheelsOnGround++;
				HK_ASSERT2(0x1b1069ac, wheelData.m_frictionData.m_atoms.m_friction.m_axle == &wheelData.m_axle, "Axle mismatch");
				HK_ASSERT2(0x65212087, wheelData.m_axle.m_numWheelsOnGround > 0, "Axle got negative wheels on ground");
			}
		}
	}
}

void hkpVehiclePerWheelSimulation::postSimulationCallback( hkpWorld* world )
{
	hkpRigidBody* chassis = m_instance->getChassis();
	const hkReal deltaTime = world->m_dynamicsStepInfo.m_stepInfo.m_deltaTime.val();

	for ( int w_it = 0; w_it < m_wheelData.getSize(); ++w_it )
	{
		WheelData& wheelData = m_wheelData[w_it];
		hkpVehicleInstance::WheelInfo& wheel_info = m_instance->m_wheelsInfo[w_it];
		const hkpVehicleData::WheelComponentParams& wheelParams = m_instance->m_data->m_wheelParams[w_it];

		// Keep spin angle in range
		if ( wheel_info.m_spinAngle > HK_REAL_PI * 1000.0f )
		{
			wheel_info.m_spinAngle -= HK_REAL_PI * 1000.0f;
		}
		else if ( wheel_info.m_spinAngle < -HK_REAL_PI * 1000.0f )
		{
			wheel_info.m_spinAngle += HK_REAL_PI * 1000.0f;
		}

		// Get the applied friction impulse
		if (wheelData.m_frictionConstraint != HK_NULL)
		{
			// Compute slip velocity
			hkVector4 slipImpulse;
			slipImpulse.setMul( hkSimdReal::fromFloat( wheelData.m_frictionData.getForwardSlipImpulse() ), wheelData.m_forwardDirectionWs );
			slipImpulse.addMul( hkSimdReal::fromFloat( wheelData.m_frictionData.getSideSlipImpulse() ), wheelData.m_sideDirectionWs );

			hkVector4 slipAngular;
			computeAngularChassisAcceleration( slipImpulse, wheelData.m_contactLocal, slipAngular );

			hkVector4 slipVelocity;
			slipVelocity.setCross( slipAngular, wheelData.m_contactLocal );
			slipVelocity.addMul( slipImpulse, hkSimdReal::fromFloat( chassis->getMassInv() ) );

			// Apply damping to forward slip velocity
			if (!wheelData.m_axle.m_isFixed)
			{
				hkReal forwardSlipVelocity = slipVelocity.dot<3>(wheelData.m_forwardDirectionWs).getReal();
				wheelData.m_axle.m_spinVelocity -= (forwardSlipVelocity / wheelParams.m_radius) * m_slipDamping * deltaTime;
			}

			// Compute the side force and side slip velocity
			wheel_info.m_sideForce = wheelData.m_frictionData.getSideFrictionImpulse() / deltaTime;
			wheel_info.m_sideSlipVelocity = slipVelocity.dot<3>( wheelData.m_sideDirectionWs ).getReal();

			// Compute skid
			hkReal translationalEnergy = 0.5f * chassis->getMass() * slipVelocity.lengthSquared<3>().getReal(); // Joules
			wheel_info.m_skidEnergyDensity = translationalEnergy * 0.001f; // to kilojoules

			// Now that we've extracted the solver results in m_frictionImpulse, we can clear them from the constraint data
			wheelData.m_frictionData.resetSolverData();

			// Calculate forward slip
			hkVector4 contactVelocity; contactVelocity.setCross(wheelData.m_contactLocal, chassis->getAngularVelocity());
			contactVelocity.add(chassis->getLinearVelocity());

			wheel_info.m_noSlipIdealSpinVelocity = contactVelocity.dot<3>(wheelData.m_forwardDirectionWs).getReal() / wheelParams.m_radius;
			wheel_info.m_forwardSlipVelocity = (wheelData.m_axle.m_spinVelocity - wheel_info.m_noSlipIdealSpinVelocity) * wheelParams.m_radius;
		}
		else
		{
			wheel_info.m_skidEnergyDensity = 0.0f;
			wheel_info.m_sideForce = 0.0f;
			wheel_info.m_forwardSlipVelocity = 0.0f;
			wheel_info.m_sideSlipVelocity = 0.0f;
			wheel_info.m_noSlipIdealSpinVelocity = wheel_info.m_spinVelocity;
		}

		// Integrate spin
		wheel_info.m_spinVelocity = wheelData.m_axle.m_spinVelocity;
		wheel_info.m_spinAngle += deltaTime * wheelData.m_axle.m_spinVelocity;
	}
}

void hkpVehiclePerWheelSimulation::computeAngularChassisAcceleration( const hkVector4& worldLinearFrictionImpulse, const hkVector4& contactLocal, hkVector4& worldAngularVelocityDeltaOut )
{
	// values caching
	hkpRigidBody* chassis = m_instance->getChassis();
	const hkRotation& worldFromChassis = chassis->getMotion()->getTransform().getRotation();
	const hkVector4& chassisInertiaMassInv = chassis->getMotion()->m_inertiaAndMassInv;

	// using worldLinearFrictionImpulse, applies torque (from hardpoint to chassis Center-of-mass) and integrates to get angular acceleration
	hkVector4 worldFrictionAngularImpulse; 
	worldFrictionAngularImpulse.setCross( contactLocal, worldLinearFrictionImpulse );
	hkVector4 localFrictionAngularImpulse; 
	localFrictionAngularImpulse._setRotatedInverseDir( worldFromChassis, worldFrictionAngularImpulse ); // to chassis space to get accel dividing by I
	hkVector4 localAngularVelocityDelta; 
	localAngularVelocityDelta.setMul(chassisInertiaMassInv, localFrictionAngularImpulse);
	worldAngularVelocityDeltaOut.setRotatedDir(worldFromChassis,localAngularVelocityDelta); // back to world space
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
