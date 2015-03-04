/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>

#include <Physics2012/Vehicle/hkpVehicleInstance.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Vehicle/Simulation/Default/hkpVehicleDefaultSimulation.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Vehicle/VelocityDamper/Default/hkpVehicleDefaultVelocityDamper.h>
#include <Physics/ConstraintSolver/VehicleFriction/hkpVehicleFrictionSolver.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
hkpVehicleDefaultSimulation::hkpVehicleDefaultSimulation()
{
}


// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::simulateVehicle( hkpVehicleInstance* instance, const hkStepInfo& stepInfo, const SimulationInput& simulInput, hkpVehicleJobResults* vehicleJobResults )
{
	const hkArray<hkReal>& suspensionForceAtWheel = simulInput.suspensionForceAtWheel;
	const hkArray<hkReal>& totalLinearForceAtWheel = simulInput.totalLinearForceAtWheel;
	const hkpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo = simulInput.aerodynamicsDragInfo;
	hkpVehicleData* vehicleData = instance->m_data;

	// Vehicles are presumed to have two axles: one with all of the steering wheels, and the other with the rest.
	HK_ASSERT(0x606560c5, instance->m_data->m_numWheelsPerAxle.getSize() == 2);

	// Set up friction params
	hkpVehicleFrictionSolverParams frictionParams;
	for ( int i = 0; i < 2; ++i )
	{
		frictionParams.m_axleParams[i].initialize();
	}
	hkpRigidBody* groundBody[2] = { HK_NULL, HK_NULL };
	// Use local storage for velocity accumulators to represent the ground body/bodies.
	hkpVelocityAccumulator groundAccum[2];
	hkpVelocityAccumulator groundAccumAtIntegration[2];
	hkVector4 groundOriginalLinearVel[2];
	hkVector4 groundOriginalAngularVel[2];
	hkReal deltaTime = stepInfo.m_deltaTime;

	// Gather friction parameters for each wheel and each axle
	prepareAxleParams( *instance, deltaTime, suspensionForceAtWheel, totalLinearForceAtWheel, groundBody, frictionParams, stepInfo, groundAccum, groundAccumAtIntegration );
	prepareChassisParams( *instance, stepInfo, frictionParams );

	// Apply forces to accumulators
	hkpVehicleJobResults temporaryVehicleResults;
	hkpVehicleJobResults& vehicleResults = vehicleJobResults ? *vehicleJobResults : temporaryVehicleResults;
	applyAerodynamicDrag( *instance, aerodynamicsDragInfo, frictionParams, deltaTime );
	applyVelocityDamping( deltaTime, frictionParams, instance->m_velocityDamper );
	applySuspensionForces( *instance, deltaTime, suspensionForceAtWheel, groundBody, frictionParams, vehicleResults.m_groundBodyImpulses );
	applyDampingToAxleAccumulators( stepInfo, groundBody, frictionParams, groundOriginalLinearVel, groundOriginalAngularVel );
	applyDampingToChassisAccumulator( stepInfo, frictionParams.m_chassis, instance->getChassis()->getMotion() );
	// Fast turn torque
	getExtraTorqueFactor( *instance, deltaTime, frictionParams.m_chassis );

	// APPLY THE FRICTION SOLVER  
	applyFrictionSolver( *instance, stepInfo, frictionParams, m_frictionStatus );

	// Gathering results
	calcChassisVelocities( *instance, frictionParams.m_chassis, vehicleResults );
	calcGroundBodyVelocities( *instance, deltaTime, frictionParams.m_axleParams, groundBody, vehicleResults, groundOriginalLinearVel, groundOriginalAngularVel );

	// Applying forces from step (only if no object was provided to save the output)
	// if the output was provided, then it will be used outside in a multi-threading environment
	if ( vehicleJobResults == HK_NULL )
	{
		vehicleResults.applyForcesFromStep( *instance );
	}

	// Output results to wheel info
	hkVector4 forward_ws;	
	forward_ws.setRotatedDir( instance->getChassis()->getTransform().getRotation(), vehicleData->m_chassisOrientation.getColumn<1>() );
	const hkReal chassis_lin_vel = frictionParams.m_chassis.m_linearVel.dot<3>( forward_ws ).getReal();

	for ( int w_it = 0; w_it < instance->getNumWheels(); ++w_it )
	{
		hkpVehicleInstance::WheelInfo &wheel_info = instance->m_wheelsInfo[w_it];

		int axle = vehicleData->m_wheelParams[w_it].m_axle;
		// take the wheel data to fill, and the axle which it corresponds to
		const hkpVehicleFrictionStatus::AxisStatus& axisStatus = m_frictionStatus.m_axis[ axle ];

		wheel_info.m_forwardSlipVelocity = axisStatus.m_forward_slip_velocity;
		wheel_info.m_sideForce = axisStatus.m_side_force;
		wheel_info.m_sideSlipVelocity = axisStatus.m_side_slip_velocity;
		wheel_info.m_skidEnergyDensity = axisStatus.m_skid_energy_density;

		// spin velocity (with and w/o slipping and spin angle depend on chassis linear velocity)
		if ( !instance->m_isFixed[w_it] )
		{
			const hkReal virt_wheel_velocity = chassis_lin_vel;
			hkReal surface_relative_wheel_vel = virt_wheel_velocity;
			if( groundBody[axle] )
			{
				const hkVector4 linVelWs = frictionParams.m_axleParams[axle].m_groundObject->m_linearVel;
				surface_relative_wheel_vel = virt_wheel_velocity - linVelWs.dot<3>( forward_ws ).getReal();
			}

			hkReal spin_velocity = ( surface_relative_wheel_vel ) / (vehicleData->m_wheelParams[w_it].m_radius);
			wheel_info.m_noSlipIdealSpinVelocity = spin_velocity;

			surface_relative_wheel_vel += wheel_info.m_forwardSlipVelocity;

			spin_velocity = ( surface_relative_wheel_vel ) / (vehicleData->m_wheelParams[w_it].m_radius);
			wheel_info.m_spinVelocity = spin_velocity;
			const hkReal current_angle = instance->m_wheelsInfo[w_it].m_spinAngle;
			const hkReal delta_angle = spin_velocity * deltaTime;
			wheel_info.m_spinAngle = current_angle + delta_angle;
		}
		else
		{
			wheel_info.m_spinAngle = instance->m_wheelsInfo[w_it].m_spinAngle;
			wheel_info.m_spinVelocity = 0.0f;
			wheel_info.m_noSlipIdealSpinVelocity = 0.0f;
		}

		// Keep spin angle in range
		if ( wheel_info.m_spinAngle > HK_REAL_PI * 1000.0f )
		{
			wheel_info.m_spinAngle -= HK_REAL_PI * 1000.0f;
		}
		else if ( wheel_info.m_spinAngle < -HK_REAL_PI * 1000.0f )
		{
			wheel_info.m_spinAngle += HK_REAL_PI * 1000.0f;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::prepareAxleParams( const hkpVehicleInstance& vehicleInstance, hkReal deltaTime, const hkArray<hkReal>& suspensionForceAtWheel, 
													const hkArray<hkReal>& totalLinearForceAtWheel,
													hkpRigidBody* groundBody[], hkpVehicleFrictionSolverParams& frictionParams, 
													const hkStepInfo& stepInfo,
													hkpVelocityAccumulator groundAccum[], 
													hkpVelocityAccumulator groundAccumAtIntegration[] )
{
	//
	// Gather wheel params
	//
	hkReal estimatedCarSpeed = vehicleInstance.getChassis()->getLinearVelocity().length<3>().getReal();
	for ( int w_it = 0 ; w_it < vehicleInstance.m_data->m_numWheels ; ++w_it )
	{
		const int axle = vehicleInstance.m_data->m_wheelParams[w_it].m_axle;
		const hkpVehicleInstance::WheelInfo& wheel_info = vehicleInstance.m_wheelsInfo[w_it];
		hkpRigidBody *const ground = wheel_info.m_contactBody;

		// search for the ground with the highest mass
		if ( ground && ( !groundBody[axle] || ground->getRigidMotion()->getMassInv() < groundBody[axle]->getRigidMotion()->getMassInv() ) )
		{
			groundBody[axle] = ground;
		}

		getAxleParamsFromWheel( vehicleInstance, w_it, totalLinearForceAtWheel[w_it], suspensionForceAtWheel[w_it], estimatedCarSpeed, frictionParams.m_axleParams[axle] );
	}

	//
	// Gather axle params
	//
	frictionParams.m_maxVelocityForPositionalFriction = vehicleInstance.m_data->m_maxVelocityForPositionalFriction;

	for ( int ax_it = 0; ax_it < vehicleInstance.m_data->m_numWheelsPerAxle.getSize(); ++ax_it )
	{
		hkpVehicleFrictionSolverAxleParams &axle_params = frictionParams.m_axleParams[ax_it];
		axle_params.m_constraintNormalWs.normalize<3>();
		axle_params.m_forwardDirWs.normalize<3>();
#ifdef USE_IMPORTANCE
		const hkReal inv_total_importance = 1.0f / (axle_params.m_wheelDownForce + 0.01f * vehicleData.m_numWheelsPerAxle[ax_it]);
		axle_params.m_contactPoint.m_position.   mul4(inv_total_importance);
		axle_params.m_frictionCoefficient.       mul4(inv_total_importance);
		axle_params.m_viscosityFrictionCoefficient *= inv_total_importance;
		axle_params.m_slipVelocityFactor           *= inv_total_importance;
		axle_params.m_maxFrictionCoefficient       *= inv_total_importance;
		HK_ASSERT2(0x681e3abc,  axle_params.m_maxFrictionCoefficient > 0.0f, "New wheel parameter 'maxFriction' (since version 2.2.1) not set" );
#endif
		hkpRigidBody* ground = groundBody[ax_it];
		axle_params.m_groundObject = &groundAccum[ax_it];
		axle_params.m_groundObjectAtLastIntegration = &groundAccum[ax_it];

		if ( !ground  )
		{
			groundAccum[ax_it].setFixed();
		}
		else if ( ground->isFixed() )
		{
			handleFixedGroundAccum( ground, groundAccum[ax_it] );
		}
		else if ( ax_it > 0 && ground == groundBody[0] )
		{
			axle_params.m_groundObject = frictionParams.m_axleParams[0].m_groundObject;
			axle_params.m_groundObjectAtLastIntegration = frictionParams.m_axleParams[0].m_groundObjectAtLastIntegration;
		}
		else
		{
			hkpMotion *const motion = ground->getRigidMotion();
			hkRigidMotionUtilBuildAccumulators( stepInfo, &motion, 1, 0, &groundAccum[ax_it] );

			// If the vehicle is much heavier than the object it collides with (and/or the wheels slip due to the object's friction)
			// the wheel can end up with a very high slip velocity that it then fails to recover from,
			// here we clip relative masses to maxMassRatio:1 if necessary
			const hkSimdReal chassisMassInv = vehicleInstance.getChassis()->getRigidMotion()->getMassInv();
			const hkSimdReal groundMassInv  = motion->getMassInv();
			const hkSimdReal maxMassRatio = hkSimdReal::fromFloat(vehicleInstance.m_data->m_maxFrictionSolverMassRatio);
			if ( chassisMassInv * maxMassRatio < groundMassInv)
			{
				// Scale mass by f to get masses to be 1:maxMassRatio (car:body) so scale inverse masses by inverse of this
				const hkSimdReal massRatio = chassisMassInv / groundMassInv;
				const hkSimdReal f = massRatio * maxMassRatio;
				groundAccum[ax_it].m_invMasses.mul( f );
			}

			const hkMotionState* ms = motion->getMotionState();
			hkVector4 linVel;
			hkVector4 angVel;
			hkSweptTransformUtil::getVelocity( *ms, linVel, angVel );

			groundAccumAtIntegration[ax_it] = groundAccum[ax_it];
			axle_params.m_groundObjectAtLastIntegration = &groundAccumAtIntegration[ax_it];
			groundAccumAtIntegration[ax_it].m_angularVel._setRotatedDir( groundAccum[ax_it].getCoreFromWorldMatrix(), angVel );
			groundAccumAtIntegration[ax_it].m_linearVel  = linVel;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::handleFixedGroundAccum( hkpRigidBody* ground, hkpVelocityAccumulator& accum )
{
	accum.setFixed();
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::getAxleParamsFromWheel( const hkpVehicleInstance& vehicleInstance, int w_it, hkReal linearForceAtWheel, hkReal suspensionForceAtWheel, hkReal estimatedCarSpeed, hkpVehicleFrictionSolverAxleParams& axle_params )
{
	const int axle = vehicleInstance.m_data->m_wheelParams[w_it].m_axle;
	const hkpVehicleInstance::WheelInfo& wheel_info = vehicleInstance.m_wheelsInfo[w_it];
	const hkVector4& spin_axis_ws = wheel_info.m_spinAxisWs;

	const hkVector4& contact_ws     = wheel_info.m_contactPoint.getPosition();
	const hkVector4& surf_normal_ws = wheel_info.m_contactPoint.getNormal();
	hkVector4 forward_ws;		forward_ws.setCross(surf_normal_ws, spin_axis_ws);
	if ( forward_ws.lengthSquared<3>().isLess(hkSimdReal_Eps) )
	{
		forward_ws.setCross( spin_axis_ws, vehicleInstance.m_wheelsInfo[w_it].m_suspensionDirectionWs );
	}
	forward_ws.normalize<3>();

	hkVector4 constraint_normal_ws;	constraint_normal_ws.setCross(forward_ws,surf_normal_ws);

	// using wheel_importance is very flaky !
	// please don't use it
	//#define USE_IMPORTANCE
#ifdef USE_IMPORTANCE
	const hkReal wheel_importance = suspensionForceAtWheel + 0.01f;
#else
	const hkReal wheel_importance = 1.0f / vehicleInstance.m_data->m_numWheelsPerAxle[axle];
#endif

	// Use friction of wheel and landscape
	const hkReal contactFriction = vehicleInstance.m_wheelsInfo[w_it].m_contactFriction;
	const hkReal frictionFactor = wheel_importance * contactFriction;

	axle_params.m_contactPointWs.addMul( hkSimdReal::fromFloat(wheel_importance), contact_ws );
	axle_params.m_constraintNormalWs.add( constraint_normal_ws);
	axle_params.m_forwardDirWs.add( forward_ws);
	axle_params.m_frictionCoefficient			+= frictionFactor * vehicleInstance.m_data->m_wheelParams[w_it].m_friction ;
	axle_params.m_viscosityFrictionCoefficient	+= frictionFactor * vehicleInstance.m_data->m_wheelParams[w_it].m_viscosityFriction;
	axle_params.m_maxFrictionCoefficient		+= frictionFactor * vehicleInstance.m_data->m_wheelParams[w_it].m_maxFriction;
	axle_params.m_wheelDownForce				+= suspensionForceAtWheel;
	axle_params.m_forwardForce					+= linearForceAtWheel;
	axle_params.m_wheelFixed				    = axle_params.m_wheelFixed || vehicleInstance.m_isFixed[w_it];
	axle_params.m_slipVelocityFactor			+= wheel_importance* vehicleInstance.m_data->m_wheelParams[w_it].m_slipAngle * estimatedCarSpeed;
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::prepareChassisParams( const hkpVehicleInstance& vehicleInstance, const hkStepInfo& stepInfo, hkpVehicleFrictionSolverParams& frictionParams )
{
	hkpRigidBody* chassis = vehicleInstance.getChassis();
	hkpMotion* motion = chassis->getRigidMotion();
	hkRigidMotionUtilBuildAccumulators( stepInfo, &motion, 1, 0, &frictionParams.m_chassis );

	const hkMotionState* ms = motion->getMotionState();
	hkVector4 linVel;
	hkVector4 angVel;
	hkSweptTransformUtil::getVelocity( *ms, linVel, angVel );

	// override the inertia matrix in local space
	frictionParams.m_chassis.m_invMasses = vehicleInstance.m_data->m_chassisFrictionInertiaInvDiag;
	frictionParams.m_chassisAtLastIntegration = frictionParams.m_chassis;
	frictionParams.m_chassisAtLastIntegration.m_angularVel.setRotatedDir( frictionParams.m_chassis.getCoreFromWorldMatrix(), angVel );
	frictionParams.m_chassisAtLastIntegration.m_linearVel = linVel;
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::applyAerodynamicDrag( const hkpVehicleInstance& vehicleInstance, const hkpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, hkpVehicleFrictionSolverParams& frictionParams, const hkReal deltaTime )
{
	hkpRigidBody* chassis = vehicleInstance.getChassis();
	HK_ASSERT2( 0x494f6181, ( chassis->getMotionType() == hkpMotion::MOTION_BOX_INERTIA ), "Vehicle is assumed to have box motion" );

	const hkSimdReal dt = hkSimdReal::fromFloat(deltaTime);

	hkVector4 impulse;
	impulse.setMul( dt, aerodynamicsDragInfo.m_aerodynamicsForce );
	frictionParams.m_chassis.m_linearVel.addMul( chassis->getRigidMotion()->getMassInv(), impulse );

	impulse.setMul( dt, aerodynamicsDragInfo.m_aerodynamicsTorque );
	impulse._setRotatedInverseDir( chassis->getTransform().getRotation(), impulse );
	hkVector4 dangVel;
	dangVel.setMul( chassis->getMotion()->m_inertiaAndMassInv, impulse );
	dangVel._setRotatedDir( chassis->getTransform().getRotation(), dangVel );
	frictionParams.m_chassis.m_angularVel.add(dangVel);
}
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::applyVelocityDamping( const hkReal deltaTime, hkpVehicleFrictionSolverParams& frictionParams, const hkpVehicleVelocityDamper* damper )
{
	hkVector4 angularVel = frictionParams.m_chassis.m_angularVel;
	const hkReal spinSqrd = angularVel.lengthSquared<3>().getReal();

	const hkpVehicleDefaultVelocityDamper* defaultDamper = static_cast< const hkpVehicleDefaultVelocityDamper* >( damper );

	hkReal exp_time;
	if ( spinSqrd > defaultDamper->m_collisionThreshold * defaultDamper->m_collisionThreshold )
	{
		exp_time = hkMath::max2( hkReal(0.0f), 1.0f - deltaTime * defaultDamper->m_collisionSpinDamping );
	}
	else
	{
		exp_time = hkMath::max2( hkReal(0.0f), 1.0f - deltaTime * defaultDamper->m_normalSpinDamping );
	}

	frictionParams.m_chassis.m_angularVel.setMul( hkSimdReal::fromFloat(exp_time), angularVel );
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::applySuspensionForces( const hkpVehicleInstance& vehicleInstance, hkReal deltaTime, const hkArray<hkReal>& suspensionForceAtWheel, hkpRigidBody* groundBody[], hkpVehicleFrictionSolverParams& frictionParams, hkVector4* suspensionForcesOnGround )
{
	HK_ASSERT2( 0x3e8fc4b5, ( vehicleInstance.getNumWheels() <= hkpVehicleInstance::s_maxNumLocalWheels ), "Number of wheels is greater than max number expected");

	hkpVehicleData* data = vehicleInstance.m_data;
	for ( int w_it = 0; w_it < vehicleInstance.getNumWheels(); ++w_it )
	{
		const hkpVehicleInstance::WheelInfo& wheel_info = vehicleInstance.m_wheelsInfo[w_it];
		hkpRigidBody *const ground = wheel_info.m_contactBody;
		hkVector4 susp_impulse_ws;
		susp_impulse_ws.setMul( hkSimdReal::fromFloat(deltaTime * suspensionForceAtWheel[w_it]), wheel_info.m_contactPoint.getNormal() );

		if ( suspensionForceAtWheel[w_it] > 0 )
		{
			frictionParams.m_chassis.m_linearVel.addMul( vehicleInstance.getChassis()->getMotion()->getMassInv(), susp_impulse_ws );

			// apply point impulse to accumulator
			hkVector4 relMassCenter; relMassCenter.setSub( wheel_info.m_hardPointWs, vehicleInstance.getChassis()->getMotion()->m_motionState.getSweptTransform().m_centerOfMass1 );
			hkVector4 crossWs; crossWs.setCross( relMassCenter, susp_impulse_ws );

			hkVector4 crossLs; crossLs._setRotatedInverseDir( vehicleInstance.getChassis()->getMotion()->getTransform().getRotation(), crossWs );
			hkVector4 deltaVelLs; deltaVelLs.setMul( vehicleInstance.getChassis()->getMotion()->m_inertiaAndMassInv, crossLs );
			hkVector4 deltaVelWs; deltaVelWs._setRotatedDir( vehicleInstance.getChassis()->getMotion()->getTransform().getRotation(), deltaVelLs );
			frictionParams.m_chassis.m_angularVel.add( deltaVelWs );
		}

		if ( ground && !ground->isFixed() )
		{
			const int axle = data->m_wheelParams[w_it].m_axle;		

			// Scale and clip the impulse so it can be applied to the ground.
			{
				// HVK-962: Vehicle force feedback parameter needs asserts
				// -------------------------------------------------------
				// We may want to add an assert at the point that a massive chassis vs
				// lightweight object interaction might occur, or add some extra inverse
				// sliding scale factor that will account for differences in masses.

				// Scale the impulse by the forceFeedbackMultiplier.
				HK_ASSERT2( 0x38521c64, data->m_wheelParams[w_it].m_forceFeedbackMultiplier >= 0.0f, "Negative values are not allowed for the force feedback multiplier" );
				HK_WARN_ONCE_ON_DEBUG_IF( data->m_wheelParams[w_it].m_forceFeedbackMultiplier > 20.0f, 0x5582dad2, "The force feedback multiplier value is large - the forces being applied to objects the vehicle runs over will be clipped." );

				susp_impulse_ws.setMul( -hkSimdReal::fromFloat(data->m_wheelParams[w_it].m_forceFeedbackMultiplier), susp_impulse_ws );

				// Limit the impulse according to the m_maxContactBodyAcceleration.
				const hkSimdReal maxImpulse = hkSimdReal::fromFloat(ground->getMass() * data->m_wheelParams[w_it].m_maxContactBodyAcceleration * deltaTime);
				const hkSimdReal lenSq = susp_impulse_ws.lengthSquared<3>();
				if ( maxImpulse * maxImpulse < lenSq )
				{
					susp_impulse_ws.mul( maxImpulse * lenSq.sqrtInverse() );
				}
			}

			if ( ground == groundBody[axle] )
			{
				frictionParams.m_axleParams[axle].m_groundObject->m_linearVel.addMul(ground->getRigidMotion()->getMassInv(), susp_impulse_ws);

				// apply point impulse to accumulator
				hkVector4 relMassCenter; relMassCenter.setSub( wheel_info.m_hardPointWs, ground->getRigidMotion()->m_motionState.getSweptTransform().m_centerOfMass1 );
				hkVector4 crossWs; crossWs.setCross( relMassCenter, susp_impulse_ws );

				hkVector4 crossLs; crossLs._setRotatedInverseDir( ground->getRigidMotion()->getTransform().getRotation(), crossWs );
				hkVector4 deltaVelLs; deltaVelLs.setMul( ground->getRigidMotion()->m_inertiaAndMassInv, crossLs );
				hkVector4 deltaVelWs; deltaVelWs._setRotatedDir( ground->getRigidMotion()->getTransform().getRotation(), deltaVelLs );
				frictionParams.m_axleParams[axle].m_groundObject->m_angularVel.add( deltaVelWs );

				// Velocity of body will be set from that of accumulator, therefore should not apply same impulse again later
				suspensionForcesOnGround[w_it].setZero();
			}
			else
			{
				// Record impulse to apply to the ground body. These impulses are applied in a thread-safe manner 
				suspensionForcesOnGround[w_it] = susp_impulse_ws;
			}
		}
		else
		{
			suspensionForcesOnGround[w_it].setZero();
		}
	}
}
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::applyDampingToAxleAccumulators( const hkStepInfo& stepInfo, hkpRigidBody** groundBodies, hkpVehicleFrictionSolverParams& frictionParams, hkVector4* originalLinearVels, hkVector4* originalAngularVels )
{
	hkSimdReal zero; zero.setZero();
	const hkSimdReal one = hkSimdReal_1;
	const hkSimdReal deltaTime = hkSimdReal::fromFloat(stepInfo.m_deltaTime);

	if ( groundBodies[0] && !groundBodies[0]->isFixed() )
	{
		hkSimdReal linearDamping;
		linearDamping.setFromHalf(groundBodies[0]->getRigidMotion()->m_motionState.m_linearDamping);
		linearDamping.mul(deltaTime);
		linearDamping.setMax(zero, one - linearDamping);

		hkSimdReal angularDamping;
		angularDamping.setFromHalf(groundBodies[0]->getRigidMotion()->m_motionState.m_angularDamping);
		angularDamping.mul(deltaTime);
		angularDamping.setMax(zero, one - angularDamping);

		hkpVelocityAccumulator* accumulator = frictionParams.m_axleParams[0].m_groundObject;
		accumulator->m_linearVel.mul( linearDamping );
		accumulator->m_angularVel.mul( angularDamping );

		// If velocity is later applied to ground body, the diff against the original velocity of the body is taken (before rotation)
		originalLinearVels[0] = accumulator->m_linearVel;
		originalAngularVels[0] = accumulator->m_angularVel;

		accumulator->m_angularVel._setRotatedDir( accumulator->getCoreFromWorldMatrix(), accumulator->m_angularVel );
	}
	if ( groundBodies[1] && !groundBodies[1]->isFixed() && groundBodies[1] != groundBodies[0] )
	{
		hkSimdReal linearDamping;
		linearDamping.setFromHalf(groundBodies[1]->getRigidMotion()->m_motionState.m_linearDamping);
		linearDamping.mul(deltaTime);
		linearDamping.setMax(zero, one - linearDamping);

		hkSimdReal angularDamping;
		angularDamping.setFromHalf(groundBodies[1]->getRigidMotion()->m_motionState.m_angularDamping);
		angularDamping.mul(deltaTime);
		angularDamping.setMax(zero, one - angularDamping);

		hkpVelocityAccumulator* accumulator = frictionParams.m_axleParams[1].m_groundObject;
		accumulator->m_linearVel.mul( linearDamping );
		accumulator->m_angularVel.mul( angularDamping );

		originalLinearVels[1] = accumulator->m_linearVel;
		originalAngularVels[1] = accumulator->m_angularVel;

		accumulator->m_angularVel._setRotatedDir( accumulator->getCoreFromWorldMatrix(), accumulator->m_angularVel );
	}
}
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::applyDampingToChassisAccumulator( const hkStepInfo& stepInfo, hkpVelocityAccumulator& accumulator, const hkpMotion* motion )
{
	hkSimdReal zero; zero.setZero();
	const hkSimdReal one = hkSimdReal_1;
	const hkSimdReal deltaTime = hkSimdReal::fromFloat(stepInfo.m_deltaTime);

	hkSimdReal linearDamping;
	linearDamping.setFromHalf(motion->m_motionState.m_linearDamping);
	linearDamping.mul(deltaTime);
	linearDamping.setMax(zero, one - linearDamping);

	hkSimdReal angularDamping;
	angularDamping.setFromHalf(motion->m_motionState.m_angularDamping);
	angularDamping.mul(deltaTime);
	angularDamping.setMax(zero, one - angularDamping);

	accumulator.m_linearVel.mul( linearDamping );
	accumulator.m_angularVel.mul( angularDamping );

	accumulator.m_angularVel._setRotatedDir( accumulator.getCoreFromWorldMatrix(), accumulator.m_angularVel );
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::getExtraTorqueFactor( const hkpVehicleInstance& vehicleInstance, hkReal deltaTime, hkpVelocityAccumulator& accumulatorForChassis ) const
{
	hkpVehicleData* data = vehicleInstance.m_data;
	if ( data->m_extraTorqueFactor )
	{
		hkReal f = vehicleInstance.m_mainSteeringAngleAssumingNoReduction * data->m_extraTorqueFactor * deltaTime;
		hkVector4 localDv; localDv.setMul( hkSimdReal::fromFloat(f), data->m_chassisOrientation.getColumn<0>() );
		localDv.mul( data->m_chassisFrictionInertiaInvDiag );
		accumulatorForChassis.m_angularVel.add(localDv );
	}
}
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::applyFrictionSolver( const hkpVehicleInstance& vehicleInstance, const hkStepInfo& stepInfo, hkpVehicleFrictionSolverParams& frictionParams, hkpVehicleFrictionStatus& frictionStatus )
{
	hkpVehicleStepInfo vehStepInfo;
	vehStepInfo.m_deltaTime = float(stepInfo.m_deltaTime);
	vehStepInfo.m_invDeltaTime = float(stepInfo.m_invDeltaTime);
	hkVehicleFrictionApplyVehicleFriction( vehStepInfo, *m_frictionDescription, frictionParams, frictionStatus );
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::calcChassisVelocities( const hkpVehicleInstance& vehicleInstance, hkpVelocityAccumulator& accumulatorForChassis, hkpVehicleJobResults& vehicleResults )
{
	hkpMotion* chassis = vehicleInstance.getChassis()->getRigidMotion();

	vehicleResults.m_chassisLinearVel = accumulatorForChassis.m_linearVel;
	vehicleResults.m_chassisAngularVel._setRotatedDir( chassis->getTransform().getRotation(), accumulatorForChassis.m_angularVel);
}

// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------
void hkpVehicleDefaultSimulation::calcGroundBodyVelocities( const hkpVehicleInstance& vehicleInstance, hkReal deltaTime, const hkpVehicleFrictionSolverAxleParams axleParams[], hkpRigidBody* groundBody[], hkpVehicleJobResults& vehicleResults, hkVector4* originalLinearVels, hkVector4* originalAngularVels )
{
	hkpVehicleData* vehicleData = vehicleInstance.m_data;
	//
	// Apply the force on the ground objects
	//
	for (int ax_it=0; ax_it<vehicleData->m_numWheelsPerAxle.getSize(); ax_it++)
	{
		vehicleResults.m_groundBodyPtr[ax_it] = HK_NULL;

		const hkpVehicleFrictionSolverAxleParams &axle_params = axleParams[ax_it];
		if ( groundBody[ax_it] && !groundBody[ax_it]->isFixed() )
		{
			if ( (ax_it==0)  || (groundBody[0] != groundBody[1]))
			{
				hkVector4 angVelWs; angVelWs._setRotatedInverseDir( axle_params.m_groundObject->getCoreFromWorldMatrix(), axle_params.m_groundObject->m_angularVel);
				hkVector4 linVelWs = axle_params.m_groundObject->m_linearVel;

				//
				// Clip angular velocity
				//
				{
					hkVector4 maxVelChange; maxVelChange.setZero(); maxVelChange.setXYZ( deltaTime * 10.0f );
					hkVector4 diff;   diff.setSub( angVelWs, originalAngularVels[ax_it] );

					diff.setMin( diff, maxVelChange );
					maxVelChange.setNeg<4>( maxVelChange );
					diff.setMax( diff, maxVelChange );
					angVelWs.setAdd( originalAngularVels[ax_it], diff );
				}

				//
				// Clip linear velocity
				//
				{
					hkVector4 maxVelChange; maxVelChange.setZero(); maxVelChange.setXYZ( deltaTime * 10.0f );
					hkVector4 diff;   diff.setSub( linVelWs, originalLinearVels[ax_it] );

					diff.setMin( diff, maxVelChange );
					maxVelChange.setNeg<4>( maxVelChange );
					diff.setMax( diff, maxVelChange );
					linVelWs.setAdd( originalLinearVels[ax_it], diff );
				}

				vehicleResults.m_groundBodyLinearVel[ax_it] = linVelWs;
				vehicleResults.m_groundBodyAngularVel[ax_it] = angVelWs;

				vehicleResults.m_groundBodyPtr[ax_it] = groundBody[ax_it];
			}
		}
	}
}

void hkpVehicleDefaultSimulation::init( hkpVehicleInstance* instance )
{
	// Initialize friction data
	hkpVehicleData* data = instance->m_data;
	{
		hkpRigidBody* chassis = instance->getChassis();

		hkpVehicleFrictionDescription::Cinfo ci;
		ci.m_chassisCenterOfMass = chassis->getCenterOfMassLocal();

		const hkRotation& t= chassis->getTransform().getRotation();

		{
			const hkVector4& invIn = data->m_chassisFrictionInertiaInvDiag;
			hkMatrix3 in;
			in.getColumn(0).setMul( invIn.getComponent<0>(), t.getColumn<0>() );
			in.getColumn(1).setMul( invIn.getComponent<1>(), t.getColumn<1>() );
			in.getColumn(2).setMul( invIn.getComponent<2>(), t.getColumn<2>() );
			ci.m_chassisFrictionInertiaInv.setMulInverse( in , t );
		}
		ci.m_chassisMassInv = chassis->getMassInv();

		// Check that this is the value of the forward direction
		ci.m_directionUp.setAbs( data->m_chassisOrientation.getColumn<0>() );
		ci.m_directionFront.setAbs( data->m_chassisOrientation.getColumn<1>() );
		ci.m_directionRight.setAbs( data->m_chassisOrientation.getColumn<2>() );

		ci.m_frictionEqualizer = data->m_frictionEqualizer;
		{
			for (int a = 0; a < 2; a++ )
			{
				ci.m_wheelAxleAngularInertia[a] = 0.0f;
			}
			for (int i = 0 ; i < data->m_numWheels; i++ )
			{
				int axle = data->m_wheelParams[i].m_axle;
				ci.m_wheelRadius[axle]  = data->m_wheelParams[i].m_radius;
				ci.m_wheelPosition[axle].setAddMul( instance->m_suspension->m_wheelParams[i].m_hardpointChassisSpace ,
					instance->m_suspension->m_wheelParams[i].m_directionChassisSpace,
					hkSimdReal::fromFloat(instance->m_suspension->m_wheelParams[i].m_length) );
				ci.m_wheelAxleAngularInertia[axle] += data->m_wheelParams[i].m_radius * data->m_wheelParams[i].m_mass;
			}
		}
		m_frictionDescription.setAndDontIncrementRefCount(new hkpVehicleFrictionDescription());
		hkVehicleFrictionDescriptionInitValues(ci , *m_frictionDescription );
	}
}

hkpVehicleSimulation* hkpVehicleDefaultSimulation::clone( hkpVehicleInstance* instance )
{
	hkpVehicleDefaultSimulation* newSimulation = new hkpVehicleDefaultSimulation();
	newSimulation->m_frictionDescription = m_frictionDescription;
	return newSimulation;
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
