/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/Brake/hknpVehicleBrake.h>
#include <Physics/Physics/Extensions/Vehicle/Engine/hknpVehicleEngine.h>
#include <Physics/Physics/Extensions/Vehicle/TyreMarks/hknpTyremarksInfo.h>
#include <Physics/Physics/Extensions/Vehicle/Steering/hknpVehicleSteering.h>
#include <Physics/Physics/Extensions/Vehicle/Suspension/hknpVehicleSuspension.h>
#include <Physics/Physics/Extensions/Vehicle/DriverInput/hknpVehicleDriverInput.h>
#include <Physics/Physics/Extensions/Vehicle/AeroDynamics/hknpVehicleAerodynamics.h>
#include <Physics/Physics/Extensions/Vehicle/Transmission/hknpVehicleTransmission.h>
#include <Physics/Physics/Extensions/Vehicle/WheelCollide/hknpVehicleWheelCollide.h>
#include <Physics/Physics/Extensions/Vehicle/VelocityDamper/hknpVehicleVelocityDamper.h>
#include <Physics/Physics/Extensions/Vehicle/Friction/hknpVehicleFriction.h>
#include <Physics/Physics/Extensions/Vehicle/VelocityDamper/Default/hknpVehicleDefaultVelocityDamper.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

// Shared physics dependencies
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics/ConstraintSolver/VehicleFriction/hkpVehicleFrictionSolver.h>
#include <Physics/ConstraintSolver/VehicleFriction/hkpVehicleFriction.h>


// A little helper
static HK_FORCE_INLINE const hknpMotion& _getMotion( const hknpWorld* world, hknpBodyId bodyId )
{
	return world->getMotion( world->getBody(bodyId).m_motionId );
}

hknpVehicleInstance::hknpVehicleInstance( hknpBodyId chassis, hknpWorld* world ) :
	hknpUnaryAction(chassis), m_world(world)
{
	m_data = HK_NULL;
	m_driverInput = HK_NULL;
	m_steering = HK_NULL;
	m_engine = HK_NULL;
	m_transmission = HK_NULL;
	m_brake = HK_NULL;
	m_suspension = HK_NULL;
	m_aerodynamics = HK_NULL;
	m_wheelCollide = HK_NULL;
	m_tyreMarks = HK_NULL;
	m_velocityDamper = HK_NULL;
	m_deviceStatus = HK_NULL;
}

static void convertToStepInfo( const hknpSolverInfo& infoIn, hkStepInfo& stepInfoOut )
{
	stepInfoOut.set( 0, infoIn.m_deltaTime.getReal() );
}

void hknpVehicleInstance::WheelInfo::init()
{
	hkVector4 up; up = hkVector4::getConstant<HK_QUADREAL_0100>();
	m_contactPoint.set( hkVector4::getZero(), up, 1);

	m_contactFriction = 0.0f;
	m_contactBodyId = hknpBodyId::invalid();
	m_contactShapeKey = HKNP_INVALID_SHAPE_KEY;
	m_rayEndPointWs.setZero();

	m_hardPointWs.setZero();
	m_rayEndPointWs.setZero();
	m_currentSuspensionLength = 0.0f;
	m_suspensionDirectionWs.setZero();

	m_spinAxisChassisSpace.setZero();
	m_spinAxisWs.setZero();
	m_spinAxisChassisSpace.set(1.0f, 0.0f, 0.0f); 
	m_steeringOrientationChassisSpace.setIdentity();

	m_spinVelocity = 0.0f;
	m_noSlipIdealSpinVelocity = 0.0f;
	m_spinAngle = 0.0f;
	m_skidEnergyDensity = 0.0f;
	m_sideForce = 0.0f;
	m_forwardSlipVelocity = 0.0f;
	m_sideSlipVelocity = 0.0f;
}

#define REMOVE_REFERENCE(obj) if(obj) obj->removeReference()

hknpVehicleInstance::~hknpVehicleInstance()
{
	REMOVE_REFERENCE( m_data );
	REMOVE_REFERENCE( m_driverInput );
	REMOVE_REFERENCE( m_steering );
	REMOVE_REFERENCE( m_engine );
	REMOVE_REFERENCE( m_transmission );
	REMOVE_REFERENCE( m_brake );
	REMOVE_REFERENCE( m_suspension );
	REMOVE_REFERENCE( m_aerodynamics );
	REMOVE_REFERENCE( m_wheelCollide );
	REMOVE_REFERENCE( m_velocityDamper );
	REMOVE_REFERENCE( m_deviceStatus );

	// Tyremarks are an optional component
	REMOVE_REFERENCE( m_tyreMarks );
}


void hknpVehicleInstance::init()
{
	hknpMotion& motion = accessChassisMotion();

	// The initialization of the data modifies the chassis
	if (!m_data->m_alreadyInitialised)
	{
		m_data->init( m_suspension->m_wheelParams, getChassisTransform(), motion);
	}

	//
	//	Create an inertia matrix for normal vehicle operation
	//

	{  // set diagonal of rot inertia tensor for normal stuff
		hkVector4 y,r,p;

		y.setAbs( m_data->m_chassisOrientation.getColumn<0>() );
		r.setAbs( m_data->m_chassisOrientation.getColumn<1>() );
		p.setAbs( m_data->m_chassisOrientation.getColumn<2>() );

		hkVector4 unitDiagonal;
		unitDiagonal.setMul(hkSimdReal::fromFloat(m_data->m_chassisUnitInertiaYaw),   y);
		unitDiagonal.addMul(hkSimdReal::fromFloat(m_data->m_chassisUnitInertiaRoll),  r);
		unitDiagonal.addMul(hkSimdReal::fromFloat(m_data->m_chassisUnitInertiaPitch), p);

		hkVector4 diagonal; diagonal.setMul( motion.getMass(), unitDiagonal);

		motion.m_inverseInertia[0].setReal<false>( 1.0f / diagonal(0) );
		motion.m_inverseInertia[1].setReal<false>( 1.0f / diagonal(1) );
		motion.m_inverseInertia[2].setReal<false>( 1.0f / diagonal(2) );
		motion.m_inverseInertia[3].setReal<false>( 1.0f / motion.getMass().getReal() );
	}

	m_wheelsInfo.setSize( m_data->m_numWheels );
	{
		for (int i=0;i<m_wheelsInfo.getSize();i++)
		{
			m_wheelsInfo[i].init();
		}
	}


	m_isFixed.setSize( m_data->m_numWheels );
	{
		for (int i=0;i<m_isFixed.getSize();i++)
		{
			m_isFixed[i] = false;
		}
	}
	m_wheelsTimeSinceMaxPedalInput = 0.0f;

	m_mainSteeringAngle = 0.0f;
	m_mainSteeringAngleAssumingNoReduction = 0.0f;
	m_wheelsSteeringAngle.setSize( m_data->m_numWheels );
	{
		for (int i=0;i<m_wheelsSteeringAngle.getSize();i++)
		{
			m_wheelsSteeringAngle[i] = 0.0f;
		}
	}

	// engine
	m_torque = 0.0f;
	m_rpm = 0.0f;

	// transmission
	m_isReversing = false;
	m_currentGear = 0;
	m_delayed = false;
	m_clutchDelayCountdown = 0.0f;

	// wheel collide
	m_wheelCollide->init( this );

	// ensure that any components that should not be shared aren't.
	HK_ASSERT2(0x4e34565e, !m_wheelCollide->m_alreadyUsed, "The wheelCollide component cannot be shared between different vehicle instances.");
	m_wheelCollide->m_alreadyUsed = true;
}

void hknpVehicleInstance::handleFixedGroundAccum( hknpBodyId ground, hkpVelocityAccumulator& accum )
{
	accum.setFixed();
}


void hknpVehicleInstance::updateBeforeCollisionDetection()
{
	HK_TIMER_BEGIN("UpdateBeforeCD", HK_NULL);

	const hkTransform& car_transform = getChassisTransform();

	//
	//	Adjust wheel info using transform.
	//
	for (int w_it=0; w_it < m_data->m_numWheels; w_it ++)
	{
		WheelInfo &wheel_info = m_wheelsInfo[ w_it ];

		wheel_info.m_suspensionDirectionWs.setRotatedDir( car_transform.getRotation(), m_suspension->m_wheelParams[w_it].m_directionChassisSpace );
		wheel_info.m_hardPointWs.setTransformedPos( car_transform, m_suspension->m_wheelParams[w_it].m_hardpointChassisSpace );

		const hkVector4& start_ws = wheel_info.m_hardPointWs;

		const hkReal    spr_length = m_suspension->m_wheelParams[w_it].m_length;
		const hkReal	wheel_radius = m_data->m_wheelParams[w_it].m_radius;

		wheel_info.m_rayEndPointWs.setAddMul( start_ws, wheel_info.m_suspensionDirectionWs, hkSimdReal::fromFloat(spr_length + wheel_radius) );
	}

	m_wheelCollide->updateBeforeCollisionDetection( this );

	HK_TIMER_END();
}

void hknpVehicleInstance::updateWheels( hkReal deltaTime, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo )
{
	const hkTransform& car_transform = getChassisTransform();

	for ( int w_it = 0 ; w_it < m_data->m_numWheels ; ++w_it )
	{
		WheelInfo &wheel_info = m_wheelsInfo[ w_it ];

		//
		// Copy data from the collision info to the wheelInfo.
		//
		{
			m_wheelsInfo[w_it].m_currentSuspensionLength = cdInfo[w_it].m_currentSuspensionLength;
			m_wheelsInfo[w_it].m_contactPoint = cdInfo[w_it].m_contactPoint;
			m_wheelsInfo[w_it].m_contactFriction = cdInfo[w_it].m_contactFriction;
			m_wheelsInfo[w_it].m_contactBodyId = cdInfo[w_it].m_contactBodyId;
			m_wheelsInfo[w_it].m_contactShapeKey = cdInfo[w_it].m_contactShapeKey;
		}

		//
		// Update the steering and wheel position.
		//
		{
			const hkReal steering_angle = m_wheelsSteeringAngle[w_it];

			// setAxisAngle version optimized for small angles
			hkQuaternion steering_rotation;
			{
				hkReal halfAngle = 0.5f * steering_angle;
				hkSimdReal sinHalf = hkSimdReal::fromFloat(halfAngle);
				steering_rotation.m_vec.setMul(sinHalf, m_suspension->m_wheelParams[w_it].m_directionChassisSpace);
				steering_rotation.m_vec(3) = 1;
				steering_rotation.m_vec.normalize<4>();
			}
			wheel_info.m_steeringOrientationChassisSpace = steering_rotation;
			wheel_info.m_spinAxisChassisSpace = m_data->m_chassisOrientation.getColumn<2>();

			hkVector4 spin_axis_cs;
			spin_axis_cs.setRotatedDir(wheel_info.m_steeringOrientationChassisSpace, wheel_info.m_spinAxisChassisSpace);

			wheel_info.m_spinAxisWs.setRotatedDir( car_transform.getRotation(), spin_axis_cs);
		}

		//
		// Spin angle calculation is done later to take account of surface velocity.
		//
	}
}


void hknpVehicleInstance::updateDriverInput( hkReal deltaTime, hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo )
{
	filteredDriverInputInfo.m_tryingToReverse = m_tryingToReverse;
	m_driverInput->calcDriverInput( deltaTime, this, m_deviceStatus, filteredDriverInputInfo );
	m_tryingToReverse = filteredDriverInputInfo.m_tryingToReverse;
}


void hknpVehicleInstance::updateSteering( hkReal deltaTime, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo )
{
	hknpVehicleSteering::SteeringAnglesOutput steeringInfo;
	steeringInfo.m_mainSteeringAngle = m_mainSteeringAngle;
	steeringInfo.m_mainSteeringAngleAssumingNoReduction = m_mainSteeringAngleAssumingNoReduction;
	steeringInfo.m_wheelsSteeringAngle.setSize( m_wheelsSteeringAngle.getSize() );
	{
		for ( int i = 0 ; i < m_wheelsSteeringAngle.getSize() ; i++ )
		{
			steeringInfo.m_wheelsSteeringAngle[i] = m_wheelsSteeringAngle[i];
		}
	}
	m_steering->calcSteering( deltaTime, this, filteredDriverInputInfo, steeringInfo );

	m_mainSteeringAngle = steeringInfo.m_mainSteeringAngle;
	m_mainSteeringAngleAssumingNoReduction = steeringInfo.m_mainSteeringAngleAssumingNoReduction;
	{
		for ( int i = 0 ; i < m_wheelsSteeringAngle.getSize() ; i++ )
		{
			m_wheelsSteeringAngle[i] = steeringInfo.m_wheelsSteeringAngle[i];
		}
	}
}


void hknpVehicleInstance::updateTransmission( hkReal deltaTime, hknpVehicleTransmission::TransmissionOutput& transmissionInfo )
{
	transmissionInfo.m_numWheelsTramsmittedTorque = m_data->m_numWheels;
	transmissionInfo.m_isReversing = m_isReversing;
	transmissionInfo.m_currentGear = m_currentGear;
	transmissionInfo.m_delayed = m_delayed;
	transmissionInfo.m_clutchDelayCountdown = m_clutchDelayCountdown;
	m_transmission->calcTransmission( deltaTime, this, transmissionInfo );
	m_isReversing = transmissionInfo.m_isReversing;
	m_currentGear = transmissionInfo.m_currentGear;
	m_delayed = transmissionInfo.m_delayed;
	m_clutchDelayCountdown = transmissionInfo.m_clutchDelayCountdown;
}


void hknpVehicleInstance::updateEngine( hkReal deltaTime, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo, const hknpVehicleTransmission::TransmissionOutput& transmissionInfo )
{
	hknpVehicleEngine::EngineOutput engineInfo;
	engineInfo.m_rpm = m_rpm;
	engineInfo.m_torque = m_torque;
	m_engine->calcEngineInfo( deltaTime, this, filteredDriverInputInfo, transmissionInfo, engineInfo );
	m_rpm = engineInfo.m_rpm;
	m_torque = engineInfo.m_torque;
}


void hknpVehicleInstance::updateBrake( hkReal deltaTime, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo, hknpVehicleBrake::WheelBreakingOutput& wheelBreakingInfo )
{
	const int numIsFixed = m_isFixed.getSize();
	wheelBreakingInfo.m_isFixed.setSize( numIsFixed );
	wheelBreakingInfo.m_brakingTorque.setSize( numIsFixed );
	{
		for (int i = 0 ; i < numIsFixed; ++i )
		{
			wheelBreakingInfo.m_isFixed[i] = m_isFixed[i];
		}
	}
	wheelBreakingInfo.m_wheelsTimeSinceMaxPedalInput = m_wheelsTimeSinceMaxPedalInput;

	m_brake->calcBreakingInfo( deltaTime, this, filteredDriverInputInfo, wheelBreakingInfo );

	// copy back the brake cache data
	{
		for (int i = 0 ; i < numIsFixed; ++i )
		{
			m_isFixed[i] = wheelBreakingInfo.m_isFixed[i];
		}
		m_wheelsTimeSinceMaxPedalInput = wheelBreakingInfo.m_wheelsTimeSinceMaxPedalInput;
	}
}


void hknpVehicleInstance::updateSuspension( hkReal deltaTime, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hkArray<hkReal>& suspensionForces )
{
	m_suspension->calcSuspension( deltaTime, this, cdInfo, suspensionForces.begin() );
}


void hknpVehicleInstance::updateAerodynamics( hkReal deltaTime, hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo )
{
	m_aerodynamics->calcAerodynamics( deltaTime, this, aerodynamicsDragInfo );
}


void hknpVehicleInstance::applySuspensionForces( hkReal deltaTime, const hkArray<hkReal>& suspensionForceAtWheel, hknpBodyId groundBody[], hkpVehicleFrictionSolverParams& frictionParams, hkVector4* suspensionForcesOnGround )
{
	HK_ASSERT2( 0x3e8fc4b5, ( getNumWheels() <= hknpVehicleInstance::s_maxNumLocalWheels ), "Number of wheels is greater than max number expected");

	const hkRotation& chassisRotation = getChassisTransform().getRotation();
	const hknpMotion& motion = getChassisMotion();

	for ( int w_it = 0; w_it < getNumWheels(); ++w_it )
	{
		const WheelInfo& wheel_info = m_wheelsInfo[w_it];
		hknpBodyId ground = wheel_info.m_contactBodyId;

		hkVector4 susp_impulse_ws;
		susp_impulse_ws.setMul( hkSimdReal::fromFloat(deltaTime * suspensionForceAtWheel[w_it]), wheel_info.m_contactPoint.getNormal() );

		if ( suspensionForceAtWheel[w_it] > 0 )
		{
			frictionParams.m_chassis.m_linearVel.addMul( motion.getInverseMass(), susp_impulse_ws );

			// apply point impulse to accumulator
			hkVector4 relMassCenter; relMassCenter.setSub( wheel_info.m_hardPointWs, motion.getCenterOfMassInWorld() );
			hkVector4 crossWs; crossWs.setCross( relMassCenter, susp_impulse_ws );

			hkVector4 crossLs; crossLs._setRotatedInverseDir( chassisRotation, crossWs );
			hkVector4 m_inertiaAndMassInv; motion.getInverseInertiaLocal(m_inertiaAndMassInv);
			hkVector4 deltaVelLs; deltaVelLs.setMul( m_inertiaAndMassInv, crossLs );
			hkVector4 deltaVelWs; deltaVelWs._setRotatedDir( chassisRotation, deltaVelLs );

			frictionParams.m_chassis.m_angularVel.add( deltaVelWs );
		}

		if ( ground.isValid() )
		{
			const hknpMotion& groundMotion = _getMotion(m_world, ground);
			if( !groundMotion.isStatic() )
			{
				const int axle = m_data->m_wheelParams[w_it].m_axle;

				// Scale and clip the impulse so it can be applied to the ground.
				{
					// HVK-962: Vehicle force feedback parameter needs asserts
					// -------------------------------------------------------
					// We may want to add an assert at the point that a massive chassis vs
					// lightweight object interaction might occur, or add some extra inverse
					// sliding scale factor that will account for differences in masses.

					// Scale the impulse by the forceFeedbackMultiplier.
					HK_ASSERT2( 0x38521c64, m_data->m_wheelParams[w_it].m_forceFeedbackMultiplier >= 0.0f, "Negative values are not allowed for the force feedback multiplier" );
					HK_WARN_ONCE_ON_DEBUG_IF( m_data->m_wheelParams[w_it].m_forceFeedbackMultiplier > 20.0f, 0x5582dad2, "The force feedback multiplier value is large - the forces being applied to objects the vehicle runs over will be clipped." );

					susp_impulse_ws.setMul( -hkSimdReal::fromFloat(m_data->m_wheelParams[w_it].m_forceFeedbackMultiplier), susp_impulse_ws );

					// Limit the impulse according to the m_maxContactBodyAcceleration.
					const hkSimdReal maxImpulse = hkSimdReal::fromFloat( groundMotion.getMass().getReal() * m_data->m_wheelParams[w_it].m_maxContactBodyAcceleration * deltaTime);
					const hkSimdReal lenSq = susp_impulse_ws.lengthSquared<3>();
					if ( maxImpulse * maxImpulse < lenSq )
					{
						susp_impulse_ws.mul( maxImpulse * lenSq.sqrtInverse() );
					}
				}

				if ( ground == groundBody[axle] )
				{
					frictionParams.m_axleParams[axle].m_groundObject->m_linearVel.addMul( groundMotion.getInverseMass(), susp_impulse_ws );

					// apply point impulse to accumulator
					hkVector4 relMassCenter; relMassCenter.setSub( wheel_info.m_hardPointWs, groundMotion.getCenterOfMassInWorld() );
					hkVector4 crossWs; crossWs.setCross( relMassCenter, susp_impulse_ws );

					hkVector4 crossLs; crossLs._setRotatedInverseDir( m_world->getBody(ground).getTransform().getRotation(), crossWs );
					hkVector4 invInertiaLocal; groundMotion.getInverseInertiaLocal(invInertiaLocal);
					hkVector4 deltaVelLs; deltaVelLs.setMul( invInertiaLocal, crossLs );
					hkVector4 deltaVelWs; deltaVelWs._setRotatedDir( m_world->getBody(ground).getTransform().getRotation(), deltaVelLs );
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
		}
		else
		{
			suspensionForcesOnGround[w_it].setZero();
		}
	}
}

void hknpVehicleInstance::getAxleParamsFromWheel( int w_it, hkReal linearForceAtWheel, hkReal suspensionForceAtWheel, hkReal estimatedCarSpeed, hkpVehicleFrictionSolverAxleParams& axle_params )
{
	const int axle = m_data->m_wheelParams[w_it].m_axle;
	const WheelInfo& wheel_info = m_wheelsInfo[w_it];
	const hkVector4& spin_axis_ws = wheel_info.m_spinAxisWs;

	const hkVector4& contact_ws     = wheel_info.m_contactPoint.getPosition();
	const hkVector4& surf_normal_ws = wheel_info.m_contactPoint.getNormal();
	hkVector4 forward_ws;		forward_ws.setCross(surf_normal_ws, spin_axis_ws);
	if ( forward_ws.lengthSquared<3>().isLess(hkSimdReal_Eps) )
	{
		forward_ws.setCross( spin_axis_ws, m_wheelsInfo[w_it].m_suspensionDirectionWs );
	}
	forward_ws.normalize<3>();

	hkVector4 constraint_normal_ws;	constraint_normal_ws.setCross(forward_ws,surf_normal_ws);

	// using wheel_importance is very flaky !
	// please don't use it
	//#define USE_IMPORTANCE
#ifdef USE_IMPORTANCE
	const hkReal wheel_importance = suspensionForceAtWheel + 0.01f;
#else
	const hkReal wheel_importance = 1.0f / m_data->m_numWheelsPerAxle[axle];
#endif

	// Use friction of wheel and landscape
	const hkReal contactFriction = m_wheelsInfo[w_it].m_contactFriction;
	const hkReal frictionFactor = wheel_importance * contactFriction;

	axle_params.m_contactPointWs.addMul( hkSimdReal::fromFloat(wheel_importance), contact_ws );
	axle_params.m_constraintNormalWs.add( constraint_normal_ws);
	axle_params.m_forwardDirWs.add( forward_ws);
	axle_params.m_frictionCoefficient			+= frictionFactor * m_data->m_wheelParams[w_it].m_friction ;
	axle_params.m_viscosityFrictionCoefficient	+= frictionFactor * m_data->m_wheelParams[w_it].m_viscosityFriction;
	axle_params.m_maxFrictionCoefficient		+= frictionFactor * m_data->m_wheelParams[w_it].m_maxFriction;
	axle_params.m_wheelDownForce				+= suspensionForceAtWheel;
	axle_params.m_forwardForce					+= linearForceAtWheel;
	axle_params.m_wheelFixed				    = axle_params.m_wheelFixed || m_isFixed[w_it];
	axle_params.m_slipVelocityFactor			+= wheel_importance* m_data->m_wheelParams[w_it].m_slipAngle * estimatedCarSpeed;
}

void hknpVehicleInstance::prepareAxleParams( hkReal deltaTime, const hkArray<hkReal>& suspensionForceAtWheel, const hkArray<hkReal>& totalLinearForceAtWheel,
										   hknpBodyId groundBody[], hkpVehicleFrictionSolverParams& frictionParams, const hkStepInfo& stepInfo,
										   hkpVelocityAccumulator groundAccum[], hkpVelocityAccumulator groundAccumAtIntegration[] )
{
	//
	// Gather wheel params
	//
	hkReal estimatedCarSpeed = getChassisMotion().getLinearVelocity().length<3>().getReal();
	for ( int w_it = 0 ; w_it < m_data->m_numWheels ; ++w_it )
	{
		const int axle = m_data->m_wheelParams[w_it].m_axle;
		const WheelInfo& wheel_info = m_wheelsInfo[w_it];
		hknpBodyId ground = wheel_info.m_contactBodyId;

		// search for the ground with the highest mass
		if ( ground.isValid() && ( !groundBody[axle].isValid() || _getMotion(m_world,ground).getInverseMass() < _getMotion(m_world,groundBody[axle]).getInverseMass() ) )
		{
			groundBody[axle] = ground;
		}

		getAxleParamsFromWheel( w_it, totalLinearForceAtWheel[w_it], suspensionForceAtWheel[w_it], estimatedCarSpeed, frictionParams.m_axleParams[axle] );
}

	//
	// Gather axle params
	//
	frictionParams.m_maxVelocityForPositionalFriction = m_data->m_maxVelocityForPositionalFriction;

	for ( int ax_it = 0; ax_it < m_data->m_numWheelsPerAxle.getSize(); ++ax_it )
	{
		hkpVehicleFrictionSolverAxleParams &axle_params = frictionParams.m_axleParams[ax_it];
		axle_params.m_constraintNormalWs.normalize<3>();
		axle_params.m_forwardDirWs.normalize<3>();
	#ifdef USE_IMPORTANCE
		const hkReal inv_total_importance = 1.0f / (axle_params.m_wheelDownForce + 0.01f * m_data->m_numWheelsPerAxle[ax_it]);
		axle_params.m_contactPoint.m_position.   mul4(inv_total_importance);
		axle_params.m_frictionCoefficient.       mul4(inv_total_importance);
		axle_params.m_viscosityFrictionCoefficient *= inv_total_importance;
		axle_params.m_slipVelocityFactor           *= inv_total_importance;
		axle_params.m_maxFrictionCoefficient       *= inv_total_importance;
		HK_ASSERT2(0x681e3abc,  axle_params.m_maxFrictionCoefficient > 0.0f, "New wheel parameter 'maxFriction' (since version 2.2.1) not set" );
	#endif
		hknpBodyId ground = groundBody[ax_it];
		axle_params.m_groundObject = &groundAccum[ax_it];
		axle_params.m_groundObjectAtLastIntegration = &groundAccum[ax_it];

		if ( !ground.isValid()  )
		{
			groundAccum[ax_it].setFixed();
		}
		else if ( _getMotion(m_world,ground).isStatic() )
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
			const hknpMotion& motion = _getMotion( m_world, ground );
			buildAccumulator( stepInfo, motion, hkTransform::getIdentity(), groundAccum[ax_it] );

			// If the vehicle is much heavier than the object it collides with (and/or the wheels slip due to the object's friction)
			// the wheel can end up with a very high slip velocity that it then fails to recover from,
			// here we clip relative masses to maxMassRatio:1 if necessary
			const hkSimdReal chassisMassInv = getChassisMotion().getInverseMass();
			const hkSimdReal groundMassInv  = motion.getInverseMass();
			const hkSimdReal maxMassRatio = hkSimdReal::fromFloat(m_data->m_maxFrictionSolverMassRatio);
			if ( chassisMassInv * maxMassRatio < groundMassInv)
			{
				// Scale mass by f to get masses to be 1:maxMassRatio (car:body) so scale inverse masses by inverse of this
				const hkSimdReal massRatio = chassisMassInv / groundMassInv;
				const hkSimdReal f = massRatio * maxMassRatio;
				groundAccum[ax_it].m_invMasses.mul( f );
			}

			hkVector4 linVel = motion.getLinearVelocity();
			hkVector4 angVel; motion.getAngularVelocity(angVel);

			groundAccumAtIntegration[ax_it] = groundAccum[ax_it];
			axle_params.m_groundObjectAtLastIntegration = &groundAccumAtIntegration[ax_it];
			groundAccumAtIntegration[ax_it].m_angularVel._setRotatedDir( groundAccum[ax_it].getCoreFromWorldMatrix(), angVel );
			groundAccumAtIntegration[ax_it].m_linearVel  = linVel;
		}
	}
}

void hknpVehicleInstance::getExtraTorqueFactor( hkReal deltaTime, hkpVelocityAccumulator& accumulatorForChassis ) const
{
	if ( m_data->m_extraTorqueFactor )
	{
		hkReal f = m_mainSteeringAngleAssumingNoReduction * m_data->m_extraTorqueFactor * deltaTime;
		hkVector4 localDv; localDv.setMul( hkSimdReal::fromFloat(f), m_data->m_chassisOrientation.getColumn<0>() );
		localDv.mul( m_data->m_chassisFrictionInertiaInvDiag );
		accumulatorForChassis.m_angularVel.add(localDv );
	}
}

void hknpVehicleInstance::applyFrictionSolver( const hkStepInfo& stepInfo, hkpVehicleFrictionSolverParams& frictionParams )
{
	hkpVehicleStepInfo vehStepInfo;
	vehStepInfo.m_deltaTime = float(stepInfo.m_deltaTime);
	vehStepInfo.m_invDeltaTime = float(stepInfo.m_invDeltaTime);
	hkVehicleFrictionApplyVehicleFriction( vehStepInfo, m_data->m_frictionDescription, frictionParams, m_frictionStatus );
}



void hknpVehicleInstance::prepareChassisParams( const hkStepInfo& stepInfo, hkpVehicleFrictionSolverParams& frictionParams )
{
	const hkTransform& transform = getChassisTransform();
	const hknpMotion& motion = getChassisMotion();

	buildAccumulator( stepInfo, motion, transform, frictionParams.m_chassis );

	const hkVector4& linVel = motion.getLinearVelocity();
	hkVector4 angVel; motion.getAngularVelocity(angVel);

	// override the inertia matrix in local space
	frictionParams.m_chassis.m_invMasses = m_data->m_chassisFrictionInertiaInvDiag;
	frictionParams.m_chassisAtLastIntegration = frictionParams.m_chassis;
	frictionParams.m_chassisAtLastIntegration.m_angularVel.setRotatedDir( frictionParams.m_chassis.getCoreFromWorldMatrix(), angVel );
	frictionParams.m_chassisAtLastIntegration.m_linearVel = linVel;
}

void hknpVehicleInstance::calcChassisVelocities( hkpVelocityAccumulator& accumulatorForChassis, hknpVehicleJobResults& vehicleResults )
{
	hkQTransform transform; getChassisMotion().getWorldTransform(transform);
	vehicleResults.m_chassisLinearVel = accumulatorForChassis.m_linearVel;
	vehicleResults.m_chassisAngularVel._setRotatedDir( transform.getRotation(), accumulatorForChassis.m_angularVel);
}

void hknpVehicleInstance::calcGroundBodyVelocities( hkReal deltaTime, const hkpVehicleFrictionSolverAxleParams axleParams[], hknpBodyId groundBody[], hknpVehicleJobResults& vehicleResults, hkVector4* originalLinearVels, hkVector4* originalAngularVels )
{
	//
	// Apply the force on the ground objects
	//
	for (int ax_it=0; ax_it<m_data->m_numWheelsPerAxle.getSize(); ax_it++)
	{
		vehicleResults.m_groundBodyPtr[ax_it] = hknpBodyId::invalid();

		const hkpVehicleFrictionSolverAxleParams &axle_params = axleParams[ax_it];
		if ( groundBody[ax_it].isValid() && !_getMotion(m_world,groundBody[ax_it]).isStatic() )
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

void hknpVehicleInstance::applyResultsToWheelInfo( hkReal deltaTime, hknpBodyId groundBodyIds[], const hkpVehicleFrictionSolverParams& frictionParams )
{
	for (int w_it=0; w_it< m_data->m_numWheels; w_it++)
	{
		WheelInfo &wheel_info = m_wheelsInfo[w_it];

		const int axle = m_data->m_wheelParams[w_it].m_axle;

		hkpVehicleFrictionStatus::AxisStatus& astat = m_frictionStatus.m_axis[axle];

		wheel_info.m_skidEnergyDensity = _getMotion(m_world,m_body).isStatic() ?  astat.m_skid_energy_density : 0.0f;
		wheel_info.m_sideForce = astat.m_side_force;
		wheel_info.m_sideSlipVelocity = astat.m_side_slip_velocity;
		wheel_info.m_forwardSlipVelocity = astat.m_forward_slip_velocity;

		//
		// JF: Moved wheel spin angle to here to factor in relative surface vel
		//
		hkVector4 forward_ws;	forward_ws.setRotatedDir( getChassisTransform().getRotation(), m_data->m_chassisOrientation.getColumn<1>() );

		const hkReal chassis_lin_vel = frictionParams.m_chassis.m_linearVel.dot<3>( forward_ws ).getReal();

		if ( ! m_isFixed[w_it] )
		{
			const hkReal virt_wheel_velocity = chassis_lin_vel;

			hkReal surface_relative_wheel_vel = virt_wheel_velocity;

			if( groundBodyIds[axle].isValid() )
			{
				const hkVector4 linVelWs = frictionParams.m_axleParams[axle].m_groundObject->m_linearVel;
				surface_relative_wheel_vel = virt_wheel_velocity - linVelWs.dot<3>( forward_ws ).getReal();
			}

			hkReal spin_velocity = ( surface_relative_wheel_vel ) / (m_data->m_wheelParams[w_it].m_radius);
			wheel_info.m_noSlipIdealSpinVelocity = spin_velocity;

			surface_relative_wheel_vel += wheel_info.m_forwardSlipVelocity;

			spin_velocity = ( surface_relative_wheel_vel ) / (m_data->m_wheelParams[w_it].m_radius);
			wheel_info.m_spinVelocity = spin_velocity;
			const hkReal current_angle = wheel_info.m_spinAngle;
			const hkReal delta_angle = spin_velocity * deltaTime;

			wheel_info.m_spinAngle = current_angle + delta_angle;
		}
		else
		{
			wheel_info.m_spinVelocity = 0.0f;
			wheel_info.m_noSlipIdealSpinVelocity = 0.0f;
		}
	}
}

void hknpVehicleInstance::updateComponents( const hkStepInfo& stepInfo, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, hkArray<hkReal>& suspensionForceAtWheel, hkArray<hkReal>& totalLinearForceAtWheel )
{
	HK_TIMER_BEGIN( "UpdateComponents", HK_NULL );

	hknpVehicleBrake::WheelBreakingOutput wheelBreakingInfo;
	hknpVehicleTransmission::TransmissionOutput transmissionInfo;
	// Allocate space for temporary structure
	transmissionInfo.m_wheelsTransmittedTorque = hkAllocateStack<hkReal>( m_data->m_numWheels );
	{
		hknpVehicleDriverInput::FilteredDriverInputOutput filteredDriverInputInfo;
		{
			const hkReal deltaTime = stepInfo.m_deltaTime;

			updateWheels( deltaTime, cdInfo );
			updateDriverInput( deltaTime, filteredDriverInputInfo );
			updateSteering( deltaTime, filteredDriverInputInfo );
			updateTransmission( deltaTime, transmissionInfo );
			updateEngine( deltaTime, filteredDriverInputInfo, transmissionInfo );
			updateBrake( deltaTime, filteredDriverInputInfo, wheelBreakingInfo );
			updateSuspension( deltaTime, cdInfo, suspensionForceAtWheel );
			updateAerodynamics( deltaTime, aerodynamicsDragInfo );
		}
	}

	// Combine engine & brake torque for each wheel.
	for ( int w_it = 0; w_it < m_data->m_numWheels; ++w_it )
	{
		const hkReal total_torque = wheelBreakingInfo.m_brakingTorque[w_it] + transmissionInfo.m_wheelsTransmittedTorque[w_it];
		totalLinearForceAtWheel[w_it] = total_torque / m_data->m_wheelParams[w_it].m_radius;
	}

	hkDeallocateStack(transmissionInfo.m_wheelsTransmittedTorque, m_data->m_numWheels);

	HK_TIMER_END();
}

void hknpVehicleInstance::applyAerodynamicDrag( const hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, hkpVehicleFrictionSolverParams& frictionParams, const hkReal deltaTime )
{
	const hkSimdReal dt = hkSimdReal::fromFloat(deltaTime);

	hkVector4 impulse;
	impulse.setMul( dt, aerodynamicsDragInfo.m_aerodynamicsForce );

	const hknpMotion& motion = getChassisMotion();
	frictionParams.m_chassis.m_linearVel.addMul( motion.getInverseMass(), impulse );

	impulse.setMul( dt, aerodynamicsDragInfo.m_aerodynamicsTorque );

	hkQTransform transform; motion.getWorldTransform(transform);
	impulse._setRotatedInverseDir( transform.getRotation(), impulse );
	hkVector4 inertiaInvLocal; motion.getInverseInertiaLocal(inertiaInvLocal);
	hkVector4 dangVel; dangVel.setMul( inertiaInvLocal, impulse );
	dangVel._setRotatedDir( transform.getRotation(), dangVel );
	frictionParams.m_chassis.m_angularVel.add(dangVel);
}

void hknpVehicleInstance::applyVelocityDamping( const hkReal deltaTime, hkpVehicleFrictionSolverParams& frictionParams, const hknpVehicleVelocityDamper* damper )
{
	hkVector4 angularVel = frictionParams.m_chassis.m_angularVel;
	const hkReal spinSqrd = angularVel.lengthSquared<3>().getReal();

	const hknpVehicleDefaultVelocityDamper* defaultDamper = static_cast< const hknpVehicleDefaultVelocityDamper* >( damper );

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

void hknpVehicleInstance::applyDampingToAxleAccumulators( const hkStepInfo& stepInfo, hknpBodyId groundBodyIds[], hkpVehicleFrictionSolverParams& frictionParams, hkVector4* originalLinearVels, hkVector4* originalAngularVels )
{
	hkSimdReal zero; zero.setZero();
	const hkSimdReal one = hkSimdReal_1;
	const hkSimdReal deltaTime = hkSimdReal::fromFloat(stepInfo.m_deltaTime);

	if ( groundBodyIds[0].isValid() && !_getMotion(m_world,groundBodyIds[0]).isStatic() )
	{
		const hknpMotionProperties& motionProps = m_world->getMotionPropertiesLibrary()->getEntry(
			_getMotion(m_world,groundBodyIds[0]).m_motionPropertiesId );

		hkSimdReal linearDamping;
		linearDamping.setFromFloat(motionProps.m_linearDamping);
		linearDamping.mul(deltaTime);
		linearDamping.setMax(zero, one - linearDamping);

		hkSimdReal angularDamping;
		angularDamping.setFromFloat(motionProps.m_angularDamping);
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

	if ( groundBodyIds[1].isValid() && !m_world->getBody(groundBodyIds[1]).isStatic() && groundBodyIds[1] != groundBodyIds[0] )
	{
		const hknpMotionProperties& motionProps = m_world->getMotionPropertiesLibrary()->getEntry(
			_getMotion(m_world,groundBodyIds[1]).m_motionPropertiesId );

		hkSimdReal linearDamping;
		linearDamping.setFromFloat(motionProps.m_linearDamping);
		linearDamping.mul(deltaTime);
		linearDamping.setMax(zero, one - linearDamping);

		hkSimdReal angularDamping;
		angularDamping.setFromFloat(motionProps.m_angularDamping);
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

void hknpVehicleInstance::applyDampingToChassisAccumulator( const hkStepInfo& stepInfo, hkpVelocityAccumulator& accumulator, const hknpMotion& motion )
{
	hkSimdReal zero; zero.setZero();
	const hkSimdReal one = hkSimdReal_1;
	const hkSimdReal deltaTime = hkSimdReal::fromFloat(stepInfo.m_deltaTime);

	const hknpMotionProperties& mp = m_world->getMotionPropertiesLibrary()->getEntry( motion.m_motionPropertiesId );

	hkSimdReal linearDamping;
	linearDamping.setFromFloat(mp.m_linearDamping);
	linearDamping.mul(deltaTime);
	linearDamping.setMax(zero, one - linearDamping);

	hkSimdReal angularDamping;
	angularDamping.setFromFloat(mp.m_angularDamping);
	angularDamping.mul(deltaTime);
	angularDamping.setMax(zero, one - angularDamping);

	accumulator.m_linearVel.mul( linearDamping );
	accumulator.m_angularVel.mul( angularDamping );

	accumulator.m_angularVel._setRotatedDir( accumulator.getCoreFromWorldMatrix(), accumulator.m_angularVel );
}

void hknpVehicleInstance::simulateVehicle( const hkStepInfo& stepInfo, const hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, const hkArray<hkReal>& suspensionForceAtWheel, const hkArray<hkReal>& totalLinearForceAtWheel, hknpVehicleJobResults& vehicleResults )
{
	HK_TIMER_BEGIN( "SimulateVehicle", HK_NULL );

	// Vehicles are presumed to have two axles: one with all of the steering wheels, and the other with the rest.
	HK_ASSERT(0x606560c5, m_data->m_numWheelsPerAxle.getSize() == 2);

	// Set up friction params
	hkpVehicleFrictionSolverParams frictionParams;
	for ( int i = 0; i < 2; ++i )
	{
		frictionParams.m_axleParams[i].initialize();
	}
	hknpBodyId groundBody[2] = { hknpBodyId::invalid(), hknpBodyId::invalid() };
	// Use local storage for velocity accumulators to represent the ground body/bodies.
	hkpVelocityAccumulator groundAccum[2];
	hkpVelocityAccumulator groundAccumAtIntegration[2];
	hkVector4 groundOriginalLinearVel[2];
	hkVector4 groundOriginalAngularVel[2];
	hkReal deltaTime = stepInfo.m_deltaTime;

	// Gather friction parameters for each wheel and each axle
	prepareAxleParams( deltaTime, suspensionForceAtWheel, totalLinearForceAtWheel, groundBody, frictionParams, stepInfo, groundAccum, groundAccumAtIntegration );
	prepareChassisParams( stepInfo, frictionParams );

	applyAerodynamicDrag( aerodynamicsDragInfo, frictionParams, deltaTime );

	applyVelocityDamping( deltaTime, frictionParams, m_velocityDamper );

	applySuspensionForces( deltaTime, suspensionForceAtWheel, groundBody, frictionParams, vehicleResults.m_groundBodyImpulses );

	applyDampingToAxleAccumulators( stepInfo, groundBody, frictionParams, groundOriginalLinearVel, groundOriginalAngularVel );

	applyDampingToChassisAccumulator( stepInfo, frictionParams.m_chassis, getChassisMotion() );

	// Fast turn torque
	getExtraTorqueFactor( deltaTime, frictionParams.m_chassis );

	// APPLY THE FRICTION SOLVER
	applyFrictionSolver( stepInfo, frictionParams );

	calcChassisVelocities( frictionParams.m_chassis, vehicleResults );
	calcGroundBodyVelocities( deltaTime, frictionParams.m_axleParams, groundBody, vehicleResults, groundOriginalLinearVel, groundOriginalAngularVel );

	// Update the wheel info for tire marks and spin angle
	applyResultsToWheelInfo( deltaTime, groundBody, frictionParams );

	HK_TIMER_END();
}

void hknpVehicleInstance::applyForcesFromStep( hknpVehicleJobResults& vehicleResults )
{
	HK_TIMER_BEGIN( "ApplyVehicleForces", HK_NULL );

	// NOTE: The proceeding lines are commented out on purpose, due to missing functionality!

	// Apply suspensions impulses per wheel to ground bodies
// 	for ( int w_it = 0; w_it < getNumWheels(); ++w_it )
// 	{
// 		if ( !vehicleResults.m_groundBodyImpulses[w_it].allEqualZero<3>(hkSimdReal::fromFloat(1e-3f)) )
// 		{
// 			// Apply impulses to rigid body (not to motion) such that the entity is activated
// 			m_wheelsInfo[w_it].m_contactBody->applyPointImpulse( vehicleResults.m_groundBodyImpulses[w_it], m_wheelsInfo[w_it].m_hardPointWs );
// 		}
// 	}

	// Set vehicle velocity
	// Velocities are applied to hkpRigidBody such that the vehicle will be activated
	hknpMotion& motion = accessChassisMotion();
	motion.setAngularVelocity( vehicleResults.m_chassisAngularVel );
	motion.setLinearVelocity( vehicleResults.m_chassisLinearVel );

	// Set velocity of ground body per axle
	// Suspension impulses are not applied to these bodies (see above), as the impulses are already applied to the corresponding accumulators
	for ( int ax_it = 0; ax_it < 2; ++ax_it )
	{
		if ( vehicleResults.m_groundBodyPtr[ax_it].isValid() )
		{
			// Set velocities in rigid body (not in rigid motion) such that the entity is activated
			// vehicleResults.m_groundBodyPtr[ax_it]->setAngularVelocity( vehicleResults.m_groundBodyAngularVel[ax_it] );
			// vehicleResults.m_groundBodyPtr[ax_it]->setLinearVelocity( vehicleResults.m_groundBodyLinearVel[ax_it] );
		}
	}

	HK_TIMER_END();
}

void hknpVehicleInstance::stepVehicleUsingWheelCollideOutput( const hkStepInfo& stepInfo, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hknpVehicleJobResults& vehicleResults )
{
	HK_TIMER_BEGIN("DoVehicle", this);

	hknpVehicleAerodynamics::AerodynamicsDragOutput aerodynamicsDragInfo;
	hkInplaceArray<hkReal,s_maxNumLocalWheels> suspensionForceAtWheel( m_data->m_numWheels );
	hkInplaceArray<hkReal,s_maxNumLocalWheels> totalLinearForceAtWheel( m_data->m_numWheels );

	updateComponents( stepInfo, cdInfo, aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel );

	simulateVehicle( stepInfo, aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel, vehicleResults );

	HK_TIMER_END();
}

// Apply action method
hknpAction::ApplyActionResult hknpVehicleInstance::applyAction( const hknpSimulationThreadContext& tl, const hknpSolverInfo& solverInfo, hknpCdPairWriter* HK_RESTRICT pairWriter )
{
	hkStepInfo stepInfo;
	convertToStepInfo( solverInfo, stepInfo);

	stepVehicle( stepInfo );

	return hknpAction::RESULT_OK;
}

void hknpVehicleInstance::stepVehicle( const hkStepInfo& stepInfo )
{
	// Update the wheels' hard points.
	updateBeforeCollisionDetection();

	// Allocate space for temporary structure
	hkLocalArray<hknpVehicleWheelCollide::CollisionDetectionWheelOutput> cdInfo( m_data->m_numWheels );
	m_wheelCollide->collideWheels( stepInfo.m_deltaTime, this, cdInfo.begin(), m_world );

	hknpVehicleJobResults vehicleResults;
	stepVehicleUsingWheelCollideOutput( stepInfo, cdInfo.begin(), vehicleResults );
	applyForcesFromStep( vehicleResults );
}


// Calculate the current position and rotation of a wheel for the graphics engine
void hknpVehicleInstance::calcCurrentPositionAndRotation( const hkTransform& chassisTransform, const hknpVehicleSuspension* suspension, int wheelNo, hkVector4& posOut, hkQuaternion& rotOut )
{
	WheelInfo& wi = m_wheelsInfo[wheelNo];

	//
	//	concatenate the matrices for the wheels: todo: move to update graphics, allows for LOD levels
	//
	{
		hkQuaternion forward_orientation_cs;
		{
			hkRotation& chassisOrientation = m_data->m_chassisOrientation;
			hkRotation forward_rotation_cs; forward_rotation_cs.setCols( chassisOrientation.getColumn<1>(), chassisOrientation.getColumn<0>(), chassisOrientation.getColumn<2>() );
			forward_orientation_cs.set( forward_rotation_cs );
		}

		const hkReal spin_angle = wi.m_spinAngle;
		hkVector4 adjusted_spin_axis_chassis_space;
		adjusted_spin_axis_chassis_space.setRotatedInverseDir( forward_orientation_cs, wi.m_spinAxisChassisSpace );
		hkQuaternion spin_rotation; spin_rotation.setAxisAngle( adjusted_spin_axis_chassis_space, -spin_angle);

		hkQuaternion chassis_orientation; chassis_orientation.set(chassisTransform.getRotation());
		hkQuaternion tmp;
		tmp.setMul( chassis_orientation, wi.m_steeringOrientationChassisSpace );
		tmp.mul( forward_orientation_cs );
		rotOut.setMul(tmp,spin_rotation);
	}


	const hkReal suspLen = hkMath::max2( hkReal(0.0f), wi.m_currentSuspensionLength );
	posOut._setTransformedPos( chassisTransform, suspension->m_wheelParams[wheelNo].m_hardpointChassisSpace );
	posOut.addMul( hkSimdReal::fromFloat(suspLen), wi.m_suspensionDirectionWs );
}


hkReal hknpVehicleInstance::calcRPM() const
{
	return m_rpm;
}


// Two speed calculations for the camera
hkReal hknpVehicleInstance::calcKMPH() const
{
	const hkReal vel_ms = getChassisMotion().getLinearVelocity().length<3>().getReal();
	const hkReal vel_kmh = vel_ms / 1000.0f * 3600.0f;

	return vel_kmh;
}

hkReal hknpVehicleInstance::calcMPH() const
{
	const hkReal vel_ms = getChassisMotion().getLinearVelocity().length<3>().getReal();
	const hkReal vel_mph = vel_ms / 1609.3f * 3600.0f;

	return vel_mph;
}

void hknpVehicleInstance::addToWorld( hknpWorld* world )
{
	world->addAction(this);
	m_wheelCollide->addToWorld( world );
}

void hknpVehicleInstance::removeFromWorld()
{
	m_wheelCollide->removeFromWorld();
	m_world->removeAction(this);
}

void hknpVehicleInstance::onShiftWorld( hkVector4Parameter offset )
{
	HK_ERROR(0x6d8e82c9,"Not Supported");	// not implemented
}

void hknpVehicleInstance::buildAccumulator(const hkStepInfo& info, const hknpMotion& motion, const hkTransform& transform, hkpVelocityAccumulator& accumulator )
{
	accumulator.m_type		= hkpVelocityAccumulator::HK_RIGID_BODY;
	accumulator.m_context	= hkpVelocityAccumulator::ACCUMULATOR_CONTEXT_BUILD_JACOBIANS;

	hkMatrix3 coreFromWorldTransform;
	coreFromWorldTransform._setTranspose(transform.getRotation());
	accumulator.setCoreFromWorldMatrix(coreFromWorldTransform);
	motion.getInverseInertiaLocal(accumulator.m_invMasses);
	accumulator.m_linearVel = motion.getLinearVelocity();
	accumulator.setCenterOfMassInWorld( motion.getCenterOfMassInWorld() );
	motion.getAngularVelocity( accumulator.m_angularVel );

	accumulator.m_gravityFactor = 1.0f;
	accumulator.m_deactivationClass = 0;
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
