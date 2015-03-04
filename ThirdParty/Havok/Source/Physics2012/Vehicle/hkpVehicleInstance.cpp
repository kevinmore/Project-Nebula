/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/hkpVehicleInstance.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpBoxMotion.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Motion/Rigid/hkpFixedRigidMotion.h>

#include <Physics2012/Vehicle/Brake/hkpVehicleBrake.h>
#include <Physics2012/Vehicle/Engine/hkpVehicleEngine.h>
#include <Physics2012/Vehicle/TyreMarks/hkpTyremarksInfo.h>
#include <Physics2012/Vehicle/Steering/hkpVehicleSteering.h>
#include <Physics2012/Vehicle/Suspension/hkpVehicleSuspension.h>
#include <Physics2012/Vehicle/DriverInput/hkpVehicleDriverInput.h>
#include <Physics2012/Vehicle/AeroDynamics/hkpVehicleAerodynamics.h>
#include <Physics2012/Vehicle/Transmission/hkpVehicleTransmission.h>
#include <Physics2012/Vehicle/WheelCollide/hkpVehicleWheelCollide.h>
#include <Physics2012/Vehicle/VelocityDamper/hkpVehicleVelocityDamper.h>

#include <Physics/ConstraintSolver/VehicleFriction/hkpVehicleFrictionSolver.h>

#include <Physics2012/Dynamics/Phantom/hkpPhantom.h>
#include <Physics2012/Vehicle/VelocityDamper/Default/hkpVehicleDefaultVelocityDamper.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>
#include <Physics2012/Vehicle/Simulation/Default/hkpVehicleDefaultSimulation.h>

HK_COMPILE_TIME_ASSERT( (int)hkpVehicleInstance::WheelInfo::MAX_NUM_SHAPE_KEYS == (int)hkpVehicleWheelCollide::CollisionDetectionWheelOutput::MAX_NUM_SHAPE_KEYS );

hkpVehicleInstance::hkpVehicleInstance( hkpRigidBody* chassis ) :
	hkpUnaryAction(chassis)
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
	m_vehicleSimulation = HK_NULL;
}

void hkpVehicleInstance::setChassis ( hkpRigidBody* chassis )
{
	setEntity(chassis);
}

void hkpVehicleInstance::WheelInfo::init()
{
	hkVector4 up; up = hkVector4::getConstant<HK_QUADREAL_0100>();
	m_contactPoint.set( hkVector4::getZero(), up, 1);

	m_contactFriction = 0.0f;
	m_contactBody = HK_NULL;
	m_contactShapeKey[0] = hkpShapeKey(-1);
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
hkpVehicleInstance::~hkpVehicleInstance()
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
	REMOVE_REFERENCE( m_vehicleSimulation );

	// Tyremarks are an optional component
	REMOVE_REFERENCE( m_tyreMarks );
}


void hkpVehicleInstance::init()
{
	// The initialization of the data modifies the chassis
	if (!m_data->m_alreadyInitialised)
	{
		m_data->init( m_suspension->m_wheelParams, getChassis());
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

		hkVector4 diagonal; diagonal.setMul( hkSimdReal::fromFloat(getChassis()->getMass()), unitDiagonal);

		hkMatrix3 matrix;
		hkMatrix3Util::_setDiagonal(diagonal, matrix);
		getChassis()->setInertiaLocal(matrix);

		// check that we have the correct inertia type for our chassis
		HK_ASSERT2(0x5c819b4e, (getChassis()->getMotionType() == hkpMotion::MOTION_BOX_INERTIA) || (getChassis()->getMotionType() == hkpMotion::MOTION_THIN_BOX_INERTIA), "Vehicle chassis MUST be of type hkpRigidBodyCinfo::MOTION_BOX_INERTIA or hkpRigidBodyCinfo::MOTION_THIN_BOX_INERTIA");
	}

	m_wheelsInfo.setSize( m_data->m_numWheels );
	{ for (int i=0;i<m_wheelsInfo.getSize();i++) { m_wheelsInfo[i].init(); } }


	m_isFixed.setSize( m_data->m_numWheels );
	{ for (int i=0;i<m_isFixed.getSize();i++) { m_isFixed[i] = false; } }
	m_wheelsTimeSinceMaxPedalInput = 0.0f;

	m_mainSteeringAngle = 0.0f;
	m_mainSteeringAngleAssumingNoReduction = 0.0f;
	m_wheelsSteeringAngle.setSize( m_data->m_numWheels );
	{ for (int i=0;i<m_wheelsSteeringAngle.getSize();i++) { m_wheelsSteeringAngle[i] = 0.0f; } }

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

	// backward compatibility with default simulation behavior
	HK_WARN_ON_DEBUG_IF( m_vehicleSimulation==HK_NULL, 0x168a27ba, "Warning: No hkpVehicleSimulation object defined. Using default for backward compatibility.\n");
	if ( m_vehicleSimulation == HK_NULL )
	{
		m_vehicleSimulation = new hkpVehicleDefaultSimulation;
	}
	m_vehicleSimulation->init( this );
}


void hkpVehicleInstance::getPhantoms( hkArray<hkpPhantom*>& phantomsOut )
{
	m_wheelCollide->getPhantoms( phantomsOut );
}



void hkpVehicleInstance::handleFixedGroundAccum( hkpRigidBody* ground, hkpVelocityAccumulator& accum )
{
	accum.setFixed();
}


void hkpVehicleInstance::updateBeforeCollisionDetection()
{
	HK_ASSERT2( 0x155bffe2, getChassis()->isAddedToWorld(), "Vehicle chassis is not added to world.");

	HK_TIMER_BEGIN("UpdateBeforeCD", HK_NULL);

	const hkTransform& car_transform = getChassis()->getTransform();

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

	// The wheel collide object should update its phantom.
	m_wheelCollide->updateBeforeCollisionDetection( this );

	HK_TIMER_END();
}

void hkpVehicleInstance::updateWheels( hkReal deltaTime, const hkpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo )
{
	const hkTransform& car_transform = getChassis()->getTransform();

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
			m_wheelsInfo[w_it].m_contactBody = cdInfo[w_it].m_contactBody;
			for (int i =0; i < hkpVehicleWheelCollide::CollisionDetectionWheelOutput::MAX_NUM_SHAPE_KEYS;i++)
			{
				m_wheelsInfo[w_it].m_contactShapeKey[i] = cdInfo[w_it].m_contactShapeKey[i];
			}
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


void hkpVehicleInstance::updateDriverInput( hkReal deltaTime, hkpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo )
{
	filteredDriverInputInfo.m_tryingToReverse = m_tryingToReverse;
	m_driverInput->calcDriverInput( deltaTime, this, m_deviceStatus, filteredDriverInputInfo );
	m_tryingToReverse = filteredDriverInputInfo.m_tryingToReverse;
}


void hkpVehicleInstance::updateSteering( hkReal deltaTime, const hkpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo )
{
	hkpVehicleSteering::SteeringAnglesOutput steeringInfo;
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


void hkpVehicleInstance::updateTransmission( hkReal deltaTime, hkpVehicleTransmission::TransmissionOutput& transmissionInfo )
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


void hkpVehicleInstance::updateEngine( hkReal deltaTime, const hkpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo, const hkpVehicleTransmission::TransmissionOutput& transmissionInfo )
{
	hkpVehicleEngine::EngineOutput engineInfo;
	engineInfo.m_rpm = m_rpm;
	engineInfo.m_torque = m_torque;
	m_engine->calcEngineInfo( deltaTime, this, filteredDriverInputInfo, transmissionInfo, engineInfo );
	m_rpm = engineInfo.m_rpm;
	m_torque = engineInfo.m_torque;
}


void hkpVehicleInstance::updateBrake( hkReal deltaTime, const hkpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo, hkpVehicleBrake::WheelBreakingOutput& wheelBreakingInfo )
{
	wheelBreakingInfo.m_isFixed.setSize( m_isFixed.getSize() );
	wheelBreakingInfo.m_brakingTorque.setSize( m_isFixed.getSize() );
	{
		for (int i = 0 ; i < m_isFixed.getSize() ; i++ )
		{
			wheelBreakingInfo.m_isFixed[i] = m_isFixed[i];
		}
	}
	wheelBreakingInfo.m_wheelsTimeSinceMaxPedalInput = m_wheelsTimeSinceMaxPedalInput;

	m_brake->calcBreakingInfo( deltaTime, this, filteredDriverInputInfo, wheelBreakingInfo );

	// copy back the brake cache data
	{
		for (int i = 0 ; i < wheelBreakingInfo.m_isFixed.getSize() ; i++ )
		{
			m_isFixed[i] = wheelBreakingInfo.m_isFixed[i];
		}
		m_wheelsTimeSinceMaxPedalInput = wheelBreakingInfo.m_wheelsTimeSinceMaxPedalInput;
	}
}


void hkpVehicleInstance::updateSuspension( hkReal deltaTime, const hkpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hkArray<hkReal>& suspensionForces )
{
	m_suspension->calcSuspension( deltaTime, this, cdInfo, suspensionForces.begin() );
}


void hkpVehicleInstance::updateAerodynamics( hkReal deltaTime, hkpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo )
{
	m_aerodynamics->calcAerodynamics( deltaTime, this, aerodynamicsDragInfo );
}

void hkpVehicleInstance::updateComponents( const hkStepInfo& stepInfo, const hkpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hkpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, hkArray<hkReal>& suspensionForceAtWheel, hkArray<hkReal>& totalLinearForceAtWheel )
{
	HK_TIMER_BEGIN( "UpdateComponents", HK_NULL );
	
	hkpVehicleBrake::WheelBreakingOutput wheelBreakingInfo;
	hkpVehicleTransmission::TransmissionOutput transmissionInfo;
	// Allocate space for temporary structure
	transmissionInfo.m_wheelsTransmittedTorque = hkAllocateStack<hkReal>( m_data->m_numWheels );
	{
		hkpVehicleDriverInput::FilteredDriverInputOutput filteredDriverInputInfo;
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

void hkpVehicleInstance::simulateVehicle( const hkStepInfo& stepInfo, const hkpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, const hkArray<hkReal>& suspensionForceAtWheel, const hkArray<hkReal>& totalLinearForceAtWheel, hkpVehicleJobResults* vehicleJobResults/*=HK_NULL*/ )
{
	HK_ASSERT2( 0x155bffe2, getChassis()->isAddedToWorld(), "Vehicle chassis is not added to world.");
	HK_ASSERT2( 0x155bffe3, m_vehicleSimulation, "Vehicle simulation object not defined." );
	HK_TIMER_BEGIN( "SimulateVehicle", HK_NULL );

	hkpVehicleSimulation::SimulationInput simulationInput( aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel );
	m_vehicleSimulation->simulateVehicle(this,stepInfo, simulationInput, vehicleJobResults);

	HK_TIMER_END();
}

void hkpVehicleInstance::stepVehicleUsingWheelCollideOutput( const hkStepInfo& stepInfo, const hkpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo )
{
	HK_ASSERT2( 0x155bffe2, getChassis()->isAddedToWorld(), "Vehicle chassis is not added to world.");

	HK_TIMER_BEGIN("DoVehicle", this);

	hkpVehicleAerodynamics::AerodynamicsDragOutput aerodynamicsDragInfo;
	hkInplaceArray<hkReal,s_maxNumLocalWheels> suspensionForceAtWheel( m_data->m_numWheels ); 
	hkInplaceArray<hkReal,s_maxNumLocalWheels> totalLinearForceAtWheel( m_data->m_numWheels );

	updateComponents( stepInfo, cdInfo, aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel );

	simulateVehicle( stepInfo, aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel );

	HK_TIMER_END();
}

// Apply action method
void hkpVehicleInstance::applyAction( const hkStepInfo& stepInfo )
{
	stepVehicle( stepInfo );
}

void hkpVehicleInstance::stepVehicle( const hkStepInfo& stepInfo )
{
	// Update the wheels' hard points.
	updateBeforeCollisionDetection();

	// Allocate space for temporary structure
	hkLocalArray<hkpVehicleWheelCollide::CollisionDetectionWheelOutput> cdInfo( m_data->m_numWheels );
	m_wheelCollide->collideWheels( stepInfo.m_deltaTime, this, cdInfo.begin() );

	stepVehicleUsingWheelCollideOutput( stepInfo, cdInfo.begin() );
}


// Calculate the current position and rotation of a wheel for the graphics engine
void hkpVehicleInstance::calcCurrentPositionAndRotation( const hkpRigidBody* chassis, const hkpVehicleSuspension* suspension, int wheelNo, hkVector4& posOut, hkQuaternion& rotOut )
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

		const hkQuaternion& chassis_orientation = chassis->getRotation();
		hkQuaternion tmp;
		tmp.setMul( chassis_orientation, wi.m_steeringOrientationChassisSpace );
		tmp.mul( forward_orientation_cs );
		rotOut.setMul(tmp,spin_rotation);
	}


	const hkReal suspLen = hkMath::max2( hkReal(0.0f), wi.m_currentSuspensionLength );
	posOut._setTransformedPos( chassis->getTransform(), suspension->m_wheelParams[wheelNo].m_hardpointChassisSpace );
	posOut.addMul( hkSimdReal::fromFloat(suspLen), wi.m_suspensionDirectionWs );
}


hkReal hkpVehicleInstance::calcRPM() const
{
	return m_rpm;
}


// Two speed calculations for the camera
hkReal hkpVehicleInstance::calcKMPH() const
{
	const hkReal vel_ms = getChassis()->getLinearVelocity().length<3>().getReal();
	const hkReal vel_kmh = vel_ms / 1000.0f * 3600.0f;

	return vel_kmh;
}

hkReal hkpVehicleInstance::calcMPH() const
{
	const hkReal vel_ms = getChassis()->getLinearVelocity().length<3>().getReal();
	const hkReal vel_mph = vel_ms / 1609.3f * 3600.0f;

	return vel_mph;
}


hkpAction* hkpVehicleInstance::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	hkpRigidBody* newChassis = (hkpRigidBody*) newEntities[0];
	hkpVehicleInstance* newV = new hkpVehicleInstance( newChassis );
	*newV = *this;

	newV->setWorld(HK_NULL);
	newV->setSimulationIsland(HK_NULL);

	// referenced already in constructor above
	newV->m_entity = newChassis;

	// Wheel collide can't be shared (phantoms etc based on the chassis)
	newV->m_wheelCollide = m_wheelCollide->clone( *newChassis, newPhantoms );
	newV->m_wheelCollide->m_alreadyUsed = true;

	// Simulation can't be shared
	newV->m_vehicleSimulation = m_vehicleSimulation->clone( newV );

	// Clone the device status so the vehicles can move individually.
	newV->m_deviceStatus = this->m_deviceStatus->clone();

	// All the rest can be shared
	newV->m_data->addReference();
	newV->m_driverInput->addReference();
	newV->m_steering->addReference();
	newV->m_engine->addReference();
	newV->m_transmission->addReference();
	newV->m_brake->addReference();
	newV->m_suspension->addReference();
	newV->m_aerodynamics->addReference();
	newV->m_velocityDamper->addReference();

	// Tyremarks are an optional componant
	if(m_tyreMarks != HK_NULL)
	{
		newV->m_tyreMarks->addReference();
	}

	return newV;
}


void hkpVehicleInstance::addToWorld( hkpWorld* world )
{
	world->addEntity( getChassis() );

	m_wheelCollide->addToWorld( world );
	m_vehicleSimulation->addToWorld( world );
}


void hkpVehicleInstance::removeFromWorld()
{
	m_vehicleSimulation->removeFromWorld();
	m_wheelCollide->removeFromWorld();

	HK_ASSERT2( 0xa45edda8, getChassis()->isAddedToWorld(), "The vehicle is not added to the world.");
	hkpWorld* world = getChassis()->getWorld();
	world->removeEntity( getChassis() );
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
