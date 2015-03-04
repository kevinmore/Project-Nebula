/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>

#include <Physics2012/Vehicle/hkpVehicleInstance.h>
#include <Physics2012/Vehicle/Steering/Ackerman/hkpVehicleSteeringAckerman.h>

hkpVehicleSteeringAckerman::hkpVehicleSteeringAckerman()
{
	m_maxSteeringAngle = 0;
	m_maxSpeedFullSteeringAngle = 0;
	m_wheelBaseLength = 2.4f;
	m_trackWidth = 2.2f;
}

void hkpVehicleSteeringAckerman::calcMainSteeringAngle( const hkReal deltaTime, const hkpVehicleInstance* vehicle, const hkpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput, SteeringAnglesOutput& steeringOutput )
{
	const hkReal input_value = filteredInfoOutput.m_steeringWheelInput;

	steeringOutput.m_mainSteeringAngle = input_value * m_maxSteeringAngle;
	steeringOutput.m_mainSteeringAngleAssumingNoReduction = steeringOutput.m_mainSteeringAngle;

	// Calculate the velocity of the car.
	const hkTransform &car_transform = vehicle->getChassis()->getTransform();
	const hkVector4& forward_cs = vehicle->m_data->m_chassisOrientation.getColumn<1>();

	//const hkVector4 forward_ws = car_transform.getTransformedDir(forward_cs);
	hkVector4 forward_ws;
	forward_ws.setRotatedDir(car_transform.getRotation(),forward_cs);

	const hkSimdReal chassis_lin_vel = vehicle->getChassis()->getLinearVelocity().dot<3>(forward_ws);

	const hkSimdReal fullAngle = hkSimdReal::fromFloat(m_maxSpeedFullSteeringAngle);
	if (chassis_lin_vel > fullAngle)
	{
		// Clip steering angle.
		const hkSimdReal s_factor = fullAngle / chassis_lin_vel;
		steeringOutput.m_mainSteeringAngle = steeringOutput.m_mainSteeringAngle  * (s_factor * s_factor).getReal();
	}
}

void hkpVehicleSteeringAckerman::calcSteering( const hkReal deltaTime, const hkpVehicleInstance* vehicle, const hkpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput, SteeringAnglesOutput& steeringOutput )
{
	// Calculate main steering angle
	calcMainSteeringAngle( deltaTime, vehicle, filteredInfoOutput, steeringOutput );

	calcAckermanOuterAngles( vehicle, steeringOutput );	
}

void hkpVehicleSteeringAckerman::calcAckermanOuterAngles( const hkpVehicleInstance* vehicle, SteeringAnglesOutput& steeringOutput )
{
	HK_ASSERT2(0x7708674b,  m_trackWidth > HK_REAL_EPSILON && m_wheelBaseLength > HK_REAL_EPSILON, "Track width and wheel base length have to be > 0" );

	hkpMotion* motion = vehicle->getChassis()->getMotion();
	hkVector4  chassisCoM  = motion->m_motionState.getSweptTransform().m_centerOfMass1;  
	hkVector4 sideDirection= motion->getTransform().getColumn(2); // right dir, used for dot product to detect where's the wheel wrt chassis

	const hkReal trackLengthRatio = m_trackWidth / m_wheelBaseLength;
	// outer wheel has to preserve Ackerman condition (cot(outer)-cot(inner)=w/l) => outer = cot^-1( (w/l) + cot(inner) ) =>...
	// ... => outer = arctan[ 1 / ( (w/l) + 1/tan(inner) ) ]
	hkReal angle = hkMath::fabs(steeringOutput.m_mainSteeringAngle);
	hkReal tanAngle = hkMath::sin(angle) / hkMath::cos(angle);
	hkReal outerAngle;
	if (hkMath::fabs(tanAngle) < HK_REAL_EPSILON)
	{
		outerAngle = 0.0f;
	}
	else
	{
		hkReal y = 1.0f;
		hkReal x = trackLengthRatio + 1.0f / tanAngle;
		outerAngle = hkMath::atan2( y, x );
	}

	// taking care of sign depending on turning direction
	if ( steeringOutput.m_mainSteeringAngle < 0.0f )
		outerAngle = -outerAngle;

	// Set to 0 for wheels that do not steer 
	for (int w_it = 0; w_it < m_doesWheelSteer.getSize(); w_it++) 
	{
		if (m_doesWheelSteer[w_it])
		{
			// convert wheel position to chassis frame. 
			hkVector4 CenterOfMassToHardPoint; CenterOfMassToHardPoint.setSub( vehicle->m_wheelsInfo[w_it].m_hardPointWs, chassisCoM );

			// outer wheel when (turning_left && wheel_on_right || turning_right && wheel_on_left)
			hkBool isOuterWheel = ( steeringOutput.m_mainSteeringAngle * CenterOfMassToHardPoint.dot<3>(sideDirection).getReal() ) < HK_REAL_EPSILON; 
			if ( isOuterWheel )
			{
				steeringOutput.m_wheelsSteeringAngle[w_it] = outerAngle;
			}
			else
			{
				// inner wheel rotates with already computed steering angle
				steeringOutput.m_wheelsSteeringAngle[w_it] = steeringOutput.m_mainSteeringAngle;
			}

		}
		else
		{
			steeringOutput.m_wheelsSteeringAngle[w_it] = 0.0f;
		}
	}
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
