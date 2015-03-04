/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/DriverInput/Default/hknpVehicleDefaultAnalogDriverInput.h>

hknpVehicleDefaultAnalogDriverInput::hknpVehicleDefaultAnalogDriverInput()
{
	// deviceStatus
	m_slopeChangePointX = 0;
	m_initialSlope = 0;
	m_deadZone = 0;
	m_autoReverse = false;
}

hknpVehicleDriverInputStatus* hknpVehicleDriverInputAnalogStatus::clone() const
{
	hknpVehicleDriverInputAnalogStatus* r = new hknpVehicleDriverInputAnalogStatus;
	*r = *this;
	return r;
}

//void hknpVehicleDefaultAnalogDriverInput::init( const hknpVehicleInstance* vehicle )
//{
//	// The vehicle stores all cached data needed.
//}


hkReal hknpVehicleDefaultAnalogDriverInput::calcAcceleratorInput( const hkReal deltaTime ,const hknpVehicleInstance* vehicle, const hknpVehicleDriverInputAnalogStatus* deviceStatus, FilteredDriverInputOutput& input) const
{
	const hkBool currently_reversing = vehicle->m_isReversing;

	const hkReal device_input_y =
		(currently_reversing && m_autoReverse) ? - deviceStatus->m_positionY : deviceStatus->m_positionY ;

	if ( !m_autoReverse && currently_reversing)
	{
		return 1.0f;
	}

	if (device_input_y>0) return 0.0f;

	const hkReal acc_pedal = -1.0f * device_input_y;

	return acc_pedal;
}

hkReal hknpVehicleDefaultAnalogDriverInput::calcBrakeInput( const hkReal deltaTime,
																const hknpVehicleInstance* vehicle,
																const hknpVehicleDriverInputAnalogStatus* deviceStatus,
																FilteredDriverInputOutput& FilteredDriverInputOutputOut ) const
{
	const hkBool currently_reversing = vehicle->m_isReversing;

	const hkReal device_input_y =
		(currently_reversing && m_autoReverse) ? - deviceStatus->m_positionY : deviceStatus->m_positionY ;

	if (device_input_y<0) return 0.0f;

	const hkReal brake_pedal = device_input_y;

	return brake_pedal;
}

hkReal hknpVehicleDefaultAnalogDriverInput::calcSteeringInput(	const hkReal deltaTime,
																	const hknpVehicleInstance* vehicle,
																	const hknpVehicleDriverInputAnalogStatus* deviceStatus,
																	FilteredDriverInputOutput& FilteredDriverInputOutputOut )	const
{
	const hkReal joystick_input = deviceStatus->m_positionX;

    const hkReal abs_joystick_input = hkMath::fabs(joystick_input);

	// Dead Zone
	if (abs_joystick_input < m_deadZone)
	{
		return 0.0f;
	}

	// else First Slope

	const hkReal input_sign = (joystick_input > 0.0f) ? 1.0f : -1.0f;

    if(abs_joystick_input < m_slopeChangePointX)
	{
		const hkReal abs_steering = ( abs_joystick_input - m_deadZone) * (m_initialSlope);

		const hkReal steering = input_sign * abs_steering;

		return steering;
    }

	// else Second Slope
	{
		// These two values could be cached
		const hkReal slopeChangePointY = ( m_slopeChangePointX - m_deadZone) * m_initialSlope;
		const hkReal secondSlope       = (1.0f - slopeChangePointY ) / ( (1.0f - m_deadZone) - (m_slopeChangePointX - m_deadZone) );

		const hkReal abs_steering = slopeChangePointY + secondSlope * (abs_joystick_input - m_slopeChangePointX);

		const hkReal steering = input_sign * abs_steering;

		return steering;
	}
}

hkBool hknpVehicleDefaultAnalogDriverInput::calcTryingToReverse(	const hkReal deltaTime,
																		const hknpVehicleInstance* vehicle,
																		const hknpVehicleDriverInputAnalogStatus* deviceStatus,
																		FilteredDriverInputOutput& FilteredDriverInputOutputOut)
{
	// If we are not in autoreverse mode, then reverse only happens when the driver hits the Reverse Button
	if (!m_autoReverse)
	{
		return deviceStatus->m_reverseButtonPressed;
	}

	hkVector4 vec_vel = vehicle->getChassisMotion().getLinearVelocity();
	const hkSimdReal lin_vel = vec_vel.length<3>();//.fastLength();
	const bool currently_reversing = vehicle->m_isReversing;
	const bool not_moving = (lin_vel < hkSimdReal::fromFloat(5.0f / 3.6f)); // Almost stopped (move float to BP)
	const bool brake_pressed = (deviceStatus->m_positionY > 0.1f);

	// If the break is pressed and we are moving
	if( (brake_pressed) && (lin_vel > hkSimdReal_Eps) )
	{
		// Check if the car is going backwards
		hkVector4 for_dir_ls = vehicle->m_data->m_chassisOrientation.getColumn<1>();

		hkTransform chassis_trans = vehicle->getChassisTransform();
		hkVector4 for_dir_ws;
		for_dir_ws._setRotatedDir(chassis_trans.getRotation(), for_dir_ls);

		vec_vel.normalize<3>();
		for_dir_ws.normalize<3>();

		hkSimdReal val = vec_vel.dot<3>(for_dir_ws);

		// If the car is going backwards then we should be in reversing.
		if(val < hkSimdReal_Eps)
		{
			return true;
		}
	}

	// If we are reversing and moving we keep reversing
	// If we are reversing but stopped we keep reversing if the brake is pressed
	// If we are reversing, stopped and no brake is pressed, we don't reverse any more
	if (currently_reversing)
	{
		if (not_moving)
		{
			return brake_pressed;
		}

		// comment out the following else block if you want proper reversing on moving platforms
		else
		{
			return true;
		}
	}

	// If we are not reversing, we start reversing only if we are stopped and pressing the brake
	if (not_moving && brake_pressed)
		return true;

	return false;
}

void hknpVehicleDefaultAnalogDriverInput::calcDriverInput( const hkReal deltaTime, const hknpVehicleInstance* vehicle, const hknpVehicleDriverInputStatus* deviceStatus, FilteredDriverInputOutput& filteredInputOut )
{
	const hknpVehicleDriverInputAnalogStatus* analogDeviceStatus = (const hknpVehicleDriverInputAnalogStatus*)(deviceStatus);
	filteredInputOut.m_acceleratorPedalInput = calcAcceleratorInput( deltaTime, vehicle, analogDeviceStatus, filteredInputOut );
	filteredInputOut.m_brakePedalInput = calcBrakeInput( deltaTime, vehicle, analogDeviceStatus, filteredInputOut );
	filteredInputOut.m_steeringWheelInput = calcSteeringInput( deltaTime, vehicle, analogDeviceStatus, filteredInputOut );
	filteredInputOut.m_handbrakeOn = analogDeviceStatus->m_handbrakeButtonPressed;
	filteredInputOut.m_tryingToReverse = calcTryingToReverse( deltaTime, vehicle, analogDeviceStatus, filteredInputOut );
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
