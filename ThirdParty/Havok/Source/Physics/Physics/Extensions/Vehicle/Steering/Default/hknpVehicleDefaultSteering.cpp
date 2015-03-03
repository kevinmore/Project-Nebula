/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/Steering/Default/hknpVehicleDefaultSteering.h>

hknpVehicleDefaultSteering::hknpVehicleDefaultSteering()
{
	m_maxSteeringAngle = 0;
	m_maxSpeedFullSteeringAngle = 0;
	// m_doesWheelSteer
}

void hknpVehicleDefaultSteering::calcMainSteeringAngle( const hkReal deltaTime, const hknpVehicleInstance* vehicle, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput, SteeringAnglesOutput& steeringOutput )
{
	const hkReal input_value = filteredInfoOutput.m_steeringWheelInput;

	steeringOutput.m_mainSteeringAngle = input_value * m_maxSteeringAngle;
	steeringOutput.m_mainSteeringAngleAssumingNoReduction = steeringOutput.m_mainSteeringAngle;

	// Calculate the velocity of the car.
	const hkTransform &car_transform = vehicle->getChassisTransform();
	const hkVector4& forward_cs = vehicle->m_data->m_chassisOrientation.getColumn<1>();

	//const hkVector4 forward_ws = car_transform.getTransformedDir(forward_cs);
	hkVector4 forward_ws;
	forward_ws.setRotatedDir(car_transform.getRotation(),forward_cs);

	const hkSimdReal chassis_lin_vel = vehicle->getChassisMotion().getLinearVelocity().dot<3>(forward_ws);

	const hkSimdReal fullAngle = hkSimdReal::fromFloat(m_maxSpeedFullSteeringAngle);
	if (chassis_lin_vel > fullAngle)
	{
		// Clip steering angle.
		const hkSimdReal s_factor = fullAngle / chassis_lin_vel;
		steeringOutput.m_mainSteeringAngle = steeringOutput.m_mainSteeringAngle  * (s_factor * s_factor).getReal();
	}
}

void hknpVehicleDefaultSteering::calcSteering( const hkReal deltaTime, const hknpVehicleInstance* vehicle, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput, SteeringAnglesOutput& steeringOutput )
{
	// Calculate main steering angle
	calcMainSteeringAngle( deltaTime, vehicle, filteredInfoOutput, steeringOutput );

	// Set to 0 for wheels that do not steer
	for (int w_it = 0; w_it < m_doesWheelSteer.getSize(); w_it++)
	{
		if (m_doesWheelSteer[w_it])
		{
			steeringOutput.m_wheelsSteeringAngle[w_it] = steeringOutput.m_mainSteeringAngle;
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
