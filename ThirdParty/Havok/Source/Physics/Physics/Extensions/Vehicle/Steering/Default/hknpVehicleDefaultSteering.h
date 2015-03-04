/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_DEFAULT_STEERING_H
#define HKNP_VEHICLE_DEFAULT_STEERING_H

#include <Physics/Physics/Extensions/Vehicle/Steering/hknpVehicleSteering.h>

extern const class hkClass hknpVehicleDefaultSteeringClass;

/// The hknpVehicleDefaultSteering class implements a hknpVehicleSteering that maps the steering input from the
/// hknpVehicleDriverInput to a main steering angle, using a velocity-dependent mapping
/// (for high velocities the steering is reduced).
/// The steering for each wheel is either 0.0f (no steering) or equals the main steering angle, depending on a hkBool
/// value specified for each wheel.
///
/// The main steering angle is calculated using the following formula:
///
/// angle = input * maxAngle;
///
/// as input goes from -1 to 1, angle goes from -maxAngle to maxAngle .
class hknpVehicleDefaultSteering : public hknpVehicleSteering
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Default constructor
		hknpVehicleDefaultSteering();

			/// Serialization constructor.
		hknpVehicleDefaultSteering(hkFinishLoadedObjectFlag f) : hknpVehicleSteering(f), m_doesWheelSteer(f) {}

			/// Calculates information about the effects of steering on the vehicle.
		virtual void calcSteering(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput,
			SteeringAnglesOutput& steeringOutput );

			/// Calculates the main steering angle.
		virtual void calcMainSteeringAngle(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput,
			SteeringAnglesOutput& steeringOutput );

	public:

			/// The maximum steering angle (valid for velocities less than m_maxSpeedFullSteeringAngle).
		hkReal m_maxSteeringAngle;

			/// The maximum speed the car still can use the m_maxSteeringAngle.
		hkReal m_maxSpeedFullSteeringAngle;

			/// For every wheel, should be true if the wheel is connected to steering.
		hkArray<hkBool> m_doesWheelSteer;
};

#endif // HKNP_VEHICLE_DEFAULT_STEERING_H

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
