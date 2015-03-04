/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_DEFAULT_BRAKE_H
#define HKNP_VEHICLE_DEFAULT_BRAKE_H

#include <Physics/Physics/Extensions/Vehicle/Brake/hknpVehicleBrake.h>

extern const class hkClass hknpVehicleDefaultBrakeWheelBrakingPropertiesClass;
extern const class hkClass hknpVehicleDefaultBrakeClass;


/// Implements a braking system that is independent at every wheel.
/// For each wheel there is a maximum torque the brakes can apply, which is scaled by the input from the brake pedal.
/// In addition, wheels can automatically block if the driver presses the brake pedal strongly enough for a certain
/// amount of time. Finally, if the driver is using the handbrake, wheels connected to it will block.
class hknpVehicleDefaultBrake : public hknpVehicleBrake
{
	public:

		///
		struct WheelBrakingProperties
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleDefaultBrake::WheelBrakingProperties );
			HK_DECLARE_REFLECTION();

			/// The maximum torque the wheel can apply when braking. Increasing m_wheelsMass and
			/// m_wheelsMaxBrakingTorque for each wheel allows the vehicle to brake harder.
			hkReal m_maxBreakingTorque;

			/// The minimum amount of braking from the driver that could cause the wheel to block (range [0..1])
			hkReal m_minPedalInputToBlock;

			/// True if the particular wheel is connected to handbrake (locks when handbrake is on).
			hkBool m_isConnectedToHandbrake;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Default constructor
		hknpVehicleDefaultBrake();

			/// Serialization constructor.
		hknpVehicleDefaultBrake(hkFinishLoadedObjectFlag f) : hknpVehicleBrake(f), m_wheelBrakingProperties(f) {}

			/// Calculates information about the effects of braking on the vehicle.
		virtual void calcBreakingInfo(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleDriverInput::FilteredDriverInputOutput& FilteredDriverInputOutput,
			WheelBreakingOutput& breakingInfo );

	public:

			/// Array of braking properties for each of the wheels.
		hkArray<struct WheelBrakingProperties> m_wheelBrakingProperties;

			/// The time (in secs) after which, if the user applies enough brake input, the wheel will block.
		hkReal m_wheelsMinTimeToBlock;
};

#endif // HKNP_VEHICLE_DEFAULT_BRAKE_H

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
