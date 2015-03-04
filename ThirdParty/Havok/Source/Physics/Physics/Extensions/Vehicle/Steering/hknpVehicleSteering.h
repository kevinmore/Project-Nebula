/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_STEERING_H
#define HKNP_VEHICLE_STEERING_H

#include <Physics/Physics/Extensions/Vehicle/DriverInput/hknpVehicleDriverInput.h>

extern const class hkClass hknpVehicleSteeringClass;
class hknpVehicleInstance;


/// An abstract component for vehicle steering for both the main (chassis) and individual wheel steering angles.
/// The hknpVehicleSteering class is responsible for calculating the main steering angle of the vehicle and the steering
/// angle of each wheel. A typical implementation will be likely to use the steering wheel input value from the vehicle's
/// hknpVehicleDriverInput and transform it into a steering angle.
class hknpVehicleSteering : public hkReferencedObject
{
	public:

		/// Container for data output by the steering calculations.
		struct SteeringAnglesOutput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleSteering::SteeringAnglesOutput );

			/// The angle the steering wheel has been turned.
			hkReal m_mainSteeringAngle;
			hkReal m_mainSteeringAngleAssumingNoReduction;

			/// The angle a wheel turns (if it can be steered).
			hkInplaceArray< hkReal, 32 > m_wheelsSteeringAngle;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Serialization constructor.
		hknpVehicleSteering(hkFinishLoadedObjectFlag flag) : hkReferencedObject(flag) {}

			/// Calculates information about the effects of steering on the vehicle.
		virtual void calcSteering(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredInfoOutput,
			SteeringAnglesOutput& steeringOutput ) = 0;

	protected:

		hknpVehicleSteering() {}
};

#endif // HKNP_VEHICLE_STEERING_H

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
