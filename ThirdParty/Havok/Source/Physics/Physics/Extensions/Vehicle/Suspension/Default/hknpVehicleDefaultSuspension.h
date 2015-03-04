/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_DEFAULT_SUSPENSION_H
#define HKNP_VEHICLE_DEFAULT_SUSPENSION_H

#include <Physics/Physics/Extensions/Vehicle/Suspension/hknpVehicleSuspension.h>

extern const class hkClass hknpVehicleDefaultSuspensionWheelSpringSuspensionParametersClass;
extern const class hkClass hknpVehicleDefaultSuspensionClass;

/// The hknpVehicleDefaultSuspension class implements a default, spring-based, suspension
/// module for vehicles.
class hknpVehicleDefaultSuspension : public hknpVehicleSuspension
{
	public:

		/// A struct containing all the wheel spring suspension parameters.
		struct WheelSpringSuspensionParameters
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleDefaultSuspension::WheelSpringSuspensionParameters );
			HK_DECLARE_REFLECTION();

			/// The strength [N/m] of the suspension at each wheel.
			hkReal m_strength;

			/// The damping force [N/(m/sec)] of the suspension at each wheel.
			hkReal m_dampingCompression;

			/// The damping force [N/(m/sec)] of the suspension at each wheel.
			hkReal m_dampingRelaxation;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Constructor.
		hknpVehicleDefaultSuspension() {}

			/// Serialization constructor.
		hknpVehicleDefaultSuspension(hkFinishLoadedObjectFlag f) : hknpVehicleSuspension(f), m_wheelSpringParams(f) { }

			/// Calculates information about the effects of suspension on the vehicle.
		virtual void calcSuspension(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hkReal* suspensionForceOut );

	public:

			/// Suspension wheel parameters for each wheel.
		hkArray< WheelSpringSuspensionParameters > m_wheelSpringParams;
};

#endif // HKNP_VEHICLE_DEFAULT_SUSPENSION_H

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
