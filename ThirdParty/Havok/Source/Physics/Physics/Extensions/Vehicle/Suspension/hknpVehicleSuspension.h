/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_SUSPENSION_H
#define HKNP_VEHICLE_SUSPENSION_H

#include <Physics/Physics/Extensions/Vehicle/WheelCollide/hknpVehicleWheelCollide.h>

extern const class hkClass hknpVehicleSuspensionSuspensionWheelParametersClass;
extern const class hkClass hknpVehicleSuspensionClass;
class hknpVehicleInstance;

/// The hknpVehicleSuspension class is responsible for calculating and providing all
/// parameters related to the suspension, its forces and how they are applied.
/// Suspension forces have an important influence on a vehicle's driving behavior,
/// as do the position and length of the suspension. For a vehicle using the
/// hknpVehicleRaycastWheelCollide for collision detection, the
/// suspension component provides the basic information needed for the raycasting.
class hknpVehicleSuspension : public hkReferencedObject
{
	public:

		/// Per wheel parameters.
		struct SuspensionWheelParameters
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleSuspension::SuspensionWheelParameters );
			HK_DECLARE_REFLECTION();

			/// A point INSIDE the chassis to which the wheel suspension is attached.
			hkVector4 m_hardpointChassisSpace;

			/// The suspension direction (in Chassis Space).
			hkVector4 m_directionChassisSpace;

			/// The suspension length at rest i.e., the maximum distance from the hardpoint to the wheel center.
			hkReal m_length;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Constructor.
		hknpVehicleSuspension() {}

			/// Serialization constructor.
		hknpVehicleSuspension(hkFinishLoadedObjectFlag f) : hkReferencedObject(f), m_wheelParams(f) {}

			/// Calculates information about the effects of suspension on the vehicle.
			/// The caller of this method pre-allocates suspensionForceOut with a buffer the
			/// size of the number of wheels in the vehicle
		virtual void calcSuspension(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hkReal* suspensionForceOut ) = 0;

	public:

			/// Suspension parameters for each wheel.
		hkArray< SuspensionWheelParameters > m_wheelParams;
};

#endif // HKNP_VEHICLE_SUSPENSION_H

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
