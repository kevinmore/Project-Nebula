/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_RAY_CAST_WHEEL_COLLIDE_H
#define HKNP_VEHICLE_RAY_CAST_WHEEL_COLLIDE_H

#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/WheelCollide/hknpVehicleWheelCollide.h>

extern const class hkClass hknpVehicleRayCastWheelCollideClass;

class hkAabb;
class hknpClosestHitCollector;
class hknpWorld;


/// A hknpVehicleWheelCollide implementation which uses ray casts to determine wheel collision detection.
/// This component cannot be shared between vehicles.
class hknpVehicleRayCastWheelCollide : public hknpVehicleWheelCollide
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

		/// Default constructor
		hknpVehicleRayCastWheelCollide();

		/// Serialization constructor.
		hknpVehicleRayCastWheelCollide(hkFinishLoadedObjectFlag f) : hknpVehicleWheelCollide(f)
		{
			if( f.m_finishing )
			{
				m_type = RAY_CAST_WHEEL_COLLIDE;
			}
		}

		/// Destructor.
		virtual ~hknpVehicleRayCastWheelCollide();

		//
		// Methods
		//

		virtual void init( const hknpVehicleInstance* vehicle );

		virtual void updateBeforeCollisionDetection( const hknpVehicleInstance* vehicle );

		virtual void addToWorld( hknpWorld* world );

		virtual void removeFromWorld();

		///	Sets the collisionFilterInfo value for the ray casts.
		virtual void setCollisionFilterInfo( hkUint32 filterInfo );

		/// Implements the single threaded approach to wheel collisions by calling
		/// castSingleWheel and collideSingleWheelFromRaycast for each wheel.
		virtual void collideWheels(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			CollisionDetectionWheelOutput* cdInfoOut, hknpWorld* world );

	public:

		/// Calculates an AABB that encompasses the raycasts.
		/// \param vehicle the owning vehicle.
		/// \param aabbOut the resulting AABB.
		virtual void calcWheelsAABB(
			const hknpVehicleInstance* vehicle, hkAabb& aabbOut ) const;

		/// Perform a raycast for a single wheel.
		virtual void castSingleWheel(
			const hknpVehicleInstance::WheelInfo& wheelInfo, hknpWorld* const world,
			hknpCollisionQueryCollector* collector ) const;

		/// Convert raycast results into collision detection results for a single wheel.
		/// \param vehicle the owning vehicle.
		/// \param wheelInfoNum the number of the wheel in the vehicle.
		/// \param raycastOutput the results obtains from raycasting.
		/// \param output stores the resulting CollisionDecectionWheelOutput obtained from the raycastOutput.
		virtual void getCollisionOutputFromCastResult(
			const hknpVehicleInstance* vehicle, hkUint8 wheelInfoNum, const hknpClosestHitCollector& raycastOutput,
			CollisionDetectionWheelOutput& output, hknpWorld* world ) const;

		/// Get collision results when the wheel is not touching the ground.
		/// \param vehicle the owning vehicle.
		/// \param wheelNum the number of the wheel in the vehicle.
		/// \param output stores the resulting CollisionDecectionWheelOutput.
		virtual void getCollisionOutputWithoutHit(
			const hknpVehicleInstance* vehicle, hkUint8 wheelNum, CollisionDetectionWheelOutput& output ) const;

	public:

		/// The collision filter info of the wheels.
		/// This is initialized to 0 by the constructor. If a different value is needed,
		/// it should be assigned after construction but before calling init.
		/// After init, use setCollisionFilterInfo to modify its value.
		hkUint32 m_wheelCollisionFilterInfo;
};

#endif // HKNP_VEHICLE_RAY_CAST_WHEEL_COLLIDE_H

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
