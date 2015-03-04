/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_LINEAR_CAST_WHEEL_COLLIDE_H
#define HKNP_VEHICLE_LINEAR_CAST_WHEEL_COLLIDE_H

#include <Physics/Physics/Extensions/Vehicle/WheelCollide/hknpVehicleWheelCollide.h>

extern const class hkClass hknpVehicleLinearCastWheelCollideWheelStateClass;
extern const class hkClass hknpVehicleLinearCastWheelCollideClass;

class hknpVehicleInstance;
class hknpRigidBody;


/// This component manages the collision detection between the wheels and the ground by casting shapes from the wheel
/// hard-points down along the suspension.

class hknpVehicleLinearCastWheelCollide : public hknpVehicleWheelCollide
{
	public:

		/// Information stored per wheel.
		struct WheelState
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE,hknpVehicleLinearCastWheelCollide::WheelState);
			HK_DECLARE_REFLECTION();

			hkAabb m_aabb;				///< An AABB which encompasses the linear cast for the wheel.
			const hknpShape* m_shape;	///< The wheel shape.
			hkTransform m_transform;	///< The transform of the wheel in world space at the start of its linear cast.
			hkVector4 m_to;				///< The end position of the linear cast in world space.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

		/// Default constructor.
		hknpVehicleLinearCastWheelCollide();

		/// Serialization constructor.
		hknpVehicleLinearCastWheelCollide(hkFinishLoadedObjectFlag f) : hknpVehicleWheelCollide(f), m_wheelStates(f)
		{
			if( f.m_finishing )
			{
				m_type = LINEAR_CAST_WHEEL_COLLIDE;
			}
		}

		/// Destructor
		virtual ~hknpVehicleLinearCastWheelCollide();

		//
		// Methods
		//

		/// Initialize the wheelCollide component.
		/// If shapes have not been provided using setWheelShapes, then this method
		/// will create cylinder shapes for each wheel.
		virtual void init( const hknpVehicleInstance* vehicle );

		/// Create a wheel shape of the appropriate size.
		hknpConvexShape* createWheelShape( hkReal width, hkReal radius );

		/// Set the wheel collide object to use the provided shapes.
		/// Using this method before the init method allows the shapes to be shared
		/// within each vehicle and between many vehicles.
		/// The reference counts of the provided shapes will be incremented.
		/// \param vehicle the owning vehicle.
		/// \param shapes an array of pointers to shapes for the wheels.
		void setWheelShapes( const hknpVehicleInstance* vehicle, const hkArray<hknpShape*>& shapes );

		virtual void updateBeforeCollisionDetection( const hknpVehicleInstance* vehicle );

		virtual void addToWorld( hknpWorld* world );

		virtual void removeFromWorld();

		/// Sets the collisionFilterInfo value for the linear casts.
		virtual void setCollisionFilterInfo( hkUint32 filterInfo );

		/// Calls centerWheelContactPoint.
		virtual void wheelCollideCallback(
			const hknpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo );

		/// Move the contact point position to the center plane of the wheel.
		/// This is useful, as the contact point returned by the linear cast can flit from one side
		/// of the wheel to the other.
		void centerWheelContactPoint(
			const hknpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo ) const;

		virtual void collideWheels(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			CollisionDetectionWheelOutput* cdInfoOut,hknpWorld* world );

	public:

		//
		// Internal
		//

		/// Update the m_transform and m_to members of the WheelState.
		/// \param vehicle the owning vehicle.
		/// \param wheelNum the number of the wheel.
		void updateWheelState( const hknpVehicleInstance* vehicle, hkUint8 wheelNum );

		/// Calculates an AABB that encompasses the linear cast of the wheel, possibly excluding the top half
		/// of the wheel in its start position.
		/// \param vehicle the owning vehicle.
		/// \param wheelNum the number of the wheel.
		/// \param aabbOut the resulting AABB.
		void calcAabbOfWheel( const hknpVehicleInstance* vehicle, hkUint8 wheelNum, hkAabb& aabbOut ) const;

		/// Perform a linear cast for a single wheel.
		/// \param vehicle the owning vehicle.
		/// \param wheelNum the number of the wheel to be cast.
		/// \param linearCastOutput the result of the linear cast.
		virtual hkBool castSingleWheel(
			const hknpVehicleInstance* vehicle, hkUint8 wheelNum, hknpWorld* world,
			hknpCollisionResult& linearCastOutput ) const;

		/// Convert linear cast results into collision detection results for a single wheel.
		/// Unlike the case for raycasts, the result will not provide shape keys for
		/// the full shape hierarchy of the struck collidable. Only the first member
		/// of the m_contactShapeKey array will be valid and it will store the shape
		/// key of the struck subshape with respect to that subshape's parent.
		/// \param vehicle the owning vehicle.
		/// \param wheelNum the number of the wheel in the vehicle.
		/// \param linearCastOutput the results obtains from the linear casting.
		/// \param output stores the resulting CollisionDecectionWheelOutput obtained from the raycastOutput.
		virtual void getCollisionOutputFromCastResult(
			const hknpVehicleInstance* vehicle, hkUint8 wheelNum, const hknpCollisionResult& linearCastOutput,
			hknpWorld* world, CollisionDetectionWheelOutput& output ) const;

		/// Get collision results when the wheel is not touching the ground.
		/// \param vehicle the owning vehicle.
		/// \param wheelNum the number of the wheel in the vehicle.
		/// \param output stores the resulting CollisionDecectionWheelOutput.
		virtual void getCollisionOutputWithoutHit(
			const hknpVehicleInstance* vehicle, hkUint8 wheelNum, CollisionDetectionWheelOutput& output ) const;

	public:

		/// The per wheel information.
		hkArray< WheelState > m_wheelStates;

		/// Parameters used when linear casting.
		hkReal m_maxExtraPenetration;
		hkReal m_startPointTolerance;

		hknpBodyId m_chassisBody; //+overridetype(hkUint32)
};

#endif // HKNP_VEHICLE_LINEAR_CAST_WHEEL_COLLIDE_H

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
