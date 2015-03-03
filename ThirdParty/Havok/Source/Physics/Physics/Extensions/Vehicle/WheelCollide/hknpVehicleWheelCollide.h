/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_WHEEL_COLLIDE_H
#define HKNP_VEHICLE_WHEEL_COLLIDE_H

#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>

extern const class hkClass hknpVehicleWheelCollideClass;
class hknpVehicleInstance;
class hkContactPoint;

/// This component manages the collision detection between the wheels and the ground.
class hknpVehicleWheelCollide : public hkReferencedObject
{
	public:

		/// The various implementations of this wheel collide interface.
		enum WheelCollideType
		{
			INVALID_WHEEL_COLLIDE = 0,
			RAY_CAST_WHEEL_COLLIDE = 1,
			LINEAR_CAST_WHEEL_COLLIDE = 2,
			USER_WHEEL_COLLIDE1,
			USER_WHEEL_COLLIDE2,
			USER_WHEEL_COLLIDE3,
			USER_WHEEL_COLLIDE4,
			USER_WHEEL_COLLIDE5
		};

		/// Container for data output by the collision calculations.
		struct CollisionDetectionWheelOutput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleWheelCollide::CollisionDetectionWheelOutput );

			/// The point of contact of the wheel with the ground.
			class hkContactPoint m_contactPoint;

			/// The friction coefficient at the point of contact.
			hkReal m_contactFriction;

			/// The ground body the vehicle is in contact.
			/// This value is invalid if the wheel is not in contact with the ground.
			hknpBodyId m_contactBodyId;

			/// The shapeKey of the object at the point of contact.
			hknpShapeKey m_contactShapeKey;

			/// The length of the suspension due to the wheel being in contact at
			/// the given point.
			hkReal m_currentSuspensionLength;

			/// The relative speed along the suspension normal.
			hkReal m_suspensionClosingSpeed;

			/// Scaling factor used to handle curb interaction.
			/// Forces along the contact normal are scaled by this factor.
			/// This ensures that the suspension force component is unscaled.
			/// Clipping is affected by hknpVehicleData::m_normalClippingAngle.
			hkReal m_suspensionScalingFactor;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Serialization constructor.
		hknpVehicleWheelCollide(hkFinishLoadedObjectFlag f) : hkReferencedObject(f)
		{
			if( f.m_finishing )
			{
				m_type = INVALID_WHEEL_COLLIDE;
			}
		}

			/// Destructor
		virtual ~hknpVehicleWheelCollide() {}

			/// Returns the type of the wheel collide object.
		WheelCollideType getType() const { return m_type; }

			/// Initialize the wheelCollide component.
		virtual void init( const hknpVehicleInstance* vehicle ) = 0;

			/// Obtain collision detection information for the wheels of the vehicle.
			/// The caller of this method pre-allocates cdInfoOut with a buffer the size of the number of wheels in the vehicle.
		virtual void collideWheels( const hkReal deltaTime, const hknpVehicleInstance* vehicle, CollisionDetectionWheelOutput* cdInfoOut, hknpWorld* world ) = 0;

			/// Perform any operations needed prior to collision detection.
		virtual void updateBeforeCollisionDetection( const hknpVehicleInstance* vehicle ) = 0;

			/// Add any worldObjects (e.g. Phantoms) to the world, as appropriate.
		virtual void addToWorld( hknpWorld* world ) = 0;

			/// Remove any WorldObjects from the world, as appropriate.
		virtual void removeFromWorld() = 0;

			/// Assign the wheels to have the provided collision filter info.
			/// This value is used by some collision filters, such as the provided hknpGroupFilter.
		virtual void setCollisionFilterInfo( hkUint32 filterInfo ) = 0;

			/// A callback which is called after collision detection on the specified wheel.
			/// It can be used, for example, to override friction.
			/// \param vehicle the owning vehicle.
			/// \param wheelIndex the number of the wheel in the vehicle.
			/// \param cdInfo the collision detection information obtained from wheel collision detection.
		virtual void wheelCollideCallback( const hknpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo ) { }

	protected:

		hknpVehicleWheelCollide() : m_alreadyUsed(false) { m_type = INVALID_WHEEL_COLLIDE; }

	public:

		    /// This component cannot be shared between vehicle instances - this variable
		    /// indicates if a vehicle already owns it.
		hkBool m_alreadyUsed;

			/// Used to distinguish the implementations of this wheel collide interface.
		hkEnum<WheelCollideType, hkUint8> m_type; //+nosave
};

#endif // HKNP_VEHICLE_WHEEL_COLLIDE_H

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
