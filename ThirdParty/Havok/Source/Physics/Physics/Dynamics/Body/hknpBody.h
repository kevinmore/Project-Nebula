/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BODY_H
#define HKNP_BODY_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Types/Geometry/Aabb16/hkAabb16.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>
#include <Common/Base/Math/Vector/hkPackedVector3.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQuality.h>

extern const hkClass hknpBodyClass;
extern const hkClass hknpBodyCinfoClass;

struct hknpBodyCinfo;

#if defined(HK_PLATFORM_WIN32)
#	pragma warning( 3 : 4820 )		
#endif


/// The rigid body structure.
/// Bodies are allocated and owned by the world, which creates a persistent ID for each body.
/// The ID is an index into a body array and will not change during the lifetime of the body.
/// Typically users get only read only access to this data structure. Use the hknpWorld::setBodyXxx() functions for
/// write access.
class hknpBody
{
	// +version(1)
	public:

		/// Body Flags.
		/// These flags use the lower 20 bits of a 32 bit field, and can be OR'ed with hknpMaterial::Flags to produce
		/// a 32 bit flags value. A modifier is executed if at least one of these 32 flags is registered with any
		/// modifier in hknpModifierManager.
		enum FlagsEnum
		{
			//
			// Internal flags. These are not user controllable.
			// Valid bodies must have either the IS_STATIC or IS_DYNAMIC flag set.
			//

			IS_STATIC		= 1<<0,		///< Set for bodies using the static motion.
			IS_DYNAMIC		= 1<<1,		///< Set for bodies using a dynamic motion.
			IS_KEYFRAMED	= 1<<2,		///< Set for bodies using a dynamic motion with infinite mass.
			IS_ACTIVE		= 1<<3,		///< Set for bodies using a dynamic motion which is currently active.

			//
			// Flags to enable event raising modifiers.
			//

			/// Raise a hknpTriggerVolumeEvent whenever a shape using a trigger volume material is entered or exited.
			/// See hknpMaterial::m_triggerVolumeType and hknpMaterial::m_triggerVolumeTolerance for related settings.
			RAISE_TRIGGER_VOLUME_EVENTS		= 1<<4,

			/// Raise a hknpManifoldStatusEvent whenever a contact manifold using this body is created or destroyed
			/// during collision detection.
			RAISE_MANIFOLD_STATUS_EVENTS	= 1<<5,

			/// Raise a hknpManifoldProcessedEvent whenever a contact manifold using this body is processed
			/// during collision detection, including the step when a manifold is created.
			RAISE_MANIFOLD_PROCESSED_EVENTS	= 1<<6,

			/// Raise a hknpContactImpulseEvent whenever a contact Jacobian using this body starts applying non-zero
			/// impulses during solving. Events can be raised for subsequent solver steps by calling
			/// hknpContactImpulseEvent::set[Continued|Finished]EventsEnabled() from your event handler.
			RAISE_CONTACT_IMPULSE_EVENTS	= 1<<7,

			//
			// Flags to enable special simulation behaviors.
			//

			/// Disable all collision cache creation for this body, independent of how the collision filter is set up.
			/// This can be used if a body is only ever intended to be queried, for example.
			DONT_COLLIDE					= 1<<8,

			/// Disable all contact solving for this body.
			/// This is typically set in the collision cache flags by the trigger volume modifier.
			DONT_BUILD_CONTACT_JACOBIANS	= 1<<9,

			//
			// Flags that are cleared after every simulation step. These are used to either:
			//  - trigger specific collision detection functions, or
			//  - trigger single step modifiers
			//

			/// Force collision caches to be rebuilt during the next collision detection step.
			TEMP_REBUILD_COLLISION_CACHES	= 1<<10,

			/// Allows for optimizations of exploding debris scenarios, see
			/// hknpMaterial::m_disablingCollisionsBetweenCvxCvxDynamicObjectsDistance.
			TEMP_DROP_NEW_CVX_CVX_COLLISIONS = 1<<11,

			TEMP_USER_FLAG_0 = 1<<12,
			TEMP_USER_FLAG_1 = 1<<13,

			/// Do not add the body to the world.
			IS_NON_RUNTIME	 = 1<<14,	// This flag is temporary and will be removed in the future.

			/// Destruction specific. Indicates the body can break into pieces
			IS_BREAKABLE	 = 1<<15,

			//
			// User flags. Can be used to enable user modifiers.
			//

			USER_FLAG_0	= 1<<16,
			USER_FLAG_1	= 1<<17,
			USER_FLAG_2	= 1<<18,
			USER_FLAG_3	= 1<<19,

			//
			// Masks
			//

			FLAGS_MASK			= 0x000fffff,	///< Mask of all allowed flags
			INTERNAL_FLAGS_MASK	= 0xf << 0,		///< Mask of all "internal" flags.
			EVENT_FLAGS_MASK	= 0xf << 4,		///< Mask of all "event" flags.
			TEMP_FLAGS_MASK		= 0xf << 10		///< Mask of all "temp" flags.
		};

		typedef hkFlags<FlagsEnum, hkUint32> Flags;

		enum SpuFlagsEnum
		{
			FORCE_NARROW_PHASE_PPU	= 1 << 0,

			// Not supported yet
			// FORCE_SOLVER_PPU		= 1 << 1
		};

		typedef hkFlags<SpuFlagsEnum, hkUint8> SpuFlags;

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBody );
		HK_DECLARE_REFLECTION();
		HK_DECLARE_POD_TYPE();

		/// Construct an uninitialized body.
		HK_FORCE_INLINE hknpBody();

		/// Copy operator.
		HK_FORCE_INLINE void operator=( const hknpBody& other );

		/// Returns non zero if the body has been initialized (as static or dynamic).
		HK_FORCE_INLINE hkBool32 isValid() const;

		/// Returns non zero if the body is added to the world.
		HK_FORCE_INLINE hkBool32 isAddedToWorld() const;

		/// Returns non zero if the body is static and not deleted.
		HK_FORCE_INLINE hkBool32 isStatic() const;

		/// Returns non zero if the body is a dynamic body and not deleted.
		/// Note that keyframed bodies are dynamic bodies.
		HK_FORCE_INLINE hkBool32 isDynamic() const;

		/// Returns non zero if the body is keyframed and not deleted.
		HK_FORCE_INLINE hkBool32 isKeyframed() const;

		/// Returns non zero if the body is static or keyframed.
		/// Note that keyframed bodies are dynamic bodies and can deactivate.
		HK_FORCE_INLINE hkBool32 isStaticOrKeyframed() const;

		/// Returns non zero if the body is dynamic and active.
		/// Note that static bodies are never active.
		HK_FORCE_INLINE hkBool32 isActive() const;

		/// Returns non zero if the body is a deactivated dynamic body.
		HK_FORCE_INLINE hkBool32 isInactive() const;

		/// Get the body transform.
		HK_FORCE_INLINE const hkTransform& getTransform() const;

		/// Get the look ahead distance.
		/// See hknpBodyCinfo::m_collisionLookAheadDistance for details.
		HK_FORCE_INLINE const hkReal& getCollisionLookAheadDistance() const;

		/// Get the center of mass in local space.
		/// If you transform this position into world space, you will get the motion's space origin.
		HK_FORCE_INLINE void getCenterOfMassLocal( hkVector4& comLocalOut ) const;

		//
		// Methods for internal use only
		//

		/// Set (some of) the body properties based on construction information.
		void initialize( hknpBodyId id, const hknpBodyCinfo& info );

		/// Set the shape and the cached shape size.
		HK_FORCE_INLINE void setShape( const hknpShape* shape );

		/// Set the transform (XYZ components only).
		HK_FORCE_INLINE void setTransform( const hkTransform& newTransform );

		/// Set all components of the transform, including the local COM and the look ahead distance.
		HK_FORCE_INLINE void setTransformComAndLookAhead( const hkTransform& newTransform );

		/// Set the look ahead distance.
		HK_FORCE_INLINE void setCollisionLookAheadDistance( hkReal distance );

		/// Set the mass center in local space of the body.
		void setBodyToMotionTranslation( hkVector4Parameter offset );

		/// Gets the transform from body space to motion (center of mass) space.
		/// This works even for static bodies (that are associated with the special motion 0).
		void getBodyToMotionTransform( hkQTransform& transform ) const;

		/// Get the relative transform between body and motion.
		/// Note the the orientation is only 16bit accurate.
		HK_FORCE_INLINE void getMotionToBodyTransformEstimate( hkQTransform* bodyFromMotionOut ) const;

		/// Sets the relative transform between body and motion: m_motionToBodyTranslation and m_motionToBodyRotation.
		/// This needs to be called if the bodies motion was moved and you want to keep the bodies transform unchanged.
		/// In this case the relative transform between body and motion changes. This function will calculate and set
		/// this relative transform.
		/// Note if you have the body orientation in quaternion form you should pass it in \a cachedBodyOrientation
		/// to avoid unnecessary conversions.
		void updateMotionToBodyTransform( const hknpMotion& motion, const hkQuaternion* cachedBodyOrientation = HK_NULL );

		/// Calculate and update the center-of-mass centered bounding radius.
		void updateComCenteredBoundingRadius( const hknpMotion& motion );

		/// Syncs local COM and m_motionToBodyRotation if a static body has been repositioned.
		void syncStaticMotionToBodyTransform();

		/// Get the index of the deactivated island to be used in the hknpDeactivationManager.
		HK_FORCE_INLINE int getDeactivatedIslandIndex() const;

	protected:

		/// Body position and orientation in world space in the XYZ components, and the following in the W components:
		///  - W[0,1,2] : mass center in local space
		///  - W[3]     : look ahead distance
		hkTransform m_transform;

		friend struct hknpInternalMotionUtilFunctions2;	

	public:

		

		/// The expanded AABB of the body in integer space.
		/// Use hknpWorld::m_intSpaceUtil to convert to a floating point hkAabb.
		hkAabb16 m_aabb;

		

		/// The collision shape of the body.
		/// Note that this is not reference counted.
		const hknpShape* m_shape;

		/// Body flags. See FlagsEnum.
		Flags m_flags;

		/// Filter information for hknpCollisionFilter.
		hkUint32 m_collisionFilterInfo;

		

		/// This body's identifier. This can be used in the hknpWorld::setBodyXxx() methods.
		hknpBodyId m_id; //+overridetype(hkUint32)

		/// Identifier of the next body which shares the same motion.
		/// All bodies attached to a motion form a ring. To gather all bodies sharing this motion, iterate through them
		/// until you reach this body's ID again.
		hknpBodyId m_nextAttachedBodyId; //+overridetype(hkUint32)

		/// The motion identifier. Use hknpWorld::getMotion() to retrieve the motion.
		/// Note that static bodies always have a motion ID of zero.
		hknpMotionId m_motionId; //+overridetype(hkUint32)

		/// The material identifier. Use hknpWorld::getMaterial() to retrieve the material.
		/// Note that composite shape children can override their material by using a Shape Tag Codec.
		hknpMaterialId m_materialId; //+overridetype(hkUint16)

		/// The quality identifier. Use hknpWorld::getQuality() to retrieve the quality.
		hknpBodyQualityId m_qualityId; //+overridetype(hkUint8)

		/// Byte size of the shape divided by 16. Used on SPU for DMA transfers.
		hkUint8 m_shapeSizeDiv16;

		

		/// An identifier which can be used by the broad phase.
		hknpBroadPhaseId m_broadPhaseId;

		/// The index into the activeBodies in the body manager if the body is active, or
		/// the index into the island list if the body is deactivated (use getDeactivatedIslandIndex()), or
		/// hknpBody::invalid() if the body is not added to the world.
		hknpBodyId::Type m_indexIntoActiveListOrDeactivatedIslandId;

		/// The radius of the bounding sphere of the shape around the mass center.
		hkHalf m_radiusOfComCenteredBoundingSphere;

		/// The maximum distance for contact manifold generation during narrow-phase collision detection.
		/// Calculated in each step based on quality, velocity, bounding radius, and hknpWorld::m_collisionTolerance.
		hknpLinearTimType m_maxContactDistance;

		/// The maximum distance any point on the body's surface traveled in the last integration step.
		/// To compute a floating point distance multiply this value with hknpWorld::m_solverInfo.m_linearTimToDistance.
		hknpLinearTimType m_maxTimDistance;

		/// The angle the object rotated in the last integration step. Range [0:255] maps to [0:90] degrees.
		
		hkUint8	m_timAngle;

		/// Flags detailing SPU support.
		SpuFlags m_spuFlags;  //+default(0)

		

		/// The orientation from motion to body space.
		/// The motion space is always the space where the inertia tensor becomes a diagonal matrix.
		hkPackedUnitVector<4> m_motionToBodyRotation; //+overridetype(hkInt16[4])

#if (HK_POINTER_SIZE == 4)
		HKNP_PADDING_BYTES( m_padding2, 4 );	//+nosave
#endif

		/// User data. Not used by the engine.
		
		mutable hkUint64 m_userData;

		
};

#if defined(HK_PLATFORM_WIN32)
#	pragma warning( disable : 4820 )
#endif


/// Serializable construction information for a hknpBody.
struct hknpBodyCinfo
{
	// +version(1)
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyCinfo );
		HK_DECLARE_REFLECTION();

		/// Constructor. Initializes the members with default values.
		hknpBodyCinfo();

		/// Serialization constructor.
		hknpBodyCinfo( hkFinishLoadedObjectFlag flag ) : m_name(flag), m_localFrame(flag) {}

	public:

		/// The collision shape.
		/// Note: This is not reference counted.
		const hknpShape* m_shape;	//+owned(false)

		/// Optional preallocated ID for this body.
		hknpBodyId m_reservedBodyId; //+overridetype(hkUint32) // default( hknpBodyId::invalid() )

		/// Optional existing motion ID for this body.
		hknpMotionId m_motionId;	//+overridetype(hkUint32) // default( hknpMotionId::invalid() )

		/// The quality identifier.
		/// Defaults to INVALID, in which case the STATIC or DYNAMIC quality will be automatically chosen.
		hknpBodyQualityId m_qualityId; //+overridetype(hkUint8) // default( hknpBodyQualityId::invalid() )

		/// The material identifier.
		/// Note that composite shape children can override their material by using a Shape Tag Codec.
		/// Defaults to hknpMaterialId::DEFAULT.
		hknpMaterialId m_materialId; //+overridetype(hkUint16)

		/// Collision filter information. Used by hknpCollisionFilter.
		/// Defaults to 0.
		hkUint32 m_collisionFilterInfo; //+default(0)

		/// Body flags, mainly used to enable events and modifiers.
		/// Defaults to 0.
		hknpBody::Flags m_flags; //+default(0)

		/// Extra collision distance for the collision detector which effectively enables continuous collision
		/// solving for this body.
		/// If set, the collision detector will virtually expand this body by its velocity.length().
		/// (imagine an extra shell around the body with a certain thickness):
		///		- contact points will be created as soon as this shell touches another object.
		///     - the body acceleration will be limited so that body can never leave this shell in one frame.
		///       Example: If this is 50cm and physics is running at 30Hz, the maximum acceleration of the body is
		///       15 meter/second every physics step.
		/// Note that by increasing the collision look ahead, you get potentially much more collision pairs, so try
		/// to set this value to reasonable values.
		/// If you set this parameter on the hknpBody, you can set a tempExpansionVelocity as well. This parameter
		/// will do an extra expansion of the body's AABB by that velocity (using the direction of the velocity).
		/// The tempExpansionVelocity of the body will be cleared at the end of the simulation step. This tempExpansionVelocity
		/// should be used if you expect a rapid change of velocity in that direction, e.g. object just in front of
		/// a car.
		hkReal m_collisionLookAheadDistance;	//+default(0)

		/// Optional body name.
		hkStringPtr m_name;

		/// Initial body position in world space.
		hkVector4 m_position;

		/// Initial body orientation.
		hkQuaternion m_orientation; //+default(0.0f,0.0f,0.0f,1.0f)

		/// Flags detailing SPU support.
		hknpBody::SpuFlags m_spuFlags; //+default(0)

		/// Pointer to a local frame exported by the content tools. Not directly used in Physics.
		hkRefPtr<hkLocalFrame> m_localFrame;
};

#include <Physics/Physics/Dynamics/Body/hknpBody.inl>

#endif // HKNP_BODY_H

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
