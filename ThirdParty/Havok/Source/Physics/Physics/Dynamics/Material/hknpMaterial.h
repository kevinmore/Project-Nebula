/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MATERIAL_H
#define HKNP_MATERIAL_H

#include <Physics/Physics/hknpTypes.h>

class hknpSurfaceVelocity;


#if defined(HK_PLATFORM_WIN32)
	//#pragma warning( 3 : 4820 )	
#endif


/// A material structure for bodies.
/// Materials typically are just a list of values, which are used by hknpModifier instances to archive specific behaviors.
/// If you create a material with non-default material values, synchronizeFlags() will be called to enable the
/// specific modifiers to implement this material.
/// Note: If you ever feel restricted by the material parameters, you can always create your own hknpModifier.
class hknpMaterial
{
	public:

		/// Material Flags.
		/// These flags use the upper 12 bits of a 32 bit field, and can be OR'ed with hknpBody::Flags to produce
		/// a 32 bit flags value. A modifier is executed if at least one of these 32 flags is registered with any
		/// modifier in hknpModifierManager.
		enum FlagsEnum
		{
			//
			// Flags to enable built in modifiers.
			//

			ENABLE_RESTITUTION			= 1<<20,	///< Enables restution for any objects using this material.
			ENABLE_TRIGGER_VOLUME		= 1<<21,	///< Enables trigger volume for any objects using this material.
			ENABLE_IMPULSE_CLIPPING		= 1<<22,	///< Enables impulse clipping for any objects using this material.
			ENABLE_MASS_CHANGER			= 1<<23,	///< Enables mass changing for any objects using this material.
			ENABLE_SOFT_CONTACTS		= 1<<24,	///< Enables soft contacts for any objects using this material.
			ENABLE_SURFACE_VELOCITY		= 1<<25,	///< Enables surface velocity for any objects using this material.

			//
			// User flags. Can be used to enable user modifiers.
			//

			USER_FLAG_0	= 1<<26,
			USER_FLAG_1	= 1<<27,
			USER_FLAG_2	= 1<<28,
			USER_FLAG_3	= 1<<29,
			USER_FLAG_4	= 1<<30,

			//
			// Masks
			//

			FLAGS_MASK		= 0xfff00000,	///< Mask of all allowed flags
			AUTO_FLAGS_MASK	= 0x3f << 20	///< Mask of flags for built in modifiers
		};

		typedef hkFlags<FlagsEnum, hkUint32> Flags;

		/// Since the friction and restitution between two bodies depends on the value in each body's material,
		/// we need some policies to combine those two values into one. The policy is selected using:
		/// maximum( bodyA.m_frictionCombinePolicy, bodyB.m_frictionCombinePolicy ).
		enum CombinePolicy
		{
			COMBINE_AVG,	///< Use sqrt( frictionA * frictionB )
			COMBINE_MIN,	///< Use  min( frictionA , frictionB )
			COMBINE_MAX,	///< Use  max( frictionA , frictionB )
		};

		/// Trigger volume types.
		enum TriggerVolumeType
		{
			/// Do not treat this material as a trigger volume.
			TRIGGER_VOLUME_NONE,

			/// Raise manifold based trigger volume events, during the speculative collision detection phase.
			/// This is cheap and recommended, but may raise false positive events against fast bodies.
			TRIGGER_VOLUME_LOW_QUALITY,

			/// Raise impulse based trigger volume events, during the solving phase.
			/// This raises more accurate events than TRIGGER_VOLUME_LOW_QUALITY but at additional CPU cost.
			TRIGGER_VOLUME_HIGH_QUALITY
		};

		/// Categories for the default mass changer.
		enum MassChangerCategory
		{
			MASS_CHANGER_IGNORE,	///< Do not apply the mass changer on this object. (default)
			MASS_CHANGER_DEBRIS,	///< Reduce the mass of this object if it collides with MASS_CHANGER_HEAVY.
			MASS_CHANGER_HEAVY,		///< If this object collides with an object of category MASS_CHANGER_DEBRIS, increase this mass by the factor stored in m_massChangerHeavyObjectFactor.
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMaterial );
		HK_DECLARE_REFLECTION();

		/// Constructor. Initializes the members with default values.
		hknpMaterial();

		/// Serialization constructor.
		hknpMaterial( hkFinishLoadedObjectFlag flag );

		/// Copy constructor.
		hknpMaterial( const hknpMaterial& other );

		/// Comparison operator.
		HK_FORCE_INLINE bool operator==( const hknpMaterial& other ) const;

		/// Update the m_flags field based on the values set in the material.
		void synchronizeFlags();

		// Internal: Operations for hkFreeListArray.
		struct FreeListArrayOperations
		{
			HK_FORCE_INLINE static void setEmpty( hknpMaterial& material, hkUint32 next );
			HK_FORCE_INLINE static hkUint32 getNext( const hknpMaterial& material );
			HK_FORCE_INLINE static hkBool32 isEmpty( const hknpMaterial& material );
		};

	public:

		/// A user-defined name for the material.
		HK_ALIGN16( hkStringPtr ) m_name;

		/// If set to TRUE, this material is not allowed to be merged with other identical materials.
		/// Note: this must be the first value member (check out operator==()) and must be 32bit since
		/// it is reused the the hknpMaterialLibrary:: FreeListArray
		hkBool32 m_isExclusive;

		/// Material flags. See FlagsEnum.
		/// These flags are set automatically based on the values in hknpMaterial.
		Flags m_flags;

		//
		// Friction and restitution
		//

		/// Dynamic friction. See also m_frictionCombinePolicy.
		hkHalf m_dynamicFriction; // default(0.5f)

		/// Static friction. Combine policy is always COMBINE_MAX.
		/// This is active only if static friction is greater than dynamic friction.
		hkHalf m_staticFriction; // default(0.5f)

		/// Restitution. See also m_restitutionCombinePolicy.
		/// Note: If you set this value after the body is created, you need to ensure that
		/// m_flags has ENABLE_RESTITUTION set.
		hkHalf m_restitution; // default(0.0f)

		/// Selected algorithm for combining the two dynamic friction values of the colliding bodies into one.
		/// Note: The combine policy of static friction is always COMBINE_MAX.
		
		hkEnum<CombinePolicy, hkUint8> m_frictionCombinePolicy; // default(COMBINE_MIN)

		/// Selected algorithm for combining the two restitution values of the colliding bodies into one.
		
		hkEnum<CombinePolicy, hkUint8> m_restitutionCombinePolicy; // default(COMBINE_MAX)

		//
		// Welding
		//

		/// Welding tolerance is used when the hknpBodyQualityId is set to a type which supports welding.
		/// Welding will fix up internal edge collisions, e.g. when an object slides over a landscape made of connected
		/// triangles and the object hits an internal triangle edge. If two triangles are not perfectly aligned, the
		/// m_weldingTolerance specifies a vertical distance or 'step height' where 2 triangle would be treated connected.
		/// So if your landscape has a small step smaller than this m_weldingTolerance, it will be completely smoothed.
		///
		/// Notes:
		///		- should the landscape body return childShapes with child materials, the final welding tolerance will calculated
		///		  from the max between both body materials and the child materials.
		///		- welding is enabled by the hknpBodyCinfo::m_qualityId
		hkHalf m_weldingTolerance;	// default(0.05f)

		//
		// Trigger volume
		//

		/// If set (!= TRIGGER_VOLUME_NONE) then any shape using this material becomes a trigger volume.
		/// Trigger volume events will be raised if the body has the RAISE_TRIGGER_VOLUME_EVENTS flags set.
		/// No collision response will ever occur. Note: if you set this value after the body is created, you must
		/// ensure that m_flags has ENABLE_TRIGGER_VOLUME set.
		hkEnum<TriggerVolumeType, hkUint8> m_triggerVolumeType; // default(TRIGGER_VOLUME_NONE)

		/// A distance defining the collision detection accuracy for trigger volumes.
		/// Smaller values result in improved accuracy by performing collision detection more frequently.
		/// For example, a value of 0.1f means that the trigger volume surface is accurate within around 10cm.
		hkUFloat8 m_triggerVolumeTolerance; // default(HK_REAL_MAX)

		//
		// Impulse clipping
		//

		/// Limits the impulse that contacts can apply, and raises hknpContactImpulseExceededEvents if exceeded.
		/// Note: if you set this value after the body is created, you need to ensure that
		/// m_flags has ENABLE_IMPULSE_CLIPPING set.
		hkReal m_maxContactImpulse; // default(HK_REAL_MAX)

		/// A fraction of the max contact impulse to apply between bodies when the impulse is clipped.
		/// If set to 1.0f, an impulse of m_maxContactImpulse will be applied.
		/// If set to 0.0f, no impulse will be applied.
		hkReal m_fractionOfClippedImpulseToApply; // default(1.0f)

		//
		// Mass changer
		//

		/// Enable the mass changer for this body. See MassChangerCategory for details.
		hkEnum<MassChangerCategory, hkUint8> m_massChangerCategory; // default(MASS_CHANGER_IGNORE)

		/// The mass changer factor if category MASS_CHANGER_HEAVY is enabled.
		hkHalf m_massChangerHeavyObjectFactor; // default(1.0f)

		//
		// Soft contact
		//

		/// If set (value !=0), the soft contact modifier is enabled for all collisions using this material.
		/// This value defines a factor of the forces applied. For values less than 1.0, this results in a penetration.
		/// Note: If you set this value after the body is created, you need to ensure that
		/// m_flags has ENABLE_SOFT_CONTACTS set.
		hkHalf m_softContactForceFactor;

		/// The damping of soft contacts. The lower this value is, the more springy the contact becomes.
		/// Must be smaller than 1.0.
		hkHalf m_softContactDampFactor;

		/// Because of the "softness" of soft contacts, objects can penetrate deeply. This parameter defines how quickly
		/// the objects get pushed apart. Note: If you set this value after the body is created, you need to ensure
		/// that m_flags has ENABLE_SOFT_CONTACTS set.
		hkUFloat8 m_softContactSeperationVelocity;

		//
		// Surface velocity
		//

		/// An optional surface velocity implementation.
		/// Note: If you set this value after the body is created, you need to ensure that
		/// m_flags has ENABLE_SURFACE_VELOCITY set.
		hknpSurfaceVelocity* m_surfaceVelocity; // default(HK_NULL)

		//
		// Advanced
		//

		/// Allows to disable new collisions between dynamic objects which both have the TEMP_DROP_NEW_CVX_CVX_COLLISIONS
		/// bit set in hknpBody::m_flags.
		/// This is a special feature to help to reduce the performance spike when lots of debris pieces are inserted
		/// into the world roughly at the same position. If you set hknpBody::m_flags |= TEMP_DROP_NEW_CVX_CVX_COLLISIONS,
		/// than all new collisions generated at that frame between dynamic convex bodies using this material will have
		/// their collisions filtered until the combined movement of the two bodies is bigger than
		/// m_disablingCollisionsBetweenCvxCvxDynamicObjectsDistance.
		hkHalf m_disablingCollisionsBetweenCvxCvxDynamicObjectsDistance; // default(5.0f)

	protected:

		/// Set if this material was merged with another material in the material library.
		/// Note: This must be the last member.
		hkBool m_isShared;

		friend class hknpMaterialLibrary;
};

#if defined(HK_PLATFORM_WIN32)
	//#pragma warning( disable : 4820 )
#endif


/// A referenced counted material wrapper.
class hknpRefMaterial : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		HK_FORCE_INLINE hknpRefMaterial() {}
		hknpRefMaterial( hkFinishLoadedObjectFlag flag );

	public:

		hknpMaterial m_material;
};


/// A material descriptor is used to refer to an abstract material that may have not been created yet in the material
/// library. The description may be a name string, a full material specification or a material ID. To obtain an actual
/// material corresponding to this description you must call hknpMaterialLibrary::addEntry() with it. Refer to the method
/// description for details about how descriptors are used to obtain materials.
struct hknpMaterialDescriptor
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMaterialDescriptor );
		HK_DECLARE_REFLECTION();

		HK_FORCE_INLINE hknpMaterialDescriptor();
		hknpMaterialDescriptor( hkFinishLoadedObjectFlag flag );

	public:

		/// A name that will be used for late binding to an existing material if provided.
		hkStringPtr m_name;

		/// An optional material that will be inserted into the material library if no name was provided
		/// or no matching name was found.
		hkRefPtr<hknpRefMaterial> m_material;

		/// A material ID in the material library.
		hknpMaterialId m_materialId; //+overridetype(hkUint16)
};


#include <Physics/Physics/Dynamics/Material/hknpMaterial.inl>

#endif // HKNP_MATERIAL_H

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
