/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BODY_QUALITY_ID_H
#define HKNP_BODY_QUALITY_ID_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkHandle.h>


/// Body quality identifier.
HK_DECLARE_HANDLE( hknpBodyQualityIdBase, hkUint8, 0xff );
struct hknpBodyQualityId : public hknpBodyQualityIdBase
{
	/// Some preset IDs
	enum Preset
	{
		INVALID = InvalidValue,

		/// Default behavior for static bodies.
		///	 - all flags are supported
		STATIC = 0,

		/// Debris objects
		///	 - no welding, this means this body will hit internal landscape edges.
		///  - might penetrate ground slightly.
		///	 - the debris collisions are still fully continuous.
		///	 - if this object collides with a higher priority object, it still might penetrate the object
		///    (even CRITICAL) objects
		DEBRIS,

		/// Default behavior for dynamic bodies.
		///  - requests neighbor welding (or motion welding on PlayStation(R)3) or no welding, depending on your hknpBodyQualityLibraryCinfo.
		///	 - small penetrations possible.
		///	 - might get pushed by heavy fast objects through static meshes.
		DYNAMIC,

		/// Like DYNAMIC with neighbor welding enabled.
		/// On PlayStation(R)3, this welding type is not available and the engine will revert to MOTION_WELDING.
		NEIGHBOR_WELDING,

		/// Like DYNAMIC with motion welding enabled.
		MOTION_WELDING,

		/// Like DYNAMIC with triangle welding enabled.
		/// On PlayStation(R)3, this welding type is not available and the engine will revert to MOTION_WELDING.
		TRIANGLE_WELDING,

		/// Use this for critical key game objects
		///  - requests neighbor welding (or motion welding on PlayStation(R)3) or no welding, depending on your hknpBodyQualityLibraryCinfo.
		///	 - body should never tunnel through static landscape collisions.
		///	 - this type might limit the angular velocity to avoid tunneling, this comes without extra CPU costs.
		CRITICAL,

		/// Suitable for large, fast bodies moving over terrain.
		///  - uses motion welding, high quality welding which tries to predict the collisions of the car.
		///	 - unlike MOTION_WELDING, it does not request USE_DISCRETE_AABB_EXPANSION.
		VEHICLE,

		/// Suitable for character controllers and items carried by the character.
		/// - uses neighbor welding (or motion welding on PlayStation(R)3).
		/// - does not request USE_DISCRETE_AABB_EXPANSION.
		CHARACTER,

		/// Suitable for small fast objects, which should bounce and slide as good as possible.
		///	 - uses motion and neighbor welding (only on non-PlayStation(R)3 platforms) to avoid ghost collisions.
		///	 - might clip angular velocity.
		GRENADE,

		/// Suitable for fast projectiles, where you don't care what happens after the first hit.
		///  - uses motion welding to avoid ghost collisions.
		///	 - might clip angular velocity.
		PROJECTILE,

		NUM_PRESETS,

		/// Users may use quality IDs from here to MAX_NUM_QUALITIES-1.
		FIRST_USER_ID = 16,
		MAX_NUM_QUALITIES = 32
	};

	/// Empty constructor
	HK_FORCE_INLINE hknpBodyQualityId() {}

	/// Construct from int
	explicit HK_FORCE_INLINE hknpBodyQualityId(int i) : hknpBodyQualityIdBase( i ) {}

	/// Construct from enum
	HK_FORCE_INLINE hknpBodyQualityId(Preset p) : hknpBodyQualityIdBase( int(p) ) {}

	/// Construct from base
	HK_FORCE_INLINE hknpBodyQualityId(const hknpBodyQualityIdBase& b) { m_value = b.valueUnchecked(); }
};


#endif	// HKNP_BODY_QUALITY_ID_H

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
