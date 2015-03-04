/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_SIGNED_DISTANCE_COLLISION_CACHE_H
#define HKNP_SIGNED_DISTANCE_COLLISION_CACHE_H

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>


#if defined(HK_PLATFORM_WIN32)
#pragma warning( 3 : 4820 )  // warn about structure padding
#endif


/// Cache used by the hknpConvexDistanceFieldCollisionDetector.
class hknpSignedDistanceFieldChildCollisionCache : public hknpManifoldCollisionCache
{
	public:

		/// Constructor.
		HK_FORCE_INLINE void init();

	public:

		/// The vertex indices into the convex object for the 4 contact points.
		hkUint16 m_vertexIndices[4];

		/// The shape tag of the last frame.
		/// This allows for detecting a change in shape tag.
		hknpShapeTag m_shapeTag;

#if defined(HK_REAL_IS_DOUBLE)
		hkUint8	m_padding4[26];
#else
		hkUint8	m_padding4[6];
#endif
};


/// Cache used for shape-distance field collisions.
struct hknpSignedDistanceFieldCollisionCache : hknpCompositeCollisionCache
{
	public:

		/// Returns the size in bytes.
		HK_FORCE_INLINE int getSizeInBytes() const { return sizeof(this); }

		/// Constructor.
		static void HK_CALL construct(
			const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
			hknpCdCacheWriter* cacheWriterInOut );

	private:

		hknpSignedDistanceFieldCollisionCache() {}
};


#if defined(HK_PLATFORM_WIN32)
#pragma warning( disable : 4820 )
#endif

enum
{
	HKNP_MAX_SDF_CHILD_CACHE_SIZE = sizeof(hknpSignedDistanceFieldChildCollisionCache) + hknpConvexConvexCollisionCache::MAX_PROPERTY_BUFFER_SIZE
};


#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceCollisionCache.inl>

#endif // HKNP_SIGNED_DISTANCE_COLLISION_CACHE_H

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
