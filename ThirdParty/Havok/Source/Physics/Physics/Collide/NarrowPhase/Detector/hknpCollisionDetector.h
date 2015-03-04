/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_COLLISION_DETECTOR_H
#define HKNP_COLLISION_DETECTOR_H

class hknpInternalCollideSharedData;
class hknpCdCacheStream;
class hknpCdCacheWriter;
class hknpMxJacobianSorter;
class hknpLiveJacobianInfoWriter;
struct hknpCompositeCollisionCache;
struct hknpConvexConvexCollisionCache;
struct hknpCollisionCache;


/// The base class of a collision detector.
/// All collision detectors are registered in the hknpModifierManager.
/// hknpCollisionCache::m_type is used to selected the correct hknpCollisionDetector.
/// Note that convex-convex collisions are special and handled differently.
class hknpCollisionDetector
{
	public:

		hknpCollisionDetector() { m_useChildCaches = false; }

		HK_ON_CPU( virtual ~hknpCollisionDetector() {} )

		///
		virtual void collide(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
			hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,
			hknpConvexConvexCollisionCache* HK_RESTRICT cdCache,
			hknpMxJacobianSorter* jacMxSorter, hknpLiveJacobianInfoWriter* liveJacInfoWriter )
		{
			HK_ASSERT2( 0xf0c4f123, false, "Not implemented");
		}

		///
		virtual void destructCollisionCache(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			hknpCollisionCache* cacheToDestruct,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
			hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
			hknpCdCacheDestructReason::Enum reason ) = 0;

	public:

		HK_PAD_ON_SPU( hkBool32 ) m_useChildCaches;
};


/// Base class of all collision detectors which uses child caches.
/// Note that the hknpCollisionCache implementation needs to subclass from hknpCompositeCollisionCache.
class hknpCompositeCollisionDetector : public hknpCollisionDetector
{
	public:

		/// Constructor
		hknpCompositeCollisionDetector() { m_useChildCaches = true; }

		///
		virtual void collideWithChildren(
			const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
			hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
			hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,
			hknpCompositeCollisionCache* compositeCdCache,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu, // needed by the consumer
			hknpCdCacheWriter* childCdCacheWriter, hknpMxJacobianSorter* jacMxSorter, hknpLiveJacobianInfoWriter* liveJacInfoWriter
			) = 0;

		/// Move the child caches of this stream to childCdCacheWriter.
		virtual void moveAndConsumeChildCaches(
			const hknpSimulationThreadContext& tl, hknpCompositeCollisionCache* compositeCdCache,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
			hknpCdCacheWriter* childCdCacheWriter );

#if !defined(HK_PLATFORM_SPU)
		/// Copy the child caches (does not consume)
		virtual void moveChildCachesWithoutConsuming(
			const hknpSimulationThreadContext& tl, hknpCompositeCollisionCache* compositeCdCache,
			hknpCdCacheWriter* childCdCacheWriter );
#endif

		// hknpCollisionDetector implementation.
		virtual void destructCollisionCache(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			hknpCollisionCache* cacheToDestruct,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
			hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
			hknpCdCacheDestructReason::Enum reason );
};


///
class hknpSetShapeKeyACdDetector : public hknpCollisionDetector
{
	public:

		// hknpCollisionDetector implementation.
		virtual void destructCollisionCache(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			hknpCollisionCache* cacheToDestruct,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
			hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
			hknpCdCacheDestructReason::Enum reason );
};


#endif // HKNP_COLLISION_DETECTOR_H

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
