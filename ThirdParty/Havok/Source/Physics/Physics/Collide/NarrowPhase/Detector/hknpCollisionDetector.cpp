/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/hknpCollisionDetector.h>

#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


void hknpCompositeCollisionDetector::moveAndConsumeChildCaches(
	const hknpSimulationThreadContext& tl, hknpCompositeCollisionCache* compositeCdCache,
	hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
	hknpCdCacheWriter* childCdCacheWriter )
{
	// copy cvx child caches
	hknpCdCacheConsumer cvxReader;
	cvxReader.initSpu(HKNP_SPU_DMA_GROUP_STALL, 1, "moveChildCachesCvxCompositeCacheConsumer");
	cvxReader.setToRange( tl.m_heapAllocator, childCdCacheStream, childCdCacheStreamPpu,  &compositeCdCache->m_childCdCacheRange );
	compositeCdCache->m_childCdCacheRange.setStartPoint( childCdCacheWriter );	// set new start point after we got the old values
	for ( const hknpCollisionCache* srcChildCdCache = cvxReader.access(); srcChildCdCache; srcChildCdCache = cvxReader.consumeAndAccessNext(srcChildCdCache) )
	{
		hknpCollisionCache* dstCache = childCdCacheWriter->reserve( HKNP_MAX_CVX_CVX_CACHE_SIZE );
		hkString::memCpy16NonEmpty(dstCache, srcChildCdCache, srcChildCdCache->m_sizeInQuads);
		childCdCacheWriter->advance( srcChildCdCache->getSizeInBytes() );
	}
	compositeCdCache->m_childCdCacheRange.setEndPoint( childCdCacheWriter );
	cvxReader.exitSpu();
	HK_ON_SPU(hkSpuDmaManager::waitForAllDmaCompletion());
}


/// Copy the child caches (does not consume)
#if !defined(HK_PLATFORM_SPU)
void hknpCompositeCollisionDetector::moveChildCachesWithoutConsuming(
	const hknpSimulationThreadContext& tl, hknpCompositeCollisionCache* compositeCdCache,
	hknpCdCacheWriter* childCdCacheWriter )
{
	hknpCdCacheReader childReader; childReader.setToRange( &compositeCdCache->m_childCdCacheRange );
	compositeCdCache->m_childCdCacheRange.setStartPoint(childCdCacheWriter);
	for ( const hknpCollisionCache* c = childReader.access(); c; c = childReader.advanceAndAccessNext(c->getSizeInBytes()))
	{
		childCdCacheWriter->write16( c );
	}
	compositeCdCache->m_childCdCacheRange.setEndPoint(childCdCacheWriter);
}
#endif


void hknpCompositeCollisionDetector::destructCollisionCache(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	hknpCollisionCache* cacheToDestruct,
	hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
	hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
	hknpCdCacheDestructReason::Enum reason )
{
	hknpCompositeCollisionCache* compCache = static_cast<hknpCompositeCollisionCache*>( cacheToDestruct );
	hknpCdCacheConsumer cvxReader;
	cvxReader.initSpu(HKNP_SPU_DMA_GROUP_STALL, 1, "DeleteChildCdCacheConsumer");
	cvxReader.setToRange( tl.m_heapAllocator, childCdCacheStream, childCdCacheStreamPpu,  &compCache->m_childCdCacheRange );
	for ( const hknpCollisionCache* childCdCache = cvxReader.access(); childCdCache; childCdCache = cvxReader.consumeAndAccessNext(childCdCache) )
	{
		childCdCache->getLeafShapeKeys( &cdBodyA->m_shapeKey, &cdBodyB->m_shapeKey );
		const_cast<hknpCollisionCache*>(childCdCache)->destruct( tl, sharedData, HK_NULL, HK_NULL, cdBodyA, cdBodyB, reason );
	}
	cvxReader.exitSpu();
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
