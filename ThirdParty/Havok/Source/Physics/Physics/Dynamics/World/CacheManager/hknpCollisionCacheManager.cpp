/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Dispatcher/hknpCollisionDispatcher.h>

#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


hknpCollisionCacheManager::hknpCollisionCacheManager( hkThreadLocalBlockStreamAllocator* tlAllocator, int numLinkCells )
{
	m_cdCacheStream.		initBlockStream( tlAllocator );
	m_newCdCacheStream.		initBlockStream( tlAllocator );
	m_newUserCdCacheStream. initBlockStream( tlAllocator );
	m_inactiveCdCacheStream.initBlockStream( tlAllocator );

	m_cdCacheGrid   .setSize( numLinkCells );
	m_newCdCacheGrid.setSize( numLinkCells );
#if defined(HK_PLATFORM_HAS_SPU)
	m_cdCachePpuGrid   .setSize( numLinkCells );
	m_newCdCachePpuGrid.setSize( numLinkCells );
#endif

	m_childCdCacheStream.		 initBlockStream( tlAllocator );
	m_newChildCdCacheStream.	 initBlockStream( tlAllocator );
	m_newUserChildCdCacheStream. initBlockStream( tlAllocator );
	m_inactiveChildCdCacheStream.initBlockStream( tlAllocator );
}

hknpCollisionCacheManager::~hknpCollisionCacheManager()
{
}


void hknpCollisionCacheManager::clear( hkThreadLocalBlockStreamAllocator* tlAllocator )
{
	m_cdCacheStream             .clear( tlAllocator );
	m_newCdCacheStream          .clear( tlAllocator );
	m_newUserCdCacheStream		.clear( tlAllocator );
	m_inactiveCdCacheStream     .clear( tlAllocator );
	m_childCdCacheStream        .clear( tlAllocator );
	m_newChildCdCacheStream     .clear( tlAllocator );
	m_newUserChildCdCacheStream .clear( tlAllocator );
	m_inactiveChildCdCacheStream.clear( tlAllocator );
}

void hknpCollisionCacheManager::addNewPairs(
	const hknpSimulationThreadContext& tl, hkBlockStream<hknpBodyIdPair>::Reader* newPairsReader, int numPairs )
{
	hknpCdCacheWriter cdCachesWriter;
	cdCachesWriter.setToEndOfStream( tl.m_heapAllocator, &m_newCdCacheStream );
	tl.m_world->m_collisionDispatcher->dispatchBodyPairs( tl, newPairsReader, numPairs, &cdCachesWriter );
	cdCachesWriter.finalize();
}

void HK_CALL hknpCollisionCacheManager::filterDeletedPairs( hknpWorld* world, hkArray<hknpBodyIdPair>& pairs )
{
	int d = 0;
	const int n = pairs.getSize();
	for (int i = 0; i < n; i++ )
	{
		const hknpBodyIdPair& pair = pairs[i];
		const hknpBody& bodyA = world->getBodyUnchecked(pair.m_bodyA);
		if ( !bodyA.isValid() || !bodyA.isAddedToWorld() )
		{
			continue;
		}
		const hknpBody& bodyB = world->getBodyUnchecked(pair.m_bodyB);
		if ( !bodyB.isValid() || !bodyB.isAddedToWorld() )
		{
			continue;
		}
		pairs[d++] = pair;
	}
	pairs.setSize( d );
}

void HK_CALL hknpCollisionCacheManager::appendPairsToStream(
	const hknpSimulationThreadContext& tl, hkBlockStream<hknpBodyIdPair>* pairsStream,
	const hknpBodyIdPair* pairs, int numPairs )
{
	hkBlockStream<hknpBodyIdPair>::Writer pairWriter;
	pairWriter.setToEndOfStream( tl.m_tempAllocator, pairsStream );
	for (int i=0; i<numPairs; i++)
	{
		hknpBodyIdPair* pair = pairWriter.reserve<hknpBodyIdPair>();
		*pair = pairs[i];
		pairWriter.advance(sizeof(hknpBodyIdPair));
	}
	pairWriter.finalize();
}

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
