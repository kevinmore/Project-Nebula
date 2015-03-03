/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_CACHE_MANAGER_H
#define HKNP_COLLISION_CACHE_MANAGER_H

#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>


/// A utility class dispatching and storing all collision caches.
class hknpCollisionCacheManager
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollisionCacheManager );

		/// Constructor.
		hknpCollisionCacheManager( hkThreadLocalBlockStreamAllocator* tlAllocator, int numLinkCells );

		/// Destructor.
		~hknpCollisionCacheManager();

		/// Add new collision pairs.
		void addNewPairs(
			const hknpSimulationThreadContext& tl, hkBlockStream<hknpBodyIdPair>::Reader* newPairsReader,
			int numPairs );

		/// Append pairs to a stream.
		static void HK_CALL appendPairsToStream(
			const hknpSimulationThreadContext& tl, hkBlockStream<hknpBodyIdPair>* pairsStream,
			const hknpBodyIdPair* pairs, int numPairs );

		/// Remove any pairs from the supplied list which involve deleted bodies.
		static void HK_CALL filterDeletedPairs( hknpWorld* world, hkArray<hknpBodyIdPair>& pairs );

		/// Clear all local data. This must be called before the destructor.
		void clear( hkThreadLocalBlockStreamAllocator* tlAllocator );

		//
		//	Internal public section
		//

	public:

		//	m_newCacheStream: storing added CacheStreamEntry which will be merged to m_cacheStream later

		/// One entry for every colliding body pair. This list is sorted by bodyBId
		hknpCdCacheStream m_cdCacheStream;
		hknpCdCacheStream m_childCdCacheStream;

		/// A list of new streams coming from the broad phase or deactivation mgr.
		hknpCdCacheStream m_newCdCacheStream;
		hknpCdCacheStream m_newChildCdCacheStream;

		/// A list on inactive cd Caches.
		hknpCdCacheStream m_inactiveCdCacheStream;
		hknpCdCacheStream m_inactiveChildCdCacheStream;	// inactive collision caches

		/// A list of new streams coming from the user
		hknpCdCacheStream m_newUserCdCacheStream;
		hknpCdCacheStream m_newUserChildCdCacheStream;

		/// Pairs added by rebuildBodyPairCollisionCaches()
		hkArray<hknpBodyIdPair> m_newUserCollisionPairs;

		//
		// if simulation is a gridSimulation the next variables are used
		//

		// all data is stored in this grid
		hknpCdCacheGrid m_cdCacheGrid;

		// the next variable is not persistent, it is temporarily created at the end of the broad phase
		// for new agents and consumed in the next frame with the first collision pass .
		hknpCdCacheGrid m_newCdCacheGrid;

	#if defined(HK_PLATFORM_HAS_SPU)
		/// Grids for PPU only caches.
		hknpCdCacheGrid m_cdCachePpuGrid;
		hknpCdCacheGrid m_newCdCachePpuGrid;
	#endif
};


#endif // HKNP_COLLISION_CACHE_MANAGER_H

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
