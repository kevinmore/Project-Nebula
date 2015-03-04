/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CACHE_SORTER_H
#define HKNP_CACHE_SORTER_H

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>


/// Helper class to sort collision caches into grids. Used by hknpMultithreadedSimulation.
class hknpCacheSorter
{
	public:

		/// Sort a cachestream into gridcells. Does not touch the childCdCaches. On PS3 an additional grid must be
		/// provided for PPU only caches.
		/// Note if gridOut == HK_NULL, no sorting into a grid will happen.
		static void HK_CALL sortCaches(
			const hknpSimulationThreadContext& tl, hknpCdCacheStream* cacheStreamInOut, hknpCdCacheGrid* gridOut,
			hknpCdCacheGrid* gridPpuOut = HK_NULL);

		/// Merge all the caches in a grid into a single stream deterministically
		static void mergeCdCacheRanges(
			const hknpSimulationThreadContext& tl,
			hknpCdCacheGrid& cdCacheGridIn, hknpCdCacheStream& streamOut );

		/// Copy all deactivated caches into the cache stream in the deactivated island
		static void deactivateCdCacheRanges(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hkArray<hknpBodyId>& deactivatedBodies, hknpCdCacheGrid** inactiveCdCacheGrids, int numGrids,
			hknpCdCacheStream& mergedChildCdCacheStreamIn, bool useExclusiveCdCacheRanges,
			hknpCdCacheStream& streamOut, hknpCdCacheStream& childCdCacheStreamOut, hkArray<hknpBodyIdPair>& deactivatedDeletedCachesOut,
			hknpCdCacheRange& newInactiveCdCachesRange, hknpCdCacheRange& newInactiveChildCdCachesRange );
};


#endif	// HKNP_CACHE_SORTER_H

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
