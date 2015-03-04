/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/Simulation/Utils/hknpCacheSorter.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/World/Deactivation/CdCacheFilter/hknpDeactiveCdCacheFilter.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>


namespace
{
struct CacheInfo
{
	public:

		// Functor used to order based only on the body IDs
		struct Less
		{
			HK_FORCE_INLINE hkBool32 operator() (const CacheInfo& a, const CacheInfo& b)
			{
				HK_ASSERT( 0xf0dc5fff, a.m_linkIndex == b.m_linkIndex );
				return a.m_bodyPair < b.m_bodyPair;
			}
		};

	public:

		HK_FORCE_INLINE friend hkBool32 operator < (const CacheInfo& a, const CacheInfo& b)
		{
			if( a.m_linkIndex != b.m_linkIndex )
			{
				return a.m_linkIndex < b.m_linkIndex;
			}
			else
			{
				// we need to use bodyPair to keep a deterministic sorting, as quicksort does
				// not keep the order for equal elements. Also this is necessary for mesh collisions
				// and helps with c
				return a.m_bodyPair < b.m_bodyPair;
			}
		}

	public:

		int m_linkIndex;
		hknpBodyIdPair m_bodyPair;
		const void* m_data;
};
}


static void hknpCacheSorter_sortCdCaches( CacheInfo* HK_RESTRICT data, int numData, int maxKey )
{
	if ( numData < 50 )
	{
		hkAlgorithm::quickSort( data, numData );
	}
	else	// radix sort,
	{
		hkLocalBuffer<int> sizesBuffer(maxKey);
		hkLocalBuffer<CacheInfo> copyBuffer(numData);
		int* HK_RESTRICT sizes = &sizesBuffer[0];
		CacheInfo* HK_RESTRICT copy = &copyBuffer[0];

		for (int i =0; i < maxKey; i++)
		{
			sizes[i] = 0;
		}

		// count sizes
		for (int i=0; i < numData; i++)
		{
			sizes[ data[i].m_linkIndex ]++;
			copy[i] = data[i];
		}

		// convert sizes to indices
		int index = 0;
		for (int i =0; i < maxKey; i++)
		{
			int k = sizes[i];
			sizes[i] = index;
			index += k;
		}

		// output
		{
			for(int i=0;i<numData;++i)
			{
				data[sizes[copy[i].m_linkIndex]++] = copy[i];
			}
		}

		// sort output
		{
			int base = 0;
			for (int i =0; i < maxKey; i++)
			{
				int lastIndex = sizes[i];
				int size = lastIndex - base;
				if ( size )
				{
					hkAlgorithm::quickSort( data + base, size, CacheInfo::Less() );
				}
				base = lastIndex;
			}
		}
	}
}


void HK_CALL hknpCacheSorter::sortCaches(
	const hknpSimulationThreadContext& threadContext, hknpCdCacheStream* HK_RESTRICT cacheStreamInOut,
	hknpCdCacheGrid* HK_RESTRICT gridOut, hknpCdCacheGrid* HK_RESTRICT gridPpuOut)
{
	if ( cacheStreamInOut->isEmpty() )
	{
		return;
	}
	hknpWorld* world = threadContext.m_world;

	//HK_TIMER_BEGIN_LIST("SortAndBuild", "Prepare");

	hknpSpaceSplitter* splitter = world->m_spaceSplitter;
	hkLocalArray<CacheInfo> cacheInfos( cacheStreamInOut->getTotalNumElems() );
#if defined(HK_PLATFORM_HAS_SPU)
	// On PS3 why need to provide an output grid for PPU only caches unless we are running single threaded.
	HK_ASSERT(0x16862ae0, gridPpuOut || !gridOut);

	hkLocalArray<CacheInfo> cachePpuInfos( cacheStreamInOut->getTotalNumElems() );
#endif

	// Fill in cache infos
	{
		hknpCdCacheReader cacheReader; cacheReader.setToStartOfStream( cacheStreamInOut );
		const hknpCollisionCache* cache = cacheReader.access();

		while ( cache )
		{
			// Select destination array according to cache type (normal or PPU)
			CacheInfo* HK_RESTRICT cacheInfo;
		#if !defined(HK_PLATFORM_HAS_SPU)
			cacheInfo = cacheInfos.expandByUnchecked(1);
		#else
			if (!cache->m_spuFlags.anyIsSet(hknpBody::FORCE_NARROW_PHASE_PPU))
			{
				cacheInfo = cacheInfos.expandByUnchecked(1);
			}
			else
			{
				cacheInfo = cachePpuInfos.expandByUnchecked(1);
			}
		#endif

			// Fill in cache info
			hknpBodyId aIdx = cache->m_bodyA;
			hknpBodyId bIdx = cache->m_bodyB;
			int cIdxA = world->m_bodyManager.getCellIndex(aIdx);
			int cIdxB = world->m_bodyManager.getCellIndex(bIdx);
			cacheInfo->m_bodyPair.m_bodyA = aIdx;
			cacheInfo->m_bodyPair.m_bodyB = bIdx;
			cacheInfo->m_data = cache;

			// If there is no output grid we are running single threaded and therefore we only use link index 0
			int linkIndex = gridOut ? splitter->getLinkIdx( cIdxA, cIdxB ) : 0;
			cacheInfo->m_linkIndex = linkIndex;

			cache = cacheReader.advanceAndAccessNext( cache->getSizeInBytes() );
		}
	}

	// Create stream and writer for the sorted caches
	hknpCdCacheStream cacheStreamOut;
	cacheStreamOut.initBlockStream( threadContext.m_heapAllocator );
	hknpCdCacheWriter cacheWriter;
	cacheWriter.setToStartOfStream( threadContext.m_heapAllocator, &cacheStreamOut );

#if !defined(HK_PLATFORM_HAS_SPU)
	HK_ASSERT(0x33d61bef, cacheInfos.getSize());
	hkLocalArray<CacheInfo>* infos[] = { &cacheInfos };
	hknpCdCacheGrid* grids[] = { gridOut };
	for (int i = 0; i < 1; ++i)
#else
	HK_ASSERT(0x33d61bef, cacheInfos.getSize() || cachePpuInfos.getSize());
	hkLocalArray<CacheInfo>* infos[] = { &cacheInfos, &cachePpuInfos };
	hknpCdCacheGrid* grids[] = { gridOut, gridPpuOut };
	for (int i = 0; i < 2; ++i)
#endif
	{
		hknpCdCacheGrid* HK_RESTRICT currentGrid = grids[i];
		hkLocalArray<CacheInfo>* HK_RESTRICT currentCacheInfos = infos[i];
		if (currentCacheInfos->getSize() == 0)
		{
			continue;
		}

		//HK_TIMER_SPLIT_LIST("Sort");
		if ( currentGrid )
		{
			hknpCacheSorter_sortCdCaches( currentCacheInfos->begin(), currentCacheInfos->getSize(), splitter->getNumLinks() );
		}
		else
		{
			hkAlgorithm::quickSort( currentCacheInfos->begin(), currentCacheInfos->getSize(), CacheInfo::Less() );
		}
		//HK_TIMER_SPLIT_LIST("ResortCaches");

		int linkIndex = (currentGrid)? -1 : 0;
		hknpBodyIdPair lastPair; lastPair.m_bodyA = lastPair.m_bodyB = hknpBodyId::invalid();
		hkBlockStreamBase::Range* gridEntry = HK_NULL;

		for (int j = 0; j < currentCacheInfos->getSize(); j++)
		{
			CacheInfo& cacheInfo = (*currentCacheInfos)[j];

			// remove duplicated entries
			if ( cacheInfo.m_bodyPair == lastPair )
			{
				continue;
			}
			if( linkIndex != cacheInfo.m_linkIndex )
			{
				linkIndex = cacheInfo.m_linkIndex;
				if ( gridEntry )
				{
					gridEntry->setEndPoint( &cacheWriter );
				}
				gridEntry = &currentGrid->m_entries[ linkIndex ];
				HK_ASSERT( 0x3f6785b9, gridEntry->isEmpty() );

				cacheWriter.reserve( HKNP_MAX_CVX_CVX_CACHE_SIZE );
				gridEntry->setStartPoint( &cacheWriter );
			}
			const hknpCollisionCache* src = (const hknpCollisionCache*)cacheInfo.m_data;
			lastPair = cacheInfo.m_bodyPair;
			cacheWriter.write16( src );
		}
		if ( gridEntry )
		{
			gridEntry->setEndPoint( &cacheWriter );
		}
	}

	cacheWriter.finalize();
	cacheStreamInOut->clearAndSteal( threadContext.m_heapAllocator, &cacheStreamOut );

	//HK_TIMER_END_LIST();
}


void hknpCacheSorter::mergeCdCacheRanges( const hknpSimulationThreadContext& tl,  hknpCdCacheGrid& bodyPairGridIn, hknpCdCacheStream& streamOut )
{
	hknpCdCacheReader reader;
	hknpCdCacheWriter writer;
	writer.setToEndOfStream( tl.m_heapAllocator, &streamOut );
	for (int i = 0; i < bodyPairGridIn.m_entries.getSize(); i++)
	{
		hkBlockStreamBase::Range& range = bodyPairGridIn.m_entries[i];
		if ( range.isEmpty())
		{
			continue;
		}
		reader.setToRange( &range );

		//
		// Copy the range
		//
		for (const hknpCollisionCache* cache = reader.access(); cache; cache = reader.advanceAndAccessNext( cache ))
		{
			writer.write16( cache );
		}
	}
	writer.finalize();
}


void hknpCacheSorter::deactivateCdCacheRanges(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const hkArray<hknpBodyId>& deactivatedBodies,
	hknpCdCacheGrid** inactiveCdCacheGrids, int numGrids, hknpCdCacheStream& mergedInactiveChildCdCacheStreamIn, bool useExclusiveCdCacheRanges,
	hknpCdCacheStream& streamOut, hknpCdCacheStream& childCdCacheStreamOut, hkArray<hknpBodyIdPair>& deactivatedDeletedCachesOut,
	hknpCdCacheRange& newInactiveCdCachesRange, hknpCdCacheRange& newInactiveChildCdCachesRange
	)
{
	hknpCdCacheReader reader;
	hknpCdCacheWriter writer;		writer.setToEndOfStream( tl.m_heapAllocator, &streamOut );
	hknpCdCacheWriter childWriter;	childWriter.setToEndOfStream( tl.m_heapAllocator, &childCdCacheStreamOut);


	if (useExclusiveCdCacheRanges)
	{
		newInactiveCdCachesRange.setStartPointExclusive(&writer);
		newInactiveChildCdCachesRange.setStartPointExclusive(&childWriter);
	}
	else
	{
		newInactiveCdCachesRange.setStartPoint(&writer);
		newInactiveChildCdCachesRange.setStartPoint(&childWriter);
	}

	// Deactivate the caches in the grids
	hknpDeactiveCdCacheFilter* deactiveCdCacheFilter = tl.m_world->m_deactiveCdCacheFilter;
	for (int gridIndex = 0; gridIndex < numGrids; ++gridIndex)
	{
		hknpCdCacheGrid* grid = inactiveCdCacheGrids[gridIndex];
		for (int entryIndex = 0; entryIndex < grid->m_entries.getSize(); entryIndex++)
		{
			hkBlockStreamBase::Range& caches = grid->m_entries[entryIndex];
			if ( caches.isEmpty())
			{
				continue;
			}
			reader.setToRange( &caches );
			deactiveCdCacheFilter->deactivateCaches(tl, sharedData, deactivatedBodies, reader, mergedInactiveChildCdCacheStreamIn, writer, childWriter, deactivatedDeletedCachesOut);
		}
	}

	if (useExclusiveCdCacheRanges)
	{
		newInactiveCdCachesRange.setEndPointExclusive(&writer);
		newInactiveChildCdCachesRange.setEndPointExclusive(&childWriter);
	}
	else
	{
		newInactiveCdCachesRange.setEndPoint(&writer);
		newInactiveChildCdCachesRange.setEndPoint(&childWriter);
	}

	childWriter.finalize();
	writer.finalize();
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
