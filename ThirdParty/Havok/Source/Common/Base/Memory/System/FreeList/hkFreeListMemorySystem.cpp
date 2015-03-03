/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/FreeList/hkFreeListMemorySystem.h>
#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>

using namespace std;

hkFreeListMemorySystem::hkFreeListMemorySystem(hkMemoryAllocator* systemAllocator, hkMemoryAllocator* heapAllocator, hkMemoryAllocator::ExtendedInterface* heapExtended, SetupFlags flags)
	: m_systemAllocator(systemAllocator)
	, m_frameInfo(0)
	, m_heapAllocator(heapAllocator)
	, m_heapExtended(heapExtended)
	, m_debugAllocator(systemAllocator)
	, m_flags(flags)
{
}

hkFreeListMemorySystem::~hkFreeListMemorySystem()
{
}

// main

hkMemoryRouter* hkFreeListMemorySystem::mainInit(const FrameInfo& info, Flags flags)
{
	m_frameInfo = info;
	if( flags.get(FLAG_PERSISTENT) )
	{
		threadInit(m_mainRouter, "main", flags);
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		HK_ASSERT2(0x70b97484, m_solverAllocator.m_bufferStart == HK_NULL, "stack already initialized");
		if( info.m_solverBufferSize )
		{
			void *buffer = m_systemAllocator->blockAlloc( info.m_solverBufferSize );
			HK_ASSERT2(0x683c5a98, buffer, "Unable to allocate solver buffer");
			HK_MEMORY_TRACKER_NEW_RAW("SolverBuffer", buffer, info.m_solverBufferSize);
			m_solverAllocator.setBuffer(buffer , info.m_solverBufferSize );
		}
	}

	return &m_mainRouter;
}

hkResult hkFreeListMemorySystem::mainQuit(Flags flags)
{
	if( flags.get(FLAG_TEMPORARY) )
	{
		if( int bufSize = m_solverAllocator.getBufferSize() )
		{
			m_systemAllocator->blockFree( m_solverAllocator.m_bufferStart, m_solverAllocator.getBufferSize() );
			HK_MEMORY_TRACKER_DELETE_RAW(m_solverAllocator.m_bufferStart);
		}		
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		threadQuit(m_mainRouter, flags);
		m_solverAllocator.setBuffer( HK_NULL, 0 );
		if (m_heapExtended)
		{
			m_heapExtended->garbageCollect();
		}
	}

	m_systemAllocator = HK_NULL;

	return HK_SUCCESS;
}

// thread

void hkFreeListMemorySystem::threadInit(hkMemoryRouter& router, const char* name, Flags flags)
{
	if( flags.get(FLAG_PERSISTENT) )
	{
		m_threadDataLock.enter();

		int threadIndex;
		for ( threadIndex = 0; threadIndex < THREAD_MAX; threadIndex++)
		{
			if ( !m_threadData[threadIndex].m_inUse )
			{
				m_threadData[threadIndex].m_inUse = true;
				break;
			}
		}

		m_threadDataLock.leave();

		
		if(threadIndex >= THREAD_MAX)
		{
			HK_ERROR(0xf03454fe,"Too many threads");
		}

		ThreadData& td = m_threadData[threadIndex];
		td.m_name = name;

		td.m_heapThreadMemory.setMemory(m_heapAllocator);

		router.setTemp(HK_NULL);

		// Use heap if thread memory not used 
		hkMemoryAllocator* threadHeapAllocator = m_heapAllocator;
		if (m_flags & USE_THREAD_MEMORY)
		{
			threadHeapAllocator = &td.m_heapThreadMemory;
		}

		router.setHeap( threadHeapAllocator );
		router.setDebug(&m_debugAllocator);
		router.setSolver(HK_NULL);
		router.setUserData( (void*)hkUlong(threadIndex) );
	}

	if( flags.get(FLAG_TEMPORARY) )
	{
		//
		//	In this system we are allocating frame local buffers from the heap memory,
		//  You should not do this in a real game, as this can complete fragment the heap.	
		//  You should use your own buffers instead.

		int threadIndex = (int)(hkUlong)router.getUserData();
		ThreadData& td = m_threadData[threadIndex];

		// Set up the temp memory. As we are using heap memory for this we are using the m_heapThreadMemory and
		// not another m_tempThreadMemory as this would lead to increased thread local cached memory blocks per frame.
		// (Note these blocks are in theory free, but only accessible by one frame, so effectively lost for the other frames).
		// don't use thread memory when detecting leaks

		// Use heap if thread memory not used 
		hkMemoryAllocator* heapAllocator = m_heapAllocator;
		if (m_flags & USE_THREAD_MEMORY)
		{
			heapAllocator = &td.m_heapThreadMemory;
		}

		// Set up the stack memory.
		hkMemoryAllocator* largeAllocator = (m_flags & USE_SOLVER_ALLOCATOR_FOR_LIFO_SLABS) ? &m_solverAllocator : heapAllocator;
		router.stack().init(largeAllocator, heapAllocator, heapAllocator);

		if (m_flags & USE_LIFO_ALLOCATOR_FOR_TEMP)
		{
			router.setTemp(&router.stack());
		}
		else
		{
			router.setTemp(heapAllocator);
		}

		router.setSolver(&m_solverAllocator);
	}
}

void hkFreeListMemorySystem::threadQuit(hkMemoryRouter& router, Flags flags)
{
	int threadIndex = (int)(hkUlong)router.getUserData();
	ThreadData& td = m_threadData[threadIndex];

	if( flags.get(FLAG_TEMPORARY) )
	{
		//td.m_tempThreadMemory.releaseCachedMemory();
		router.stack().quit();
		td.m_heapThreadMemory.releaseCachedMemory();	// lets be polite and give back the memory
		router.setTemp(HK_NULL);
		router.setSolver(HK_NULL);
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		td.m_heapThreadMemory.releaseCachedMemory();
		td.m_name = HK_NULL;
		td.m_inUse = false;

		hkMemUtil::memSet( &router, 0, sizeof(router) );
	}
}

static inline int roundedPercentage(int num, int denom)
{
	return denom
		? int( (num*100LL) / denom )
		: 0;
}

void hkFreeListMemorySystem::getMemoryStatistics(MemoryStatistics& stats)
{
	stats.m_entries.clear();

	// Base 
	MemoryStatistics::Entry& entrySys = stats.m_entries.expandOne();
	entrySys.m_allocatorName = "System";
	m_systemAllocator->getMemoryStatistics(entrySys.m_allocatorStats);

	// Heap 
	MemoryStatistics::Entry& entryHeap = stats.m_entries.expandOne();
	entryHeap.m_allocatorName = "Heap";
	m_heapAllocator->getMemoryStatistics(entryHeap.m_allocatorStats);

	// Debug 
	MemoryStatistics::Entry& entryDbg = stats.m_entries.expandOne();
	entryDbg.m_allocatorName = "Debug";
	m_debugAllocator.getMemoryStatistics(entryDbg.m_allocatorStats);

	// Solver
	MemoryStatistics::Entry& entrySolv = stats.m_entries.expandOne();
	entrySolv.m_allocatorName = "Solver";
	m_solverAllocator.getMemoryStatistics(entrySolv.m_allocatorStats);}

void hkFreeListMemorySystem::printStatistics( hkOstream& ostr) const
{
	m_threadDataLock.enter();

	ostr.printf("hkFreeListMemorySystem memory overview:\n=======================================\n");

	hkMemoryAllocator::MemoryStatistics baseStats;
	m_systemAllocator->getMemoryStatistics(baseStats);

	hkMemoryAllocator::MemoryStatistics heapStats;
	m_heapAllocator->getMemoryStatistics(heapStats);

	hkMemoryAllocator::MemoryStatistics debugStats;
	m_debugAllocator.getMemoryStatistics(debugStats);

	ostr.printf("\n    Allocation totals:\n\n");
	int heapAllocated = static_cast<int>(heapStats.m_allocated);
	ostr.printf("%20i allocated by heap\n", heapAllocated);
	int debugAllocated = static_cast<int>(debugStats.m_allocated);
	ostr.printf("%20i allocated by debug\n", debugAllocated);
	int solverAllocated = m_solverAllocator.getBufferSize();
	ostr.printf("%20i allocated by solver\n", solverAllocated);
	// todo. extract cached sizes from spu util
	ostr.printf("%20s\n", "-------");
	ostr.printf("%20i computed total\n", heapAllocated + debugAllocated + solverAllocated);
	ostr.printf("%20i reported total\n", static_cast<int>(baseStats.m_allocated));

	ostr.printf("\n    Heap usage:\n\n");
	int threadUnused = 0;

	if (m_flags & USE_THREAD_MEMORY)
	{	
		for( int i = 0; i < THREAD_MAX; ++i )
		{
			if( m_threadData[i].m_inUse )
			{
				hkMemoryAllocator::MemoryStatistics ms;
				m_threadData[i].m_heapThreadMemory.getMemoryStatistics( ms );
				threadUnused += (int) ms.m_available;
			}
		}
		ostr.printf("%20i (%2i%%) unused in thread local freelists\n", threadUnused, roundedPercentage(threadUnused, (int)heapStats.m_allocated) );
	}

	// Note that you might expect threadUnused + heapUnused == heapStats.m_inUse however this is not the case.
	// hkThreadMemory caches both large and small blocks so threadmemory is not a subset of the freelists nor
	// the largeblockAllocator, but straddles both.

	int heapUnused = (int)(heapStats.m_available!=-1 ? heapStats.m_available : 0);
	ostr.printf("%20i (%2i%%) unused in main heap\n", heapUnused, roundedPercentage(heapUnused, (int)heapStats.m_allocated) );
	int heapInUse = (int)(heapStats.m_allocated - heapUnused - threadUnused);
	ostr.printf("%20i (%2i%%) used in main heap\n", heapInUse, roundedPercentage(heapInUse, (int)heapStats.m_allocated) );
	ostr.printf("%20s\n", "-------");
	ostr.printf("%20i allocated by heap\n", int(heapStats.m_allocated) );

	ostr.printf("\n    Peak usage:\n\n");
	if( heapStats.m_peakInUse != -1 )
	{
		int peak = int(heapStats.m_peakInUse);
		ostr.printf("%20i (%2i%%) peak heap used (versus current)\n", peak, roundedPercentage(heapInUse, peak) );
	}
	ostr.printf("%20i (%2i%%) peak solver used (versus available)\n", int(m_solverAllocator.m_peakUse), roundedPercentage((int)m_solverAllocator.m_peakUse, m_solverAllocator.getBufferSize()) );

	m_threadDataLock.leave();
}

void hkFreeListMemorySystem::garbageCollectThread(hkMemoryRouter& router)
{
	if (m_flags & USE_THREAD_MEMORY)
	{
		static_cast<hkThreadMemory&>( router.heap() ).releaseCachedMemory();
	}
}

void hkFreeListMemorySystem::garbageCollectShared()
{
	if (m_heapExtended)
	{
		m_heapExtended->garbageCollect();
	}
}

hkResult hkFreeListMemorySystem::getMemorySnapshot(hkMemorySnapshot& snapshot) const
{
	if( m_heapExtended )
	{
		int sysId = snapshot.addProvider("<System>", -1);
		// solver
		int solverId = snapshot.addProvider("hkSolverAllocator(Solver)", sysId);
		snapshot.addAllocation( sysId, m_solverAllocator.m_bufferStart, m_solverAllocator.getBufferSize() );
		snapshot.addAllocation( solverId, m_solverAllocator.m_bufferStart, m_solverAllocator.getBufferSize() );
		// heap
		int heapId = m_heapExtended->addToSnapshot(snapshot, sysId);
		if(heapId == -1)
			return HK_FAILURE;
		// debug
		int debugId = snapshot.addProvider("hkRecallAllocator(Debug)", sysId);
		for( const hkRecallAllocator::Header* cur=m_debugAllocator.getHead(); cur != HK_NULL; cur=cur->getNext() )
		{
			snapshot.addAllocation( sysId, cur, cur->getRequestedSize() );
			snapshot.addOverhead( debugId, cur, hkGetByteOffsetInt(cur, cur->getPayload()) );
			snapshot.addAllocation( debugId, cur->getPayload(), cur->getPayloadSize() );
			const void* alignBegin = hkAddByteOffsetConst(cur->getPayload(), cur->getPayloadSize());
			const void* alignEnd = hkAddByteOffsetConst(cur, cur->getRequestedSize());
			if(alignBegin != alignEnd)
				snapshot.addOverhead( debugId, alignBegin, hkGetByteOffsetInt(alignBegin, alignEnd) );
		}
		snapshot.setRouterWiring(heapId, heapId, heapId, debugId, solverId);
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

hkResult hkFreeListMemorySystem::setHeapSoftLimit(int nbytes)
{
	if (m_heapExtended)
	{
		return m_heapExtended->setMemorySoftLimit(nbytes);
	}
	return HK_SUCCESS;
}

int hkFreeListMemorySystem::getHeapSoftLimit() const
{
	if (m_heapExtended)
	{
		return (int)m_heapExtended->getMemorySoftLimit();
	}
	return 0;
}

bool hkFreeListMemorySystem::solverCanAllocSingleBlock( int numBytes )
{
	return m_solverAllocator.canAllocSingleBlock(numBytes);
}

bool hkFreeListMemorySystem::heapCanAllocTotal( int numBytes )
{
	if (m_heapExtended)
	{
		return m_heapExtended->canAllocTotal(numBytes);
	}
	return true;
}

hkMemoryAllocator* hkFreeListMemorySystem::getUncachedLockedHeapAllocator()
{
	return m_heapAllocator;
}

void hkFreeListMemorySystem::setHeapScrubValues(hkUint32 allocValue, hkUint32 freeValue)
{
	if(m_heapExtended)
	{
		m_heapExtended->setScrubValues(allocValue, freeValue);
	}
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
