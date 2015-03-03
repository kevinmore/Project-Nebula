/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Optimizer/hkOptimizerMemorySystem.h>

hkOptimizerMemorySystem::hkOptimizerMemorySystem()
: m_frameInfo(0), m_baseAllocator(HK_NULL)
{
}

void hkOptimizerMemorySystem::init
	( hkMemoryAllocator* a
	, OutputStringFunc output
	, void* outputUserData
	, hkOptimizerMemorySystem::DetectFlags detect )
{
	m_outputFunc = output;
	m_outputFuncArg = outputUserData;
	m_baseAllocator = a;
	m_detect = detect;
}

hkMemoryRouter* hkOptimizerMemorySystem::mainInit(const FrameInfo& info, Flags flags)
{
	m_frameInfo = info;
	if( flags.get(FLAG_PERSISTENT) )
	{
		m_tempDetector.init(m_baseAllocator, m_outputFunc, m_outputFuncArg);
		for( int i = 0; i < THREAD_MAX; ++i )
		{
			m_threadData[i].m_lifoChecker.m_tracer = &m_tempDetector.m_tracer;
		}
		threadInit(m_mainRouter, "main", flags);
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		if( int bufSize = m_frameInfo.m_solverBufferSize )
		{
			void* p = m_baseAllocator->blockAlloc( bufSize );
			m_solverAllocator.setBuffer( p, bufSize );
		}		
	}
	return &m_mainRouter;
}

hkResult hkOptimizerMemorySystem::mainQuit(Flags flags)
{
	if( flags.get(FLAG_TEMPORARY) )
	{
		if( int bufSize = m_solverAllocator.getBufferSize() )
		{
			m_baseAllocator->blockFree( m_solverAllocator.m_bufferStart, bufSize );
			m_solverAllocator.setBuffer( HK_NULL, 0 );
		}
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		threadQuit(m_mainRouter, flags);
		m_tempDetector.quit();
	}

	return HK_SUCCESS;
}

void hkOptimizerMemorySystem::threadInit(hkMemoryRouter& router, const char* name, Flags flags)
{
	if( flags.get(FLAG_PERSISTENT) )
	{
		router.setTemp(HK_NULL);
		router.setHeap(m_detect.get(DETECT_TEMP) ? &m_tempDetector : m_baseAllocator);
		router.setDebug(m_baseAllocator);
		router.setSolver(HK_NULL);
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		HK_ASSERT(0x52555219, &router.heap() );
		ThreadData& data = newThreadData();
		router.setUserData(&data);

		data.m_lifoChecker.init(m_baseAllocator, m_outputFunc, m_outputFuncArg);
		router.stack().init(m_baseAllocator, m_baseAllocator, m_baseAllocator);
		router.setTemp( m_detect.get(DETECT_LIFO) ? &data.m_lifoChecker : m_baseAllocator);
		router.setSolver(&m_solverAllocator); // and solver
	}
}

void hkOptimizerMemorySystem::threadQuit(hkMemoryRouter& router, Flags flags)
{
	ThreadData& data = *reinterpret_cast<ThreadData*>(router.getUserData());
	if( flags.get(FLAG_TEMPORARY) )
	{
		router.stack().quit();
		router.setTemp(HK_NULL);
		router.setSolver(HK_NULL);
		data.m_lifoChecker.quit();
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		hkMemUtil::memSet( &router, 0, sizeof(router) );
		hkCriticalSectionLock lock(&m_threadDataCriticalSection);
		data.m_inUse = false;
	}
}

void hkOptimizerMemorySystem::printStatistics(hkOstream& ostr) const
{
	hkMemoryAllocator::MemoryStatistics usage;
	m_baseAllocator->getMemoryStatistics(usage);
	ostr.printf("%i in use, %i available, %i peak", usage.m_inUse, usage.m_available, usage.m_peakInUse);
}

void hkOptimizerMemorySystem::getHeapStatistics(hkMemoryAllocator::MemoryStatistics& stats)
{
	m_baseAllocator->getMemoryStatistics(stats);
}

void hkOptimizerMemorySystem::getMemoryStatistics( MemoryStatistics& stats )
{
	stats.m_entries.clear();

	// Heap 
	MemoryStatistics::Entry& entrySys = stats.m_entries.expandOne();
	entrySys.m_allocatorName = "Heap";
	m_baseAllocator->getMemoryStatistics(entrySys.m_allocatorStats);
	
	// Solver
	MemoryStatistics::Entry& entrySolv = stats.m_entries.expandOne();
	entrySolv.m_allocatorName = "Solver";
	m_solverAllocator.getMemoryStatistics(entrySolv.m_allocatorStats);
	
	// Temp
	MemoryStatistics::Entry& entryDbg = stats.m_entries.expandOne();
	entryDbg.m_allocatorName = "Temp";
	m_tempDetector.getMemoryStatistics(entryDbg.m_allocatorStats);
}

hkOptimizerMemorySystem::ThreadData& hkOptimizerMemorySystem::newThreadData()
{
	hkCriticalSectionLock lock(&m_threadDataCriticalSection);
	for (int i = 0; i < THREAD_MAX; i++)
	{
		ThreadData& data = m_threadData[i];
		if (!data.m_inUse)
		{
			data.m_inUse = true;
			return m_threadData[i];
		}
	}

	HK_ASSERT2(0x1e9bfe78, false, "Too many threads");
	return m_threadData[0];
}

hkMemoryAllocator* hkOptimizerMemorySystem::getUncachedLockedHeapAllocator()
{
	return m_detect.get(DETECT_TEMP) ? &m_tempDetector : m_baseAllocator;
}

#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkUlong,hkTempDetectAllocator::AllocInfo>;
template class hkMap<hkUlong,hkTempDetectAllocator::AllocInfo>;

void* hkOptimizerMemorySystem::LifoCheckAllocator::blockAlloc( int numBytes )
{
	void* p = m_allocator->blockAlloc(numBytes);
	m_allocs._pushBack(*m_allocator, p);
	return p;
}
void hkOptimizerMemorySystem::LifoCheckAllocator::blockFree( void* p, int numBytes )
{
	if( p == HK_NULL )
	{
		return;
	}
	if( m_allocs.back() != p )
	{
		{
			int i = m_allocs.indexOf(p);
			HK_ASSERT2(0x75969044, i >= 0, "not from this allocator");
			m_allocs[i] = HK_NULL;
		}

		int numNonLifos = 0;
		for( int i = m_allocs.indexOf(HK_NULL); i >= 0; i = m_allocs.indexOf(HK_NULL,i+1) )
		{
			++numNonLifos;
		}
		// To avoid spamming, let's just report once "on the way up"
		if( numNonLifos == 5 )
		{
			m_outputFunc("\n********************************************\n* NONLIFO\n********************************************\n", m_outputFuncArg);
			m_outputFunc("Several non lifo frees found at this point. Perhaps you should be calling hkArray.reserve\n" \
				"This warning will not be repeated until the count of non-lifo frees becomes small again\n", m_outputFuncArg);
			hkUlong trace[128];
			int ntrace = m_tracer->getStackTrace( trace, HK_COUNT_OF(trace) );
			m_tracer->dumpStackTrace(trace, ntrace, m_outputFunc, m_outputFuncArg);
			m_outputFunc("-----------------------------------------------\n", m_outputFuncArg);
		}
	}
	else
	{
		m_allocator->blockFree(p, numBytes);
		m_allocs.popBack();
		while( m_allocs.getSize() && m_allocs.back()==HK_NULL )
		{
			m_allocs.popBack();
		}
	}
}

void* hkOptimizerMemorySystem::LifoCheckAllocator::bufRealloc( void* pold, int oldNumBytes, int& reqNumBytesInOut )
{
	if( m_allocs.back() == pold )
	{
		void* pnew = m_allocator->bufAlloc(reqNumBytesInOut);
		hkMemUtil::memCpy(pnew, pold, hkMath::min2(oldNumBytes,reqNumBytesInOut));
		m_allocator->bufFree(pold, oldNumBytes);
		m_allocs.back() = pnew;
		return pnew;
	}
	else
	{
		return hkMemoryAllocator::bufRealloc(pold, oldNumBytes, reqNumBytesInOut);
	}
}

void hkOptimizerMemorySystem::LifoCheckAllocator::getMemoryStatistics( MemoryStatistics& u ) const
{
	return m_allocator->getMemoryStatistics(u);
}

int hkOptimizerMemorySystem::LifoCheckAllocator::getAllocatedSize(const void* obj, int nbytes) const
{
	return m_allocator->getAllocatedSize(obj, nbytes);
}
void hkOptimizerMemorySystem::LifoCheckAllocator::init(hkMemoryAllocator* base, OutputStringFunc func, void* funcArg)
{
	HK_ASSERT(0x3318d85f, m_allocs.getSize()==0 && m_allocs.getCapacity()==0);
	m_allocator = base;
	m_outputFunc = func;
	m_outputFuncArg = funcArg;
}

void hkOptimizerMemorySystem::LifoCheckAllocator::quit()
{
	m_allocs._clearAndDeallocate(*m_allocator); m_allocator = HK_NULL;
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
