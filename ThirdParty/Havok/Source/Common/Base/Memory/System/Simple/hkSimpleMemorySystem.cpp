/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Simple/hkSimpleMemorySystem.h>

hkSimpleMemorySystem::hkSimpleMemorySystem() : m_frameInfo(0), m_allocators()
{
}

hkSimpleMemorySystem::MemoryAllocators::MemoryAllocators(
	hkMemoryAllocator *tempAllocator,
	hkMemoryAllocator *heapAllocator,
	hkMemoryAllocator *debugAllocator,
	hkMemoryAllocator *solverAllocator) :
	m_tempAllocator(tempAllocator),
	m_heapAllocator(heapAllocator),
	m_debugAllocator(debugAllocator),
	m_solverAllocator(solverAllocator)
{
}

void hkSimpleMemorySystem::setAllocator(hkMemoryAllocator* a) 
{
	// Ensure setAllocator is only called before initialization
	HK_ASSERT2(0x62b32838, (m_allocators.m_tempAllocator==HK_NULL && m_allocators.m_heapAllocator==HK_NULL &&
							m_allocators.m_debugAllocator==HK_NULL && m_allocators.m_solverAllocator==HK_NULL) || a == HK_NULL,
							"setAllocator called while allocators are already set" );
	m_allocators.m_tempAllocator = a;
	m_allocators.m_heapAllocator = a;
	m_allocators.m_debugAllocator = a;
	// Solver allocator has a different interface
	m_allocators.m_solverAllocator = HK_NULL;
}

void hkSimpleMemorySystem::setAllocators(MemoryAllocators& allocators)
{
	// Ensure setAllocators is only called before initialization
	HK_ASSERT2(0x40470589, (m_allocators.m_tempAllocator==HK_NULL && m_allocators.m_heapAllocator==HK_NULL &&
		m_allocators.m_debugAllocator==HK_NULL && m_allocators.m_solverAllocator==HK_NULL), "setAllocators called while allocators are already set");
	m_allocators = allocators;
}

hkSimpleMemorySystem::MemoryAllocators& hkSimpleMemorySystem::getAllocators()
{
	return m_allocators;
}

hkMemoryRouter* hkSimpleMemorySystem::mainInit(const FrameInfo& info, Flags flags)
{
	HK_ASSERT2(0x6a0086a9, m_allocators.m_solverAllocator || m_allocators.m_heapAllocator, "At least a heap allocator must be available");
	m_frameInfo = info;
	if( flags.get(FLAG_PERSISTENT) )
	{
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		int bufSize = m_frameInfo.m_solverBufferSize;
		if( bufSize && !m_allocators.m_solverAllocator )
		{
			void* p = m_allocators.m_heapAllocator->blockAlloc( bufSize );
			m_solverAllocator.setBuffer( p, bufSize );
		}
	}
	threadInit(m_mainRouter, "main", flags);
	return &m_mainRouter;
}

hkResult hkSimpleMemorySystem::mainQuit(Flags flags)
{
	threadQuit(m_mainRouter, flags);
	if( flags.get(FLAG_TEMPORARY) )
	{
		if( !m_allocators.m_solverAllocator )
		{
			int bufSize = m_solverAllocator.getBufferSize();
			m_allocators.m_heapAllocator->blockFree( m_solverAllocator.m_bufferStart, bufSize );
			m_solverAllocator.setBuffer( HK_NULL, 0 );
		}	
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		setAllocator( HK_NULL );
	}
	return HK_SUCCESS;
}

void hkSimpleMemorySystem::threadInit(hkMemoryRouter& router, const char* name, Flags flags)
{
	if( flags.get(FLAG_PERSISTENT) )
	{
		router.setTemp(HK_NULL);
		router.setHeap(m_allocators.m_heapAllocator);
		router.setDebug(m_allocators.m_debugAllocator);
		router.setSolver(HK_NULL);
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		HK_ASSERT(0x52555219, &router.heap() );
		router.stack().init(m_allocators.m_heapAllocator, m_allocators.m_heapAllocator, m_allocators.m_heapAllocator);
		router.setTemp(m_allocators.m_tempAllocator); // set up temp
		if(m_allocators.m_solverAllocator)
		{
			router.setSolver(m_allocators.m_solverAllocator); // and solver
		}
		else
		{
			router.setSolver(&m_solverAllocator);
		}
	}
}

void hkSimpleMemorySystem::threadQuit(hkMemoryRouter& router, Flags flags)
{
	if( flags.get(FLAG_TEMPORARY) )
	{
		router.setSolver(HK_NULL);
		router.setTemp(HK_NULL);
		router.stack().quit();
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		hkMemUtil::memSet( &router, 0, sizeof(router) );
	}
}

void hkSimpleMemorySystem::printStatistics(hkOstream& ostr) const
{
	hkMemoryAllocator::MemoryStatistics usage;
	m_allocators.m_tempAllocator->getMemoryStatistics(usage);
	ostr.printf("TEMP: %i in use, %i available, %i peak", usage.m_inUse, usage.m_available, usage.m_peakInUse);
	m_allocators.m_heapAllocator->getMemoryStatistics(usage);
	ostr.printf("HEAP: %i in use, %i available, %i peak", usage.m_inUse, usage.m_available, usage.m_peakInUse);
	m_allocators.m_debugAllocator->getMemoryStatistics(usage);
	ostr.printf("DEBUG: %i in use, %i available, %i peak", usage.m_inUse, usage.m_available, usage.m_peakInUse);
	if( m_allocators.m_solverAllocator )
	{
		m_allocators.m_solverAllocator->getMemoryStatistics(usage);
		ostr.printf("SOLVER: %i in use, %i available, %i peak", usage.m_inUse, usage.m_available, usage.m_peakInUse);
	}
}

hkMemoryAllocator* hkSimpleMemorySystem::getUncachedLockedHeapAllocator()
{
	return m_allocators.m_heapAllocator;
}

void hkSimpleMemorySystem::getMemoryStatistics( MemoryStatistics& stats )
{
	stats.m_entries.clear();

	// There is only one allocator. 
	MemoryStatistics::Entry& entryHeap = stats.m_entries.expandOne();
	entryHeap.m_allocatorName = "Heap";
	m_allocators.m_heapAllocator->getMemoryStatistics(entryHeap.m_allocatorStats);

}


#if 0
// Simplified and shorter version for the docs

hkMemoryRouter* hkSimpleMemorySystem::mainInit(const FrameInfo& info, Flags flags)
{
	if( flags.get(FLAG_PERSISTENT) )
	{
		// nothing extra to do
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		// set up solver buffer
	}
	// do thread local setup too
	threadInit(m_mainRouter, "main", flags);
	return &m_mainRouter;
}

hkResult hkSimpleMemorySystem::mainQuit(Flags flags)
{
	threadQuit(m_mainRouter, flags);
	if( flags.get(FLAG_TEMPORARY) )
	{
		// free solver buffer
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		// nothing
	}
	return HK_SUCCESS;
}

void hkSimpleMemorySystem::threadInit(hkMemoryRouter& router, const char* name, Flags flags)
{
	if( flags.get(FLAG_PERSISTENT) )
	{
		// setup heap, debug
		// all others set to null
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		// set up stack, temp and solver
	}
}

void hkSimpleMemorySystem::threadQuit(hkMemoryRouter& router, Flags flags)
{
	if( flags.get(FLAG_TEMPORARY) )
	{
		// quit stack, temp and solver
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		// quit heap, debug
	}
}

#endif

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
