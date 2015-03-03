/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/TempDetect/hkTempDetectAllocator.h>
#include <Common/Base/Thread/Thread/hkThread.h>
#include <Common/Base/Container/PointerMap/hkMap.cxx>
#include <Common/Base/Thread/Thread/hkThread.h>

template class hkMapBase<void*, int>;

static hkUlong foo1[8*(sizeof(hkUint64)+sizeof(hkTempDetectAllocator::AllocInfo))];
static hkUlong foo2[8*sizeof(int)];

hkTempDetectAllocator::hkTempDetectAllocator()
	: m_child(HK_NULL)
	, m_allocs(foo1, hkSizeOf(foo1))
	, m_freeFromAlloc(foo2, hkSizeOf(foo2))
	, m_outputFunc(HK_NULL)
	, m_outputFuncArg(HK_NULL)
{
}

void hkTempDetectAllocator::init(hkMemoryAllocator* child, OutputStringFunc output, void* outputUserData)
{
	m_child = child;
	m_callTree.init(child);
	m_outputFunc = output;
	m_outputFuncArg = outputUserData;
}

static hkBool32 shouldIgnoreCallstack( hkStackTracer& tracer, hkMemoryAllocator* allocator, const hkUlong* trace, int ntrace )
{
	struct HK_VISIBILITY_HIDDEN PrintArgs 
	{
		hkArrayBase<char> array;
		hkMemoryAllocator* allocator;

		PrintArgs(hkMemoryAllocator* a) : allocator(a) {}
		~PrintArgs() { array._clearAndDeallocate(*allocator); }

		static void appendToBuffer(const char* s, void* vp)
		{
			PrintArgs& args = *static_cast<PrintArgs*>(vp);
			args.array._append( *args.allocator, s, hkString::strLen(s) );
		}
	} ;
	
	PrintArgs args(allocator);
	tracer.dumpStackTrace(trace, ntrace, PrintArgs::appendToBuffer, &args);

	const char* ignoredStacks[] =
	{
		// all part of old versioning system which is going away
		"hkVersionUtil::",
		"hkBinaryPackfileReader::",
		"hkBinaryPackfileWriter::",
		"hkXmlPackfileReader::",
		"hkPackfileWriter::",
		"hkDynamicClassNameRegistry::merge",
		"hkSerializeUtil::",
	};
	for( int i = 0; i < (int)HK_COUNT_OF(ignoredStacks); ++i )
	{
		if( hkString::strStr(args.array.begin(), ignoredStacks[i]) != HK_NULL )
		{
			return true;
		}
	}
	return false;
}

void hkTempDetectAllocator::quit()
{
	if( m_allocs.getSize() && 0 ) // show leaks?
	{
		m_outputFunc("\n********************************************\n* LEAKS\n********************************************\n", m_outputFuncArg);
		for( hkMapBase<void*, TraceId>::Iterator it = m_allocs.getIterator();
			m_allocs.isValid(it); it = m_allocs.getNext(it) )
		{
			hkUlong trace[64];
			int ntrace = m_callTree.getCallStack( m_allocs.getValue(it).allocId, trace, HK_COUNT_OF(trace) );
			m_tracer.dumpStackTrace(trace, ntrace, m_outputFunc, m_outputFuncArg);
			m_outputFunc("-----------------------------------------------\n", m_outputFuncArg);
		}
	}

	hkBool32 somethingPrinted = false;
	
	//HK_ASSERT(0x70a041d9, m_allocs.getSize() == 0 );
	m_allocs.clearAndDeallocate(*m_child);
	for( hkMapBase<TraceId, TraceId>::Iterator it = m_freeFromAlloc.getIterator();
		m_freeFromAlloc.isValid(it); it = m_freeFromAlloc.getNext(it) )
	{
		if( m_freeFromAlloc.getValue(it) > 0 )
		{
			TraceId allocTraceId = m_freeFromAlloc.getKey(it);
			SizeInfo sinfo = {0,0,0,0};
			m_sizeFromAlloc.get(allocTraceId, &sinfo);
			if( sinfo.count < 5 )
			{
				continue;
			}
			hkUlong trace0[64] HK_ON_DEBUG(= {});
			int ntrace0 = m_callTree.getCallStack( allocTraceId, trace0, HK_COUNT_OF(trace0) );
			if( shouldIgnoreCallstack(m_tracer, m_child, trace0, ntrace0) == hkFalse32 )
			{
				hkUlong trace1[64] HK_ON_DEBUG(= {});
				int ntrace1 = m_callTree.getCallStack( m_freeFromAlloc.getValue(it), trace1, HK_COUNT_OF(trace1) );

				int prefix = 0; // find the common prefix of both callstacks
				while( trace0[ntrace0-1-prefix] == trace1[ntrace1-1-prefix] )
				{
					++prefix;
				}
				HK_ASSERT(0x2363c570, prefix > 0);

				m_outputFunc("\n\n", m_outputFuncArg);
				m_outputFunc("\n\n********************************************\n* Possible temp allocation \n********************************************\n", m_outputFuncArg);
				char buf[128]; hkString::snprintf(buf, sizeof(buf), "Hit %i times: min(%i), max(%i), avg(%i)\n", sinfo.count, sinfo.minSize, sinfo.maxSize, int(sinfo.total/sinfo.count) );
				m_outputFunc(buf, m_outputFuncArg);
				m_tracer.dumpStackTrace( &trace0[ntrace0-prefix-1], 1, m_outputFunc, m_outputFuncArg); // show probable alloc location first
				m_outputFunc("> alloc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n", m_outputFuncArg);
				m_tracer.dumpStackTrace(trace0, ntrace0-prefix, m_outputFunc, m_outputFuncArg);
				m_outputFunc("< free <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", m_outputFuncArg);
				m_tracer.dumpStackTrace(trace1, ntrace1-prefix, m_outputFunc, m_outputFuncArg);
				m_outputFunc("- common prefix -------------------------------\n", m_outputFuncArg);
				m_tracer.dumpStackTrace( &trace0[ntrace0-prefix], prefix, m_outputFunc, m_outputFuncArg);
				m_outputFunc("\n\n", m_outputFuncArg);

				somethingPrinted = true;
			}
		}
	}
	if( somethingPrinted == hkFalse32 )
	{
		m_outputFunc("\n********************************************\n* No temp allocations detected\n********************************************\n", m_outputFuncArg);
	}

	m_freeFromAlloc.clearAndDeallocate(*m_child);
	m_sizeFromAlloc.clearAndDeallocate(*m_child);
	m_callTree.quit();
}

void* hkTempDetectAllocator::internalAlloc(void* p, int size)
{
	HK_ASSERT(0x46b722c2, size > 0);
	AllocInfo info;
	info.threadId = hkThread::getMyThreadId();
	info.allocId = m_callTree.insertCallStack(m_tracer);
	info.size = size;
	HK_ASSERT(0x7994c3a4, m_allocs.hasKey(p) == false);
	m_allocs.insert(*m_child, p, info);
	return p;
}

void hkTempDetectAllocator::internalFree(void* p, int size)
{
	HK_ASSERT(0x25d51d15, size > 0);
	AllocInfo info = { hkUint64(-1), -1, -1};
	HK_ON_DEBUG( hkResult knownPtr = ) m_allocs.get(p, &info);
	HK_ASSERT(0x59b82f9a, knownPtr == HK_SUCCESS );
	TraceId thisFreeId = m_callTree.insertCallStack( m_tracer );
	TraceId prevFreeId = m_freeFromAlloc.getWithDefault(info.allocId, -1);
	if( prevFreeId == 0 ) // already given up on the allocation location
	{
		// skip
	}
	else if( prevFreeId == -1 ) // new free
	{
		m_freeFromAlloc.insert(*m_child, info.allocId, thisFreeId);
		SizeInfo sinfo = { size, size, size, 1 };
		m_sizeFromAlloc.insert(*m_child, info.allocId, sinfo);
	}
	else if( prevFreeId != thisFreeId || info.threadId != hkThread::getMyThreadId() )
	{
		m_freeFromAlloc.insert(*m_child, info.allocId, 0); // give up on this one
		m_sizeFromAlloc.remove( info.allocId );
	}
	else // update size totals
	{
		hkMapBase<TraceId, SizeInfo>::Iterator it = m_sizeFromAlloc.findKey(info.allocId);
		HK_ASSERT(0x59b82f9b, m_sizeFromAlloc.isValid(it));
		SizeInfo sinfo = m_sizeFromAlloc.getValue(it);
		sinfo.maxSize = hkMath::max2(sinfo.maxSize, size);
		sinfo.minSize = hkMath::max2(sinfo.minSize, size);
		sinfo.total += size;
		sinfo.count += 1;
		m_sizeFromAlloc.setValue(it, sinfo);
	}
	m_allocs.remove( p );
}

void* hkTempDetectAllocator::blockAlloc( int size ) 
{
	if (size == 0)
	{
		// every allocation must correspond to a different pointer
		size = 1;
	}
	hkCriticalSectionLock lock(&m_lock);
	return internalAlloc(m_child->blockAlloc( size ), size);
}

void hkTempDetectAllocator::blockFree( void* p, int size ) 
{
	if( p != HK_NULL )
	{		
		if (size == 0)
		{
			// use the correct size
			size = 1;
		}
		hkCriticalSectionLock lock(&m_lock);
		internalFree(p,size);
		m_child->blockFree(p, size);
	}
}

void* hkTempDetectAllocator::bufAlloc( int& size )
{
	if (size == 0)
	{
		// every allocation must correspond to a different pointer
		size = 1;
	}
	hkCriticalSectionLock lock(&m_lock);
	void* p = m_child->bufAlloc(size);
	return internalAlloc( p, size);
}

void hkTempDetectAllocator::bufFree( void* p, int size )
{
	if( p != HK_NULL )
	{
		hkCriticalSectionLock lock(&m_lock);
		internalFree(p, size);		
		m_child->bufFree(p, size);
	}
}

void* hkTempDetectAllocator::bufRealloc( void* pold, int oldNum, int& reqNumInOut )
{
	hkCriticalSectionLock lock(&m_lock);

	if( pold != HK_NULL )
	{
		hkMapBase<void*, TraceId>::Iterator it = m_allocs.findKey(pold);
		HK_ASSERT(0x67a1e3e7, m_allocs.isValid(it) );
		AllocInfo iold;
		iold = m_allocs.getValue(it);

		void* pnew = m_child->bufRealloc(pold, oldNum, reqNumInOut);
		if( pold != pnew )
		{
			m_allocs.remove( it );
			m_allocs.insert(*m_child, pnew, iold); // keep original alloc location
		}
		if( iold.threadId != hkThread::getMyThreadId() )
		{
			m_freeFromAlloc.insert(*m_child, iold.allocId, 0); // different thread, give up on this one
		}
		return pnew;
	}
	else
	{
		return bufAlloc(reqNumInOut);
	}
}

void hkTempDetectAllocator::advanceFrame()
{
	hkCriticalSectionLock lock(&m_lock);
	for( hkMapBase<void*, TraceId>::Iterator it = m_allocs.getIterator(); m_allocs.isValid(it); it = m_allocs.getNext(it) )
	{
		// give up on all allocations lasting longer than a frame
		TraceId tid = m_allocs.getValue(it).allocId;
		m_freeFromAlloc.insert(*m_child,tid,0);
	}
}

void hkTempDetectAllocator::getMemoryStatistics( hkTempDetectAllocator::MemoryStatistics& u ) const
{
	HK_WARN( 0xf0345465, "hkTempDetectAllocator does not support getMemoryStatistics(), data will be wrong");
	m_child->getMemoryStatistics( u );
}

int hkTempDetectAllocator::getAllocatedSize(const void* obj, int numBytes) const
{
	return m_child->getAllocatedSize( obj, numBytes );
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
