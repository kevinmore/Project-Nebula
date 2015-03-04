/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Checking/hkLeakDetectAllocator.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>

void hkLeakDetectAllocator::init(hkMemoryAllocator* child, hkMemoryAllocator* debug, OutputStringFunc output, void* outputUserData )
{
	m_childAllocator = child;
	m_debugAllocator = debug;
	m_outputFunc = output;
	m_outputFuncArg = outputUserData;
	m_inUse = 0;

	m_dynamic = new (debug->blockAlloc(sizeof(Dynamic))) Dynamic(child);
	m_callTree.init(debug);
}

void hkLeakDetectAllocator::quit()
{
	if( m_dynamic->m_activePointers.getSize() != 0 )
	{
		m_outputFunc(
			"**************************************************************\n" \
			"* Begin hkLeakDetectorAllocator                              *\n" \
			"**************************************************************\n", m_outputFuncArg);

		hkArrayBase<AllocInfo> allocs; allocs._reserve( *m_debugAllocator, m_dynamic->m_activePointers.getSize() );
		for( MapType::Iterator it = m_dynamic->m_activePointers.getIterator();
			m_dynamic->m_activePointers.isValid(it);
			it = m_dynamic->m_activePointers.getNext(it) )
		{
			allocs.pushBackUnchecked( m_dynamic->m_activePointers.getValue(it) );
		}
		hkSort( allocs.begin(), allocs.getSize() );

		for( int i = 0; i < allocs.getSize(); ++i )
		{
			if( allocs[i].traceId < 0 )
			{
				continue;
			}
			int count = 1; // kill duplicates with the same callstack
			for( int j = i+1; j < allocs.getSize(); ++j )
			{
				if( allocs[j].traceId == allocs[i].traceId )
				{
					count += 1;
					allocs[j].traceId = -1;
				}
			}
			// todo: output count too
			danger( "Memory Leaked:\n", HK_NULL, allocs[i] );
		}
		allocs._clearAndDeallocate(*m_debugAllocator);

		{
			char buf[256];
			hkString::snprintf(buf, HK_COUNT_OF(buf), "\n%i leaks found\n", m_dynamic->m_activePointers.getSize() );
			m_outputFunc(buf, m_outputFuncArg);
		}

		m_outputFunc(
			"**************************************************************\n" \
			"* End hkLeakDetectorAllocator                                *\n" \
			"**************************************************************\n", m_outputFuncArg);
	}
	m_callTree.quit();
	m_dynamic->quit(m_debugAllocator);
	m_dynamic->~Dynamic();
	m_debugAllocator->blockFree(m_dynamic, sizeof(Dynamic));
}

void* hkLeakDetectAllocator::checkedAlloc( hkBool32 isBuf, int numBytes )
{
	hkCriticalSectionLock lock( &m_criticalSection );

	void* p = m_childAllocator->blockAlloc(numBytes);
	int allocatedSize = m_childAllocator->getAllocatedSize( p, numBytes );
	m_inUse += allocatedSize;

	hkLeakDetectAllocator::AllocInfo info;
	info.time = hkStopwatch::getTickCounter();
	info.size = numBytes;
	info.isBuf = isBuf!=hkFalse32;
	{
		hkUlong trace[128];
		int ntrace = m_stackTracer.getStackTrace(trace, HK_COUNT_OF(trace));
		info.traceId = m_callTree.insertCallStack(trace, ntrace);
	}
	m_dynamic->m_activePointers.insert( *m_debugAllocator, hkUlong(p), info);
	return p;
}

void hkLeakDetectAllocator::checkedFree( hkBool32 isBuf, void* p, int numBytes )
{
	hkCriticalSectionLock lock( &m_criticalSection );

// 	{
// 		MemoryStatistics u;	m_childAllocator->getMemoryStatistics(u);
// 		HK_ASSERT( 0xf0435456, 0 == ((u.m_inUse - m_inUse) % 0x32000));
// 	}

	if( p != HK_NULL && numBytes == 0 )
	{
		m_outputFunc("Given zero bytes to free with an non-null address\n", m_outputFuncArg);
		HK_BREAKPOINT(0);
		return;
	}
	else if( p == HK_NULL && numBytes != 0)
	{
		m_outputFunc("Given nonzero bytes to free with an null address\n", m_outputFuncArg);
		HK_BREAKPOINT(0);
		return;
	}
	else if( p == HK_NULL && numBytes == 0)
	{
		return;
	}

	MapType::Iterator it = m_dynamic->m_activePointers.findKey( hkUlong(p) );
	if( m_dynamic->m_activePointers.isValid(it) == false )
	{
		m_outputFunc("Freeing block not from this allocator", m_outputFuncArg);
		HK_BREAKPOINT(0);
	}
	AllocInfo info = m_dynamic->m_activePointers.getValue(it);
	if( info.size != numBytes )
	{
		danger("Freeing block of incorrect size", p,info);
		HK_BREAKPOINT(0);
	}
	m_callTree.releaseCallStack( info.traceId );
	m_dynamic->m_activePointers.remove(it);

	int allocatedSize = m_childAllocator->getAllocatedSize( p, numBytes );
	m_inUse -= allocatedSize;

	m_childAllocator->blockFree(p, numBytes);

// 	{
// 		MemoryStatistics u;	m_childAllocator->getMemoryStatistics(u);
// 		HK_ASSERT( 0xf0435456, 0 == ((u.m_inUse - m_inUse) % 0x32000));
// 	}
}


void* hkLeakDetectAllocator::blockAlloc( int numBytes )
{
	return checkedAlloc(false, numBytes );
}

void hkLeakDetectAllocator::blockFree( void* p, int numBytes )
{
	checkedFree( false, p, numBytes );
}

void* hkLeakDetectAllocator::bufAlloc( int& reqNumInOut )
{
	return checkedAlloc( true, reqNumInOut );
}

void hkLeakDetectAllocator::bufFree( void* p, int num )
{
	checkedFree( true, p, num );
}

void* hkLeakDetectAllocator::bufRealloc( void* pold, int oldNum, int& reqNumInOut )
{
	void* pnew = checkedAlloc( true, reqNumInOut );
	hkMemUtil::memCpy( pnew, pold, hkMath::min2(reqNumInOut,oldNum) );
	checkedFree( true, pold, oldNum );
	return pnew;
}

void hkLeakDetectAllocator::danger(const char* message, const void* ptr, const hkLeakDetectAllocator::AllocInfo& info) const
{
	char buf[256];
	hkString::snprintf(buf, HK_COUNT_OF(buf), "\nAddress=0x%p size=%i\n", ptr, info.size );
	(*m_outputFunc)(message, m_outputFuncArg);
	(*m_outputFunc)(buf, m_outputFuncArg);
	hkUlong trace[32];
	int numTrace = m_callTree.getCallStack( info.traceId, trace, HK_COUNT_OF(trace));
	m_stackTracer.dumpStackTrace( trace, numTrace, m_outputFunc, m_outputFuncArg );
	m_outputFunc("-------------------------------------------------------------------\n\n", m_outputFuncArg );
}


void hkLeakDetectAllocator::getMemoryStatistics( hkLeakDetectAllocator::MemoryStatistics& u ) const
{
	m_childAllocator->getMemoryStatistics( u );
}

int hkLeakDetectAllocator::getAllocatedSize(const void* obj, int numBytes) const
{
	return m_childAllocator->getAllocatedSize( obj, numBytes );
}

static void writeToStream( const char* s, void* args )
{
	static_cast<hkStreamWriter*>(args)->write(s, hkString::strLen(s));
}

	//
static void printRecursive
	( hkOstream& os
	, int cur // current node and totals index
	, int level // depth of current node (could compute it more slowly from nodes)
	, hkStackTracer& stackTracer
	, const hkArrayBase<hkStackTracer::CallTree::Node>& nodes
	, const hkArrayBase<hkUint64>& totals
	)
{
	hkStreamWriter* sw = os.getStreamWriter();
	{
		int nleft = level;
		static const char tabs[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
		const int numTabs = HK_COUNT_OF(tabs)-1; // not trailing null
		while( nleft )
		{
			int n = hkMath::min2(nleft, numTabs);
			sw->write(tabs, n);
			nleft -= n;
		}
	}
	os.printf("%i ", totals[cur] );
	
	if( cur )
	{
		// This could be obscenely slow if the stack tracer spawns an external command for each trace
		// If it is a problem, we could batch the trace calls like is done in the memory report
		stackTracer.dumpStackTrace( &nodes[cur].m_value, 1, writeToStream, sw);
	}
	else
	{
		os.printf("\n"); // root node does not have a trace
	}

	for( int c = nodes[cur].m_firstChild; c != -1; c = nodes[c].m_next )
	{
		printRecursive(os, c, level+1, stackTracer, nodes, totals);
	}
}


void hkLeakDetectAllocator::printAllocationsByCallTree(hkOstream& os)
{
	hkCriticalSectionLock lock( &m_criticalSection );

	// total the sizes for each branch - totals & nodes use the same indices
	const hkArrayBase<hkStackTracer::CallTree::Node>& nodes = m_callTree.getNodes();
	hkArray<hkUint64>::Debug totals; totals.setSize( nodes.getSize(), 0 );

	// sum the totals from each location into their parents
	MapType& active = m_dynamic->m_activePointers;
	for( MapType::Iterator it = active.getIterator(); active.isValid(it); it = active.getNext(it) )
	{
		AllocInfo info = active.getValue(it);
		int tid = info.traceId;
		while( tid >= 0 )
		{
			totals[tid] += info.size;
			tid = nodes[tid].m_parent;
		}
	}

	// strip out common callstack prefix - advance down the tree while there's only one child
	int cur = 0;
	while( true )
	{
		int c = nodes[cur].m_firstChild;
		if( c > 0 && nodes[c].m_next < 0 )
		{
			cur = c;
		}
		else
		{
			break;
		}
	}

	hkStackTracer stackTracer;
	printRecursive( os, cur, 0, stackTracer, nodes, totals );
}



#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkUlong,hkLeakDetectAllocator::AllocInfo>;
template class hkMap<hkUlong,hkLeakDetectAllocator::AllocInfo>;

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
