/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/Checking/hkCheckingMemorySystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Memory/Allocator/Thread/hkThreadMemory.h>
#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>
#include <Common/Base/Thread/Thread/hkThread.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>

// In order to generate a proper call stack for the checking memory system
// Disable frame pointer optimization for this file only. This will increase
// code size for this file slightly.

// In order for customer code to get a reliable call stack it must contain
// frame pointers. Even with the FP, the call stack that is left after optimization
// may be different to that contained in the source code
#if ( defined(HK_COMPILER_MSVC) && defined(HK_ARCH_IA32) ) && (HK_COMPILER_MSVC_VERSION < 1700)
#pragma optimize("y", off)
#endif

//
// Forwarding
//

void* hkCheckingMemorySystem::AllocatorForwarder::blockAlloc( int numBytes )
{
	return m_parent->checkedAlloc( false, m_context, numBytes );
}

void hkCheckingMemorySystem::AllocatorForwarder::blockFree( void* p, int numBytes )
{
	if( p )
	{
		m_parent->checkedFree( false, m_context, p, numBytes );
	}
	else
	{
		HK_ASSERT(0x2c2527ed, numBytes == 0 );
	}
}

void* hkCheckingMemorySystem::AllocatorForwarder::bufAlloc( int& reqNumInOut )
{
	return m_parent->checkedAlloc( true, m_context, reqNumInOut );
}

void hkCheckingMemorySystem::AllocatorForwarder::bufFree( void* p, int num )
{
	m_parent->checkedFree( true, m_context, p, num );
}

void* hkCheckingMemorySystem::AllocatorForwarder::bufRealloc( void* pold, int oldNum, int& reqNumInOut )
{
	void* pnew = m_parent->checkedAlloc( true, m_context, reqNumInOut );
	hkMemUtil::memCpy( pnew, pold, hkMath::min2(reqNumInOut,oldNum) );
	m_parent->checkedFree( true, m_context, pold, oldNum );
	return pnew;
}

void hkCheckingMemorySystem::AllocatorForwarder::getMemoryStatistics( MemoryStatistics& u ) const
{
	u.m_available = 0;
	u.m_inUse		= m_parent->m_currentInUse;
	u.m_allocated	= u.m_inUse;
	u.m_peakInUse	= m_parent->m_peakInUse;
}

int hkCheckingMemorySystem::AllocatorForwarder::getAllocatedSize( const void* obj, int numBytes ) const
{
	return numBytes;
}

void hkCheckingMemorySystem::AllocatorForwarder::resetPeakMemoryStatistics()
{
	m_parent->m_peakInUse = m_parent->m_currentInUse;
}

//
// hkCheckingMemorySystem
//

hkCheckingMemorySystem::hkCheckingMemorySystem()
	: m_baseAllocator(HK_NULL)
	, m_debugAllocator(HK_NULL)
	, m_callTreeAllocator(HK_NULL)
	, m_checkFlags(CHECK_DEFAULT)
	, m_outputFunc(HK_NULL)
	, m_outputFuncArg(HK_NULL)
	, m_frameInfo(0)
{
	m_timeOfConstruction = hkStopwatch::getTickCounter();
	m_sumAllocatedStackMemory = 0;

	m_allocOrder = 0;
}

void hkCheckingMemorySystem::init(hkMemoryAllocator* raw, OutputStringFunc output, void* outputUserData, CheckBits checks)
{
	// Need stack tracer to out live the singleton quit.
	m_checkFlags = checks;
	m_outputFunc = output;
	m_outputFuncArg = outputUserData;
	m_rawAllocator = raw;
	hkMemoryAllocator* cooked = raw;
	if( m_checkFlags.get(CHECK_DELAYED_FREE) )
	{
		m_delayedFreeAllocator.init(cooked);
		cooked = &m_delayedFreeAllocator;
	}
	if( m_checkFlags.get(CHECK_PAD_BLOCK) )
	{
		m_paddedAllocator.init(cooked);
		cooked = &m_paddedAllocator;
	}
	if( m_checkFlags.get(CHECK_CALLSTACK) )
	{
		m_callTreeAllocator.init(raw);
		m_callTree.init(&m_callTreeAllocator);
	}
	m_baseAllocator = cooked;
	m_debugAllocator.init(raw);
}

hkBool hkCheckingMemorySystem::isInit()
{
	return m_baseAllocator != HK_NULL;
}

static const char* flagsToString(char* buf, hkCheckingMemorySystem::AllocatorFlags flags)
{
	struct Pair { int val; const char* str; };
	Pair pairs[] =
	{
		{hkCheckingMemorySystem::ALLOCATOR_HEAP, "Heap"},
		{hkCheckingMemorySystem::ALLOCATOR_TEMP, "Temp"},
		{hkCheckingMemorySystem::ALLOCATOR_SOLVER, "Solver"},
	};

	buf[0] = 0;
	char* cur = buf;
	for( unsigned i = 0; i < HK_COUNT_OF(pairs); ++i )
	{
		Pair& p = pairs[i];
		if( flags.get(p.val) )
		{
			if( cur != buf)
			{
				*cur++ = '|';
			}
			hkString::strCpy(cur, p.str);
			cur += hkString::strLen(p.str);
		}
	}
	return buf;
}

// For debugging refcount issues uncomment these lines
//#include <Common/Base/Reflection/hkClass.h>

void hkCheckingMemorySystem::danger(const char* message, const void* ptr, const hkCheckingMemorySystem::AllocInfo& info) const
{
	hkCriticalSectionLock lock( &m_section );

	char buf[256];
//	if( info.m_tagData != HK_NULL )
//	{
//		hkString::snprintf(buf, HK_COUNT_OF(buf), "Object %s\n", static_cast<const hkClass*>(info.m_tagData)->getName() );
//		(*m_outputFunc)(buf, m_outputFuncArg);
//	}
	char flagString[128];
	hkReal timeOfAlloc = hkStopwatch::secondsFromTicks( info.m_time - m_timeOfConstruction );
	hkString::snprintf(buf, HK_COUNT_OF(buf), "\nAddress=0x%p size=%i flags='%s' thread=" HK_PRINTF_FORMAT_UINT64 " time=%.2f\n", ptr, info.m_size, flagsToString(flagString, info.m_flags), info.m_threadId, timeOfAlloc );
	(*m_outputFunc)(message, m_outputFuncArg);
	(*m_outputFunc)(buf, m_outputFuncArg);
	hkUlong trace[32];
	int numTrace = m_callTree.getCallStack( info.m_traceId, trace, HK_COUNT_OF(trace));
	m_stackTracer.dumpStackTrace( trace, numTrace, m_outputFunc, m_outputFuncArg );
	m_outputFunc("-------------------------------------------------------------------\n\n", m_outputFuncArg );
}

namespace // anonymous
{

struct SummaryAllocator
{
	hkMemoryAllocator& get(const void*) { return *s_allocator; }
	static hkMemoryAllocator* s_allocator;
};

/* static */ hkMemoryAllocator* SummaryAllocator::s_allocator;

class allocTimeComparison
{
public:
	allocTimeComparison( const hkArrayBase<hkCheckingMemorySystem::AllocInfo>& allocs )
		: m_allocs( allocs )
	{
	}

	HK_FORCE_INLINE hkBool32 operator() ( const int& a, const int& b )
	{
		return m_allocs[a].m_time < m_allocs[b].m_time;
	}

protected:

	const hkArrayBase<hkCheckingMemorySystem::AllocInfo>& m_allocs;
};

} // anonymous


void hkCheckingMemorySystem::leakReportByTime()
{
	SummaryAllocator::s_allocator = m_rawAllocator;

	// Flatten the map into arrays for sorting
	const int numObjs = m_activePointers.getSize();
	hkArray<hkUlong, SummaryAllocator> addrs; addrs.reserve(numObjs);
	hkArray<AllocInfo, SummaryAllocator> allocs; allocs.reserve(numObjs);
	hkArray<int, SummaryAllocator> hitCount; hitCount.reserve(numObjs);

	hkPointerMap<hkUint64, int, SummaryAllocator> duplicateStack;
	duplicateStack.reserve(numObjs);
	{
		for(MapType::Iterator it = m_activePointers.getIterator();
			m_activePointers.isValid(it);
			it = m_activePointers.getNext(it) )
		{
			AllocInfo info = m_activePointers.getValue(it);
			
			if( info.m_traceId == -1 && info.m_bookmarkIndex == -1)
			{
				// If we are on a platform without reliable callstacks,
				// or callstacks area disabled and there are no bookmarks
				// at least report the allocation address
				addrs.pushBackUnchecked( m_activePointers.getKey(it) );
				allocs.pushBackUnchecked( m_activePointers.getValue(it) );
				hitCount.pushBackUnchecked(1);
			}
			else
			{
				hkUint64 key = info.getKey();
				hkPointerMap<hkUint64, int>::Temp::Iterator dupIt = duplicateStack.findOrInsertKey(key, -1 );
				int prevIndex = duplicateStack.getValue(dupIt);
				if( prevIndex == -1 ) // Not seen this trace before
				{
					duplicateStack.setValue(dupIt, addrs.getSize() );
					addrs.pushBackUnchecked( m_activePointers.getKey(it) );
					allocs.pushBackUnchecked( m_activePointers.getValue(it) );
					hitCount.pushBackUnchecked(1);
				}
				else // seen the stack before, we'll take only the earliest one
				{
					hitCount[prevIndex] += 1;
					if( info.m_time < allocs[prevIndex].m_time )
					{
						allocs[prevIndex] = info;
						addrs[prevIndex] = m_activePointers.getKey(it);
					}
				}
			}
		}
	}

	// Sort indirectly using indices to save copying structs around
	hkArray<int, SummaryAllocator> indices; indices.setSize( allocs.getSize() );
	for( int i = 0; i < indices.getSize(); ++i )
	{
		indices[i] = i;
	}
	allocTimeComparison cmp( allocs );
	hkAlgorithm::quickSort<int,allocTimeComparison>( indices.begin(), indices.getSize(), cmp );

	for( int indexIndex = 0; indexIndex < indices.getSize(); indexIndex++ )
	{
		int index = indices[ indexIndex ];
		int preBookmarkIndex = allocs[index].m_bookmarkIndex; //findPrecedingBookmark( allocs[index].m_time )
		int postBookmarkIndex = preBookmarkIndex+1;

		bool startIndexValid = (preBookmarkIndex >= 0 && preBookmarkIndex < m_bookmarks.getSize());
		bool endIndexValid = (postBookmarkIndex >= 0 && postBookmarkIndex < m_bookmarks.getSize());
		
		if (startIndexValid || endIndexValid) // don't bother if both invalid
		{
			const char* startBookmark = startIndexValid ? m_bookmarks[preBookmarkIndex].m_name : "(unknown)";
			const char* endBookmark = endIndexValid ? m_bookmarks[postBookmarkIndex].m_name : "(unknown)";

			m_outputFunc("Between bookmarks\n[", m_outputFuncArg);
			m_outputFunc(startBookmark, m_outputFuncArg);
			m_outputFunc("] and \n[", m_outputFuncArg);
			m_outputFunc(endBookmark, m_outputFuncArg);
			m_outputFunc("]\n", m_outputFuncArg);
		}

		hkStringBuf sb; sb.printf("Memory Leaked (%i times with this stack):\n", hitCount[index] );
		danger( sb.cString(), (void*)addrs[index], allocs[index] );
	}
}

static void _breadthFirstSearch(int startIndex, const hkArrayBase<int>& connectTo, const hkArrayBase<int>& connectStart, hkArrayBase<hkBool>& visited, hkArray<int, SummaryAllocator>& owned)
{
	int index = owned.getSize();

	visited[startIndex] = true;
	owned.pushBack(startIndex);

	for (; index < owned.getSize(); index++)
	{
		const int curIndex = owned[index];
		const int* children = connectTo.begin() + connectStart[curIndex];
		const int numChildren = connectStart[curIndex + 1] - connectStart[curIndex];

		for (int i = 0; i < numChildren; i++)
		{
			const int childIndex = children[i];
			if (!visited[childIndex])
			{
				visited[childIndex] = true;
				owned.pushBack(childIndex);
			}
		}
	}
}

static void _setVisited(const hkArrayBase<int>& owned, hkArrayBase<hkBool>& visited, hkBool value)
{
	for (int i = 0; i < owned.getSize(); i++)
	{
		visited[owned[i]] = value;
	}
}

static int _findOldest(const hkArrayBase<int>& owned, const hkArrayBase<hkCheckingMemorySystem::AllocInfo>& allocs)
{
	hkUint64 oldest = allocs[owned[0]].m_allocOrder;
	int oldestIndex = 0;

	for (int i = 1; i < owned.getSize(); i++)
	{
		const int index = owned[i];

		if (allocs[index].m_allocOrder < oldest)
		{
			oldestIndex = i;
			oldest = allocs[index].m_allocOrder;
		}
	}

	return oldestIndex;
}

void hkCheckingMemorySystem::dumpLeak(const hkArrayBase<int>& owned, const hkArrayBase<hkUlong>& addrs, const hkArrayBase<AllocInfo>& allocs)
{
	hkCriticalSectionLock lock( &m_section );

	// Dump out the root object
	char buffer[128];
	hkString::snprintf(buffer, HK_COUNT_OF(buffer), "ROOT - %d reached", owned.getSize() - 1);

	danger(buffer, (const void*)addrs[owned[0]], allocs[owned[0]]);

	{
		for (int j = 1; j < owned.getSize(); j++)
		{
			const int childIndex = owned[j];
			const void* ptr = (const void*)addrs[childIndex];
			const AllocInfo& info = allocs[childIndex];

			char flagString[128];
			hkString::snprintf(buffer, HK_COUNT_OF(buffer), "REACHED Address=0x%p size=%i flags='%s' thread=%i\n", ptr, info.m_size, flagsToString(flagString, info.m_flags), info.m_threadId );
			(*m_outputFunc)(buffer, m_outputFuncArg);
		}
	}
}

void hkCheckingMemorySystem::leakReportByOwnership()
{
	SummaryAllocator::s_allocator = m_rawAllocator;

	hkArray<hkUlong, SummaryAllocator> addrs;
	hkArray<AllocInfo, SummaryAllocator> allocs;
	hkArray<int, SummaryAllocator> connectTo;
	hkArray<int, SummaryAllocator> connectStart;

	const int numObjs = m_activePointers.getSize();
	addrs.setSize(numObjs);
	allocs.setSize(numObjs);

	{
		int i = 0;
		for(MapType::Iterator it = m_activePointers.getIterator();
			m_activePointers.isValid(it);
			it = m_activePointers.getNext(it), i++)
		{
			addrs[i] = m_activePointers.getKey(it);
			allocs[i] = m_activePointers.getValue(it);
		}
	}

	// Work out all of the connections
	{
		for (int i = 0; i < numObjs; i++)
		{
			connectStart.pushBack(connectTo.getSize());

			const hkUlong addr = addrs[i];
			const AllocInfo& info = allocs[i];

			void** objPtrs = (void**)addr;
			const int numPtrs = info.m_size / int(sizeof(void*));

			for (int j = 0; j < numPtrs; j++)
			{
				hkUlong scanPtr = hkUlong(objPtrs[j]);

				for (int k = 0; k < numObjs; k++ )
				{
					const hkUlong checkPtr = addrs[k];
					const AllocInfo& checkInfo = allocs[k];

					if (scanPtr >= checkPtr && scanPtr < checkPtr + checkInfo.m_size)
					{
						// Assume its a link
						connectTo.pushBack(k);
					}
				}
			}
		}
		connectStart.pushBack(connectTo.getSize());
	}

	hkArray<hkBool, SummaryAllocator> visited;
	visited.setSize(numObjs, false);

	// Any node that isn't indexed in the connect to list can be a root
	// I do a search from it... and find what it connects to

	{
		hkArray<hkBool, SummaryAllocator> isRoot;
		isRoot.setSize(numObjs, true);

		for (int i = 0; i < connectTo.getSize(); i++)
		{
			isRoot[connectTo[i]] = false;
		}

		// Traverse from the roots

		{
			hkArray<int, SummaryAllocator> owned;
			for (int i = 0; i < numObjs; i++)
			{
				if (isRoot[i])
				{
					// Find the children
					owned.clear();
					_breadthFirstSearch(i, connectTo, connectStart, visited, owned);

					// Dump the leak
					dumpLeak(owned, addrs, allocs);			
				}
			}
		}
	}

	// Any remaining unused nodes must be in cycles. How to choose where to start from? 

	{
		hkArray<int, SummaryAllocator> best;
		hkArray<int, SummaryAllocator> owned;

		do 
		{
			best.clear();

			for (int i = 0; i < numObjs; i++)
			{
				if (!visited[i])
				{
					// I go through all the remaining nodes, doing a depth first search, and I take the solution which has the highest
					// amount of nodes
					owned.clear();
					_breadthFirstSearch(i, connectTo, connectStart, visited, owned);
					_setVisited(owned, visited, false);

					if (owned.getSize() >= best.getSize())
					{
						// see if the root is older
						if (best.getSize() == owned.getSize())
						{
							if (allocs[owned[0]].m_allocOrder < allocs[best[0]].m_allocOrder)
							{
								best.swap(owned);
							}
						}
						else
						{
							best.swap(owned);
						}
					}
				}
			}

			if (best.getSize() > 0)
			{
				// Mark as visited
				_setVisited(best, visited, true);

				// Dump 
				dumpLeak(best, addrs, allocs);
			}

		} while (best.getSize() > 0);
	}
}

hkResult hkCheckingMemorySystem::quit()
{
	hkResult result =  HK_SUCCESS;

	hkCriticalSectionLock lock( &m_section );

	if( m_allocators.getSize() != 0 )
	{
		m_outputFunc("A thread did not clean up its allocators.\n", m_outputFuncArg);
		//HK_BREAKPOINT(0);
	}

	if( m_activePointers.getSize() != 0 )
	{
		result = HK_FAILURE;

		m_outputFunc(
			"**************************************************************\n" \
			"* BEGIN MEMORY LEAK REPORT                                   *\n" \
			"**************************************************************\n", m_outputFuncArg);

		leakReportByTime();
		//leakReportByOwnership();

		m_outputFunc(
			"**************************************************************\n" \
			"* END MEMORY LEAK REPORT                                     *\n" \
			"**************************************************************\n", m_outputFuncArg);
		//HK_BREAKPOINT(0);
	}
	else
	{
		m_outputFunc(
			"**************************************************************\n" \
			"* NO HAVOK MEMORY LEAKS FOUND                                *\n" \
			"**************************************************************\n", m_outputFuncArg);
	}

	m_activePointers.clearAndDeallocate(*m_rawAllocator);

	{
		for (int bmi=0; bmi<m_bookmarks.getSize(); bmi++)
		{
			m_bookmarks[bmi].clear(m_rawAllocator);
		}
		m_bookmarks._clearAndDeallocate(*m_rawAllocator);
	}

	if( m_checkFlags.get(CHECK_CALLSTACK) )
	{
		m_callTree.quit();
	}

	m_paddedAllocator.quit();
	m_delayedFreeAllocator.quit();
	m_allocators._clearAndDeallocate(*m_rawAllocator);
	m_baseAllocator = HK_NULL;

	return result;
}

hkMemoryAllocator* hkCheckingMemorySystem::getUncachedLockedHeapAllocator()
{
	return &m_mainRouter.heap();
}

hkResult hkCheckingMemorySystem::getAllocationCallStack(const void* ptr, hkUlong* callStack, int& stackSize, hk_size_t& allocSize)
{
	const MapType& map = m_activePointers;

	MapType::Iterator iter = map.findKey(hkUlong(ptr));
	if (map.isValid(iter))
	{
		const AllocInfo& info = map.getValue(iter);
		allocSize = info.m_size;

		if (callStack)
		{
			stackSize = m_callTree.getCallStack( info.m_traceId, callStack, stackSize);
		}
		else
		{
			stackSize = m_callTree.getCallStackSize(info.m_traceId);
		}
		return HK_SUCCESS;
	}

	return HK_FAILURE;
}

void hkCheckingMemorySystem::setHeapScrubValues(hkUint32 allocValue, hkUint32 freeValue)
{
	hkCriticalSectionLock lock( &m_section );
	// change the values in the padded allocator (so that if it is used we will get the proper scrubbing)
	m_paddedAllocator.setScrubValues(allocValue, freeValue);
}

hkCheckingMemorySystem::AllocatorForwarder* hkCheckingMemorySystem::newAllocator(AllocatorFlags flags, hkUint64 tid )
{
	hkCriticalSectionLock lock( &m_section );
	AllocatorForwarder* a = new (m_rawAllocator->blockAlloc( sizeof(AllocatorForwarder))) AllocatorForwarder();
	a->m_parent = this;
	a->m_context.flags = flags;
	a->m_context.threadId = tid;
	m_allocators._pushBack( *m_rawAllocator, a );
	return a;
}


void hkCheckingMemorySystem::deleteAllocator( hkMemoryAllocator* a )
{
	for( int i = 0; i < m_allocators.getSize(); ++i )
	{
		if( m_allocators[i] == a )
		{
			m_allocators[i]->~AllocatorForwarder();
			m_rawAllocator->blockFree( m_allocators[i], sizeof(AllocatorForwarder) );
			m_allocators.removeAt(i);
			return;
		}
	}
	HK_BREAKPOINT(0);
}


void* hkCheckingMemorySystem::checkedAlloc( hkBool32 isBuf, const hkCheckingMemorySystem::AllocationContext& context, int numBytes )
{
	hkCriticalSectionLock lock( &m_section );

	if( numBytes < 0 )
	{
		m_outputFunc("Negative size to allocate\n", m_outputFuncArg);
		HK_BREAKPOINT(0);
	}

	if( context.flags.get(ALLOCATOR_TEMP) )
	{
		context.curInUse += numBytes;
		context.peakInUse = hkMath::max2( context.peakInUse, context.curInUse );
	}
	else if( context.flags.get(ALLOCATOR_SOLVER) ) 
	{
		// solver allocs are often released in another thread so we can't update curInUse
		// however there's only one per thread so we can use this shortcut for the peak.
		context.peakInUse = hkMath::max2( context.peakInUse, numBytes );
	}

	AllocInfo info;
	info.m_threadId = context.threadId;
	info.m_flags = context.flags;
	info.m_size = numBytes;
	info.m_locks = 0;
	info.m_tagData = HK_NULL;
	info.m_allocOrder = m_allocOrder++;
	info.m_time = hkStopwatch::getTickCounter();
	info.m_bookmarkIndex = m_bookmarks.getSize() - 1;

	void* p = m_baseAllocator->blockAlloc(numBytes);

	if( m_checkFlags.get(CHECK_CALLSTACK) )
	{
		info.m_traceId = m_callTree.insertCallStack(m_stackTracer);
	}
	else
	{
		info.m_traceId = -1;
	}
	m_activePointers.insert( *m_rawAllocator, hkUlong(p), info);
	m_currentInUse += numBytes;
	m_peakInUse = hkMath::max2( m_currentInUse, m_peakInUse);
	return p;
}

void hkCheckingMemorySystem::checkedFree( hkBool32 isBuf, const hkCheckingMemorySystem::AllocationContext& context, void* p, int numBytes )
{
	hkCriticalSectionLock lock( &m_section );

	if( context.flags.get(ALLOCATOR_TEMP) )
	{
		context.curInUse -= numBytes;
	}

	if( p == HK_NULL && numBytes != 0)
	{
		m_outputFunc("Given nonzero bytes to free with an null address\n", m_outputFuncArg);
		HK_BREAKPOINT(0);
		return;
	}
	else if( p == HK_NULL && numBytes == 0)
	{
		return;
	}

	MapType::Iterator it = m_activePointers.findKey( hkUlong(p) );
	if( m_activePointers.isValid(it) == false )
	{
		m_outputFunc("Freeing block not from this allocator", m_outputFuncArg);
		HK_BREAKPOINT(0);
	}
	AllocInfo info = m_activePointers.getValue(it);
	if( info.m_size != numBytes )
	{
		danger("Freeing block of incorrect size", p,info);
		HK_BREAKPOINT(0);
	}
	if( info.m_flags != context.flags )
	{
		danger("Freeing block with different tag", p,info);
		HK_BREAKPOINT(0);
	}
	if( context.flags.get(ALLOCATOR_TEMP) && info.m_threadId != context.threadId )
	{
		danger("Freeing block from a different thread", p,info);
		HK_BREAKPOINT(0);
	}
	if( info.m_locks != 0 )
	{
		danger("Freeing locked block", p, info);
		HK_BREAKPOINT(0);
	}
	if( m_checkFlags.get(CHECK_CALLSTACK) )
	{
		m_callTree.releaseCallStack( info.m_traceId );
	}
	m_activePointers.remove(it);
	m_currentInUse -= info.m_size;
	m_baseAllocator->blockFree(p, numBytes);
}

hkBool32 hkCheckingMemorySystem::isOk() const
{
	hkCriticalSectionLock lock( const_cast<hkCriticalSection*>(&m_section) );

	hkBool32 ok = true;
	if( m_checkFlags.get(CHECK_PAD_BLOCK) )
	{
		for( MapType::Iterator it = m_activePointers.getIterator();
			m_activePointers.isValid(it);
			it = m_activePointers.getNext(it) )
		{
			const void* p = (const void*)m_activePointers.getKey(it);
			AllocInfo info = m_activePointers.getValue(it);
			if( m_paddedAllocator.isOk(p,info.m_size) == hkFalse32 )
			{
				danger( "Damaged block:\n", p, info);
				ok = false;
			}
		}
	}
	if( m_checkFlags.get(CHECK_DELAYED_FREE) )
	{
		if( m_delayedFreeAllocator.isOk() == hkFalse32 )
		{
			ok = false;
		}
	}
	return ok;
}

hkResult hkCheckingMemorySystem::getMemorySnapshot(hkMemorySnapshot& snapshot) const 
{
	hkCriticalSectionLock lock( &m_section );

	hkMemorySnapshot::ProviderId systemId = snapshot.addProvider("<System>", -1);
	hkMemorySnapshot::ProviderId callTreeId = snapshot.addProvider("hkStatsAllocator(CallTree)", systemId);
	hkMemorySnapshot::ProviderId heapId = systemId;
	if( m_checkFlags.get(CHECK_DELAYED_FREE) )
	{
		heapId = m_delayedFreeAllocator.addToSnapshot(snapshot, heapId);
	}
	if( m_checkFlags.get(CHECK_PAD_BLOCK) )
	{
		heapId = snapshot.addProvider("hkPaddedAllocator", heapId);
	}
	hkMemorySnapshot::ProviderId debugId = snapshot.addProvider("hkRecallAllocator", systemId);
	hkMemorySnapshot::ProviderId cmsId = snapshot.addProvider("hkCheckingMemorySystem");
	snapshot.addParentProvider(cmsId, heapId);
	snapshot.addParentProvider(cmsId, callTreeId);
	snapshot.setRouterWiring(cmsId, cmsId, cmsId, debugId, cmsId);
	const MapType& active = m_activePointers;
	for( MapType::Iterator it = active.getIterator(); active.isValid(it); it = active.getNext(it) )
	{
		const void* ptr = reinterpret_cast<const void*>(active.getKey(it));
		int size = active.getValue(it).m_size;
		if( m_checkFlags.get(CHECK_PAD_BLOCK) )
		{
			hkPaddedAllocator::Allocation a = m_paddedAllocator.getUnderlyingAllocation(ptr, size);
			snapshot.addAllocation( systemId, a.address, a.size ); // add system allocation
			if(a.address != ptr)
			{
				snapshot.addOverhead( heapId, a.address, hkGetByteOffsetInt(a.address, ptr) ); // overhead before
			}
			const void* sysEnd = hkAddByteOffsetConst(a.address, a.size);
			const void* padEnd = hkAddByteOffsetConst(ptr, size);
			if( sysEnd != padEnd )
			{
				snapshot.addOverhead( heapId, padEnd, hkGetByteOffsetInt(padEnd, sysEnd) ); // overhead after (alignment included)
			}
		}
		else
		{
			snapshot.addAllocation( systemId, ptr, size ); // add system allocation
		}

		hkMemorySnapshot::AllocationId aid = snapshot.addAllocation( cmsId, ptr, size );
		if( m_checkFlags.get(CHECK_CALLSTACK) )
		{
			hkUlong stack[128];
			int ns = m_callTree.getCallStack( active.getValue(it).m_traceId, stack, HK_COUNT_OF(stack) );
			snapshot.setCallStack( aid, stack, ns );
		}
	}
	const void* ptrCallTree =  m_callTree.getNodes().begin();
	int sizeCallTree = m_callTree.getNodes().getCapacity()*sizeof(hkStackTracer::CallTree::Node);
	snapshot.addAllocation( callTreeId, ptrCallTree, sizeCallTree );
	for( const hkRecallAllocator::Header* cur=m_debugAllocator.getHead(); cur != HK_NULL; cur=cur->getNext() )
	{
		snapshot.addAllocation( systemId, cur, cur->getRequestedSize() );
		snapshot.addOverhead( debugId, cur, hkGetByteOffsetInt(cur, cur->getPayload()) );
		snapshot.addAllocation( debugId, cur->getPayload(), cur->getPayloadSize() );
		const void* alignBegin = hkAddByteOffsetConst(cur->getPayload(), cur->getPayloadSize());
		const void* alignEnd = hkAddByteOffsetConst(cur, cur->getRequestedSize());
		if(alignBegin != alignEnd)
			snapshot.addOverhead( debugId, alignBegin, hkGetByteOffsetInt(alignBegin, alignEnd) );
	}
	snapshot.addOverhead( cmsId, ptrCallTree, sizeCallTree );
	snapshot.addOverhead( cmsId, m_allocators.begin(), m_allocators.getCapacity()*sizeof(AllocatorForwarder*) );
	for( int i = 0; i < m_allocators.getSize(); ++i )
	{
		snapshot.addOverhead( cmsId, m_allocators[i], sizeof(AllocatorForwarder) );
	}
	snapshot.addOverhead( cmsId, m_activePointers.getMemStart(), m_activePointers.getMemSize() );
	return HK_SUCCESS;
}


const void* hkCheckingMemorySystem::findBaseAddress(const void* pquery, int nbytes)
{
	hkCriticalSectionLock lock( &m_section );
	
	const void* addressNotFound = (const void*)(hkDebugMemorySystem::ADDRESS_NOT_FOUND);

	// try to use p as a base address
	const MapType& active = m_activePointers;
	{
		MapType::Iterator it = active.findKey( hkUlong(pquery) );
		if ( active.isValid(it) )
		{
			AllocInfo info = active.getValue(it);
			if (nbytes <= info.m_size)
			{
				return pquery;
			}
			else
			{
				return addressNotFound; // Invalid block, bigger than actual allocation
			}
		}
	}

	// try to walk backward from p
	{
		void* p = (void*)(HK_NEXT_MULTIPLE_OF(16, hkUlong(pquery)));
		for( int n = 0; n < 50; n++ )
		{
			p = hkAddByteOffset(p, -16);

			MapType::Iterator it = active.findKey( hkUlong(p) );
			if( active.isValid(it) )
			{
				const AllocInfo info = active.getValue(it);
				if ( hkAddByteOffsetConst(pquery, nbytes) <= hkAddByteOffsetConst(p, info.m_size) )
				{
					return p;
				}
				else
				{
					return addressNotFound; // Invalid block
				}
			}
		}
	}

	// search the full list
	{
		for( MapType::Iterator it = active.getIterator();
			active.isValid(it);
			it = active.getNext(it) )
		{
			void* p = (void*)active.getKey(it);
			if ( p <= pquery )
			{
				AllocInfo info = active.getValue(it);
				if( hkAddByteOffsetConst(pquery, nbytes) <= hkAddByteOffsetConst(p, info.m_size) )
				{
					return p;
				}
			}
		}
	}

	// Couldn't find the allocation.
	return addressNotFound;
}

void hkCheckingMemorySystem::lockBaseAddress(const void* p)
{
	hkCriticalSectionLock lock( &m_section );

	MapType& active = m_activePointers;
	MapType::Iterator it = active.findKey( hkUlong(p) );
	HK_ASSERT(0x5f7f696b, active.isValid(it) );
	AllocInfo info = active.getValue(it);
	info.m_locks += 1; 
	active.setValue( it, info );
}

void hkCheckingMemorySystem::unlockBaseAddress(const void* p)
{
	hkCriticalSectionLock lock( &m_section );

	MapType& active = m_activePointers;
	MapType::Iterator it = active.findKey( hkUlong(p) );
	HK_ASSERT(0x5f7f696b, active.isValid(it) );
	AllocInfo info = active.getValue(it);
	HK_ASSERT(0x52186a8f, info.m_locks > 0);
	info.m_locks -= 1; 
	active.setValue( it, info );
}

void hkCheckingMemorySystem::tagAddress(const void* baseAddress, const void* data)
{
	MapType& active = m_activePointers;
	MapType::Iterator it = active.findKey( hkUlong(baseAddress) );
	HK_ASSERT(0x5f7f696b, active.isValid(it) );
	AllocInfo info = active.getValue(it);
	info.m_tagData = data;
	active.setValue(it, info);
}

#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase<hkUlong,hkCheckingMemorySystem::AllocInfo>;
template class hkMap<hkUlong,hkCheckingMemorySystem::AllocInfo>;







hkMemoryRouter* hkCheckingMemorySystem::mainInit(const FrameInfo& info, Flags flags)
{
	m_frameInfo = info;
	if( flags.get(FLAG_PERSISTENT) )
	{
		threadInit(m_mainRouter, "main", flags);
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		//nothing
	}
	return &m_mainRouter;
}

hkResult hkCheckingMemorySystem::mainQuit(Flags flags)
{
	if( flags.get(FLAG_TEMPORARY) )
	{
		// nothing
	}
	if( flags.get(FLAG_PERSISTENT) )
	{
		threadQuit(m_mainRouter, flags);
		return this->quit();
	}

	return HK_SUCCESS;
}

void hkCheckingMemorySystem::threadInit(hkMemoryRouter& router, const char* name, Flags flags)
{
	hkCriticalSectionLock lock( &m_section );

	hkUint64 threadId = hkThread::getMyThreadId();

	if( flags.get(FLAG_PERSISTENT) )
	{
		hkMemoryAllocator* heap = newAllocator(ALLOCATOR_HEAP, threadId);
		router.setHeap(heap);
		router.setDebug(&m_debugAllocator);
		router.setTemp(HK_NULL);
		router.setSolver(HK_NULL);
	}
	if( flags.get(FLAG_TEMPORARY) )
	{
		// stack
		{
			AllocatorForwarder* stack = newAllocator(ALLOCATOR_STACK, threadId);
			router.stack().init(stack, stack, stack);
		}

		// temp
		{
			AllocatorForwarder* temp = newAllocator(ALLOCATOR_TEMP, threadId);
			router.setTemp(temp);
		}

		// solver
		{
			AllocatorForwarder* solver = newAllocator(ALLOCATOR_SOLVER, threadId);
			router.setSolver(solver);
		}
	}
}

void hkCheckingMemorySystem::threadQuit(hkMemoryRouter& router, Flags flags)
{
	hkCriticalSectionLock lock(&m_section);
	
	if (flags.get(FLAG_TEMPORARY))
	{
		// clear stack
		hkMemoryAllocator* allocators[3] = {HK_NULL, HK_NULL, HK_NULL};
		router.stack().quit(allocators);
		deleteAllocator(allocators[0]); // the three elements will be exactly the same
		// clear temp
		deleteAllocator(&router.temp());
		router.setTemp(HK_NULL);
		// clear solver
		deleteAllocator(&router.solver());
		router.setSolver(HK_NULL);
	}
	if (flags.get(FLAG_PERSISTENT))
	{	
		// clear heap
		deleteAllocator(&router.heap());
		hkMemUtil::memSet(&router, 0, sizeof(router));
	}
}

void hkCheckingMemorySystem::getMemoryStatistics(MemoryStatistics& stats)
{
	stats.m_entries.clear();

	// Base 
	MemoryStatistics::Entry& entrySys = stats.m_entries.expandOne();
	entrySys.m_allocatorName = "System";
	m_baseAllocator->getMemoryStatistics(entrySys.m_allocatorStats);

	// Heap 
	MemoryStatistics::Entry& entryHeap = stats.m_entries.expandOne();
	entryHeap.m_allocatorName = "Heap";
	m_paddedAllocator.getMemoryStatistics(entryHeap.m_allocatorStats);

	// Debug
	MemoryStatistics::Entry& entryDbg = stats.m_entries.expandOne();
	entryDbg.m_allocatorName = "Debug";
	m_debugAllocator.getMemoryStatistics(entryDbg.m_allocatorStats);
}

void hkCheckingMemorySystem::printStatistics(hkOstream& ostr) const
{
	hkCriticalSectionLock lock( &m_section );

	ostr.printf("hkCheckingMemorySystem memory overview:\n=======================================\n");

	// We deliberately don't use threadmemory block sizes so we don't hide any rounding
	const int MAX_SMALL = hkThreadMemory::MEMORY_MAX_SIZE_SMALL_BLOCK;
	const int NUM_BINS_SMALL = MAX_SMALL / 16;
	const int LOG2_LARGE = 10; // log2 ( nextPowerOf2( MAX_SMALL) )
	int countSmall[NUM_BINS_SMALL] = {};
	int sumsSmall[NUM_BINS_SMALL] = {};
	int countLarge[32] = {};
	hkLong totalLarge = 0;

	int heapAllocated = 0;
	int debugAllocated = 0;
	int stackAllocated = 0;
	int tempAllocated = 0;
	int solverAllocated = 0;

	hkMemoryAllocator::MemoryStatistics debugStats;
	m_debugAllocator.getMemoryStatistics(debugStats);
	debugAllocated = static_cast<int>(debugStats.m_allocated);
	{
		const MapType& active = m_activePointers;
		for( MapType::Iterator it = active.getIterator();
			active.isValid(it);
			it = active.getNext(it) )
		{
			AllocInfo info = active.getValue(it);
			if( info.m_flags.get(ALLOCATOR_HEAP) )
			{
				HK_ASSERT(0x6b398396, info.m_size>=0);
				heapAllocated += info.m_size;
				if( info.m_size <= MAX_SMALL )
				{
					int bin = info.m_size ? (info.m_size-1)/16 : 0;
					countSmall[ bin ] += 1;
					sumsSmall[bin] += info.m_size;
				}
				else
				{
					int bin = 0;
					unsigned size = info.m_size;
					while( size )
					{
						bin += 1;
						size >>= 1;
					}
					countLarge[bin] += 1;
					totalLarge += info.m_size;
				}
			}
			if( info.m_flags.get(ALLOCATOR_STACK) )
			{
				stackAllocated += info.m_size;
			}
			if( info.m_flags.get(ALLOCATOR_TEMP) )
			{
				tempAllocated += info.m_size;
			}
			if( info.m_flags.get(ALLOCATOR_SOLVER) )
			{
				solverAllocated += info.m_size;
			}
		}
	}

	// allocation totals
	{
		int total = 0;

		hkMemoryAllocator::MemoryStatistics rstats;
		m_rawAllocator->getMemoryStatistics(rstats);
		ostr.printf("\n    Allocation totals:\n\n");
		ostr.printf("%20i allocated by heap\n", heapAllocated);
		ostr.printf("%20i allocated by debug\n", debugAllocated );
		ostr.printf("%20i allocated by stack\n", stackAllocated );
		ostr.printf("%20i allocated by temp\n", tempAllocated);
		ostr.printf("%20i allocated by solver\n", solverAllocated);
		total = heapAllocated + debugAllocated + stackAllocated + tempAllocated + solverAllocated;
		if( m_checkFlags.get(CHECK_DELAYED_FREE) )
		{
			hkMemoryAllocator::MemoryStatistics fstats;
			m_delayedFreeAllocator.getMemoryStatistics(fstats);
			int delayedFreeAllocated = static_cast<int>(fstats.m_allocated-fstats.m_inUse);
			ostr.printf("%20i in delayed frees\n", delayedFreeAllocated );
			total += delayedFreeAllocated;
		}
		if( m_checkFlags.get(CHECK_PAD_BLOCK) )
		{
			hkMemoryAllocator::MemoryStatistics pstats;
			m_paddedAllocator.getMemoryStatistics(pstats);
			int paddedAllocated = static_cast<int>(pstats.m_allocated-pstats.m_inUse);
			ostr.printf("%20i in allocation padding\n", paddedAllocated );
			total += paddedAllocated;
		}
		if( m_checkFlags.get(CHECK_CALLSTACK) )
		{
			hkMemoryAllocator::MemoryStatistics cstats;
			m_callTreeAllocator.getMemoryStatistics(cstats);
			ostr.printf("%20i in callstacks\n", cstats.m_allocated);
			total += static_cast<int>(cstats.m_allocated);
		}

		// additional overheads in checking memory system management
		{
			int pointerMapAllocated = m_activePointers.getMemSize();
			ostr.printf("%20i in active pointer map\n", pointerMapAllocated );
			total += pointerMapAllocated;
			int allocatorForwardersAllocated = m_allocators.getCapacity()*sizeof(AllocatorForwarder*) + m_allocators.getSize()*sizeof(AllocatorForwarder);
			ostr.printf("%20i in allocator forwarders\n", allocatorForwardersAllocated );
			total += allocatorForwardersAllocated;
		}
		
		ostr.printf("%20s\n", "-------");
		ostr.printf("%20i computed total\n", total );
		ostr.printf("%20i reported total\n", static_cast<int>(rstats.m_allocated));
	}
	
	// Small block summary
	{
		ostr.printf("\n    Heap small block (<=%i) summary:\n\n", MAX_SMALL);
		int maxSum = 0;
		for( unsigned i = 0; i < HK_COUNT_OF(sumsSmall); ++i )
		{
			maxSum = hkMath::max2( maxSum, sumsSmall[i] );
		}
		int totSmall = 0;
		for( int i = 0; i < NUM_BINS_SMALL; ++i )
		{
			char hist[41] = {};
			hkString::memSet(hist, '>', (40*sumsSmall[i])/ maxSum );
			int bsize = (i+1)*16;
			ostr.printf("%20i * %6i = %4ik : %s\n", bsize, countSmall[i], (bsize*countSmall[i])/1024, hist );
			totSmall += sumsSmall[i];
		}
		ostr.printf("%20s\n", "-------");
		ostr.printf("%20i bytes total\n", totSmall);

	}

	// large block summary
	{
		ostr.printf("\n    Heap large block (>%i) summary:\n\n", MAX_SMALL);
		for( int i = LOG2_LARGE; i < 32; ++i )
		{
			if( countLarge[i] )
			{
				int blockSize = 1 << i;
				hkStringBuf sb; sb.printf("<=%i", blockSize);
				ostr.printf("%20s * %6i = %4ik\n", sb.cString(), countLarge[i], (blockSize * countLarge[i]) / 1024 );
			}
		}
		ostr.printf("%20s\n", "-------");
		ostr.printf("%20i bytes total\n", totalLarge);
	}

	// thread local temp & solver
	{
		ostr.printf("\n    Threads summary:\n\n");
		for( int i = 0; i < m_allocators.getSize(); ++i )
		{
			const AllocationContext& ctx = m_allocators[i]->m_context;
			if( ctx.flags.get(ALLOCATOR_TEMP|ALLOCATOR_SOLVER) )
			{
				char buf[128];
				ostr.printf("%10s Thread " HK_PRINTF_FORMAT_UINT64 ", %6s, peak %i\n", "", ctx.threadId, flagsToString(buf, ctx.flags), m_allocators[i]->m_context.peakInUse );
			}
		}
	}
}

void hkCheckingMemorySystem::getHeapStatistics(hkMemoryAllocator::MemoryStatistics& stats) const
{
	m_baseAllocator->getMemoryStatistics(stats);
}

void hkCheckingMemorySystem::advanceFrame()
{
	hkCriticalSectionLock lock( &m_section );

	{
		const MapType& active = m_activePointers;
		for( MapType::Iterator it = active.getIterator();
			active.isValid(it);
			it = active.getNext(it) )
		{
			AllocInfo info = active.getValue(it);
			if( info.m_flags.get(ALLOCATOR_TEMP) )
			{
				danger("A temp allocation lived past a frame advance", (void*)active.getKey(it), info );
				HK_ASSERT2(0x75a8f8dd, false, "A temp allocation lived past a frame advance, see console");
			}			
		}
	}
}

void hkCheckingMemorySystem::garbageCollectShared()
{
	hkCriticalSectionLock lock( &m_section );
	{
		//m_paddedAllocator - no caching
		m_delayedFreeAllocator.releaseDelayedFrees();
		//m_rawAllocator - must be handled by caller
	}
}

hkBool32 hkCheckingMemorySystem::isCheckCallstackEnabled() const
{ 
	return m_checkFlags.anyIsSet(CHECK_CALLSTACK);
}

void hkCheckingMemorySystem::setCheckCallstackEnabled( bool enabled )
{
	m_checkFlags.setWithMask(enabled ? CHECK_CALLSTACK : 0, CHECK_CALLSTACK);
}

void hkCheckingMemorySystem::addBookmark( const char* bookmarkName )
{
	m_bookmarks._expandOne( *m_rawAllocator );
	m_bookmarks.back().set( m_rawAllocator, bookmarkName );
}

int hkCheckingMemorySystem::findPrecedingBookmark( hkUint64 timeStamp ) const
{
	// todo binary search?
	for(int i=0; i<m_bookmarks.getSize(); i++)
	{
		if (m_bookmarks[i].m_time > timeStamp )
		{
			return i-1;
		}
	}

	return m_bookmarks.getSize();
}

void hkCheckingMemorySystem::Bookmark::set( class hkMemoryAllocator* alloc, const char* name )
{
	HK_ASSERT(0x45a8161a, !m_name);
	int len = hkString::strLen(name);
	m_name = (char*) alloc->blockAlloc( len+1 );
	if(m_name)
	{
		hkString::memCpy( m_name, name, len+1 );
	}
	m_time = hkStopwatch::getTickCounter();
}

void hkCheckingMemorySystem::Bookmark::clear( class hkMemoryAllocator* alloc )
{
	if(m_name)
	{
		int len = hkString::strLen(m_name);
		alloc->blockFree( m_name, len+1 );
		m_name = HK_NULL;
	}
}


hkUint64 hkCheckingMemorySystem::AllocInfo::getKey() const
{
	// The actual value of the key isn't important, we just want to avoid accidental
	// collisions if the bookmark index is the same as a trace ID.
	bool traceIsValid = m_traceId != -1;
	hkUint32 flags = traceIsValid ? 0 : 1;
	hkUint32 idx = traceIsValid ? (hkUint32) m_traceId : (hkUint32) m_bookmarkIndex;
	return ((hkUint64)(flags) << 32 ) | (hkUint64)(idx);
}


// This will restore the settings to those specified on the command line
#if ( defined(HK_COMPILER_MSVC) && defined(HK_ARCH_IA32) ) && (HK_COMPILER_MSVC_VERSION < 1700)
#pragma optimize("y", on)
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
