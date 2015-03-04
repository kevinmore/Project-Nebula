/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

// this
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerSnapshot.h>
#include <Common/Base/DebugUtil/MultiThreadCheck/hkMultiThreadCheck.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>

#if defined(HK_PLATFORM_PS3_PPU)
#include <Common/Base/Spu/Util/hkSpuUtil.h>
#endif

static hkMallocAllocator s_malloc;

HK_FORCE_INLINE static hkBool32 _compareClassAllocs(const hkDefaultMemoryTracker::ClassAlloc& a, const hkDefaultMemoryTracker::ClassAlloc& b)
{
	return a.m_ptr < b.m_ptr;
}

template<typename T>
static void arrayBaseSwap(hkArrayBase<T>& a, hkArrayBase<T>& b)
{
	T* ap = a.begin();
	int as = a.getSize();
	int af = a.getCapacityAndFlags();
	a._setDataUnchecked( b.begin(), b.getSize(), b.getCapacityAndFlags() );
	b._setDataUnchecked( ap, as, af );
}

hkTrackerSnapshot::hkTrackerSnapshot()
	: m_mem(&s_malloc)
{
}

hkTrackerSnapshot::hkTrackerSnapshot(hkMemoryAllocator* mem)
	: m_mem(mem)
{
}

hkTrackerSnapshot::hkTrackerSnapshot(const hkTrackerSnapshot& rhs)
	: m_mem(rhs.m_mem)
	, m_rawSnapshot(rhs.m_rawSnapshot)
{
	m_classAllocations._append( *m_mem, rhs.m_classAllocations.begin(), rhs.m_classAllocations.getSize() );
}

hkTrackerSnapshot::~hkTrackerSnapshot()
{
	m_classAllocations._clearAndDeallocate(*m_mem);
	m_memSysStatistics._clearAndDeallocate(*m_mem);
}

void hkTrackerSnapshot::swap(hkTrackerSnapshot& rhs)
{
	if ( this == &rhs)
	{
		return;
	}

	hkAlgorithm::swap(m_mem, rhs.m_mem);
	arrayBaseSwap(m_classAllocations, rhs.m_classAllocations);
}

void hkTrackerSnapshot::orderClassAllocs()
{
	hkSort(m_classAllocations.begin(), m_classAllocations.getSize(), _compareClassAllocs);
}

#if defined(HK_PLATFORM_PS3_PPU)

static void _buildSPUFreeBlocksMap(hkMapBase<hkUlong, hkUlong>& freeSPUBlocks, hkMemoryAllocator& allocator)
{
	const int numRows = hkSpuUtil::getNumFreeListRows();

	for (int i = 0; i < numRows; i++)
	{
		void* allocs[60];
		const int numAllocs = hkSpuUtil::getFreeListBlocks(i, allocs);

		for (int j = 0; j < numAllocs; j++)
		{
			freeSPUBlocks.insert(allocator, reinterpret_cast<hkUlong>(allocs[j]), 0);
		}
	}
}

#endif

hkResult hkTrackerSnapshot::init(hkMemorySystem* memorySystem, hkDefaultMemoryTracker* tracker)
{
	m_classAllocations._clearAndDeallocate(*m_mem);
	
	// Try and work
	if( tracker == HK_NULL )
	{
		hkMemoryTracker* currentTracker = hkMemoryTracker::getInstancePtr();
		if (!currentTracker || !hkDefaultMemoryTrackerClass.equals(currentTracker->getClassType()) )
		{
			return HK_FAILURE;
		}
		tracker = static_cast<hkDefaultMemoryTracker*>(currentTracker);
	}

	if (!memorySystem)
	{
		memorySystem = &hkMemorySystem::getInstance();
	}

	HK_ASSERT(0x324a4324, memorySystem || tracker);

	m_rawSnapshot.setAllocator(m_mem);
	memorySystem->getMemorySnapshot(m_rawSnapshot);
	m_rawSnapshot.sort();
	{
		// Destroy the hkOstream and the writer as soon as possible
		// The writer is allocated from the stack to avoid polluting the heap statistics while printing them.
		hkArrayStreamWriter writer(&m_memSysStatistics, *m_mem, hkArrayStreamWriter::ARRAY_BORROW);
		hkOstream memoryOstream(&writer);
		memorySystem->printStatistics(memoryOstream);
	}

	// Add all of the tracker blocks

#if defined(HK_PLATFORM_PS3_PPU)
	// On PlayStation(R)3 we have a helper thread (hkSpuHelperThreadFunc) which handles requests
	// of memory by SPUs. The thread runs continuously on PPU and is unlocked every time
	// new memory is requested. Once the memory is allocated, it is placed in global memory bins.
	// SPUs can read and write to these bins using DMA, this is the way they use to reserve an allocated
	// block for themselves. Once they are done with it, the block is placed back into the bins to
	// be used by other SPUs or eventually be freed by the helper thread once a bin is full.
	// Taking a memory snapshot happens on PPU, at this point in time we can't be sure that
	// no SPU job is running. 
	// The bins are accessible though, so we could access them read only and build a hash map of blocks 
	// allocated that are waiting in the bins.
	// Allocations performed by the helper thread are reported as used memory allocations by the hkMemorySystem
	// normally, but they are also reported to the tracker as special tracker blocks with type SPUHelperThreadBlock.
	// If any of the blocks with that name matches one of the entries in the hash map, it means that the block is
	// waiting to be reallocated inside the bin. Since it is actually free, we will not propagate this tracker block
	// any further as it doesn't correspond to any live object.
	// In practice, even if we have a match in the hash map, it might be that the block was released 
	// later and that it was actually in use when the snapshot was taken from the hkMemorySystem. If we don't have
	// a match in the hash map the block might as well be actually in the bin (just it wasn't when we were reading
	// them). 
	// There is also another really bad case for this mechanism, we can allocate a block normally on PPU and
	// then hand it over to one of the SPUs. When the SPU releases the block it will put it back in the bin,
	// but the memory tracker will not be updated with a new type of the block. The content of the block will
	// always be interpreted as the previous type. In the worst case the snapshot procedure might crash
	// while accessing an invalid pointer because it misinterprets the type of the block.
	// Please note that this is just a best-effort attempt to strip unused blocks.

	// Use an hkUlong to hkUlong map to avoid a new instantiation of the template.
	hkMapBase<hkUlong, hkUlong> freeSPUBlocks;
	_buildSPUFreeBlocksMap(freeSPUBlocks, *m_mem);
#endif

	{
		typedef hkDefaultMemoryTracker::ClassAllocMapType ClassAllocMapType;
		const ClassAllocMapType& classAllocMap = tracker->getClassAllocations();

		m_classAllocations._setSize(*m_mem, classAllocMap.getSize());
		m_classAllocations.clear();
		
		ClassAllocMapType::Iterator iter = classAllocMap.getIterator();

		for (; classAllocMap.isValid(iter); iter = classAllocMap.getNext(iter))
		{
			const ClassAlloc* clsAlloc = classAllocMap.getValue(iter);

			#if defined(HK_PLATFORM_PS3_PPU)
				// The exclamation mark is needed because is reported as "RAW"
				if( hkString::strCmp(clsAlloc->m_typeName, "!SPUHelperThreadBlock") == 0 )
				{
					if( freeSPUBlocks.hasKey(reinterpret_cast<hkUlong>(clsAlloc->m_ptr)) )
					{
						// It's contained in the map, therefore the block is actually free, do not
						// add it to the snapshot.
						continue;
					}
					if( findAllocationForClassAllocation( *clsAlloc ) == HK_NULL )
					{
						// It doesn't correspond to any allocation, probably because the allocation was
						// deleted before we actually got the getMemorySnapshot() call.
						continue;
					}
				}
			#endif

			ClassAlloc& dstClsAlloc = m_classAllocations._expandOne(*m_mem);

			dstClsAlloc = *clsAlloc;

			HK_ASSERT(0x24343242, dstClsAlloc.m_typeName);
		}
	}

#if defined(HK_PLATFORM_PS3_PPU)
	// clear the map
	freeSPUBlocks.clearAndDeallocate(*m_mem);
#endif

#if defined(HK_MEMORY_TRACKER_ENABLE)
	// Section only executed if the memory tracker is enabled.

	// The memory tracker is informed every time a new object is allocated
	// and destroyed if the proper havok allocator is used. Starting from those
	// objects it will follow references to arrays and other containers and discover
	// new blocks it didn't know about. There are some objects which are allocated
	// in a static buffer but nevertheless contain references to heap memory not
	// reachable elsewhere. To track that memory we need to add a new block for the
	// static object, but we mark it as NOT_ALLOCATED to remember that this block
	// is not actually allocated on the heap.

	if (hkMultiThreadCheck::s_stackTree)
	{
		// Add it as if it is a block
		ClassAlloc& clsAlloc = m_classAllocations._expandOne(*m_mem);
		clsAlloc.m_typeName = hkStackTracer::CallTree::getTypeIdentifier();
		clsAlloc.m_size = sizeof(hkStackTracer::CallTree);
		clsAlloc.m_ptr = hkMultiThreadCheck::s_stackTree;
		clsAlloc.m_flags = ClassAlloc::FLAG_NOT_ALLOCATED;
	}

#endif

	hkSort(m_classAllocations.begin(), m_classAllocations.getSize(), _compareClassAllocs);

	return HK_SUCCESS;
}

hkResult hkTrackerSnapshot::checkConsistent() const
{
	const hkArrayBase<hkMemorySnapshot::Allocation>& allocs = m_rawSnapshot.getAllocations();
	
	// 1: check that the allocations are properly sorted
	for (int i = 0; i < allocs.getSize() - 1; ++i)
	{
		const hkMemorySnapshot::Allocation& alloc = allocs[i];
		const hkMemorySnapshot::Allocation& next = allocs[i+1];

		if ( static_cast<const char*>(alloc.m_start) > 
			 static_cast<const char*>(next.m_start) )
		{
			return HK_FAILURE;
		}
	}

	{
		// Every class alloc must be inside an allocation
		for (int i = 0; i < m_classAllocations.getSize(); ++i)
		{
			const ClassAlloc& clsAlloc = m_classAllocations[i];
			if (clsAlloc.m_flags & ClassAlloc::FLAG_NOT_ALLOCATED)
			{
				// Won't find a matching alloc - cos this is not allocated.
				continue;
			}

			int j;
			for (j = 0; j < allocs.getSize(); ++j)
			{
				const hkMemorySnapshot::Allocation& alloc = allocs[j];
				char* ptr = (char*)clsAlloc.m_ptr;
				char* allocStart = (char*)alloc.m_start;
				char* allocEnd = allocStart + alloc.m_size;

				if (ptr < allocStart)
				{
					// Couldn't find an allocation it is in
					return HK_FAILURE;
				}

				if (ptr >= allocStart && ptr < allocEnd)
				{
					// It points inside. Make sure its all inside
					char* end = ptr + clsAlloc.m_size;
					if (end > allocEnd)
					{
						return HK_FAILURE;
					}
					// Its inside, try next
					break;
				}
			}

			if (j >= allocs.getSize())
			{
				// Didn't find the allocation it is in
				return HK_FAILURE;
			}
		}
	}

	return HK_SUCCESS;
}

const hkTrackerSnapshot::ClassAlloc* hkTrackerSnapshot::findClassAllocation(const void* ptrIn) const
{
	const hkUint8* ptr = static_cast<const hkUint8*>(ptrIn);

	// Search via binary chop
	{
		const ClassAlloc* allocs = m_classAllocations.begin();
		int size = m_classAllocations.getSize();

		while ( size > 0)
		{
			int middle = size / 2; 
			const ClassAlloc& alloc = allocs[middle];

			if (ptr < ((const hkUint8*)alloc.m_ptr))
			{
				// Must be in bottom half
				size = middle; 
			}
			else if (ptr >= ((const hkUint8*)alloc.m_ptr) + alloc.m_size)
			{
				// Must be in top half
				size = size - (middle + 1);
				allocs += middle + 1;
			}
			else
			{
				// This must be it
				return &alloc;
			}
		}
	}

	return HK_NULL;
}

/// Find the allocation that contains the given class allocation
const hkMemorySnapshot::Allocation* hkTrackerSnapshot::findAllocationForClassAllocation( const ClassAlloc& classAlloc )
{
	const hkArrayBase<hkMemorySnapshot::Allocation>& allocsArray = m_rawSnapshot.getAllocations();

	// search via binary chop
	{
		const hkMemorySnapshot::Allocation* allocs = allocsArray.begin();
		int size = allocsArray.getSize();
		while( size > 0 )
		{
			int middle = size / 2; 
			const hkMemorySnapshot::Allocation& alloc = allocs[middle];

			if( hkAddByteOffset(classAlloc.m_ptr, classAlloc.m_size) < alloc.m_start )
			{
				// Must be in bottom half
				size = middle; 
			}
			else if ( classAlloc.m_ptr >= hkAddByteOffsetConst(alloc.m_start, alloc.m_size) )
			{
				// Must be in top half
				size = size - (middle + 1);
				allocs += middle + 1;				
			}
			else if ( classAlloc.m_ptr >= alloc.m_start &&
				      hkAddByteOffset(classAlloc.m_ptr, classAlloc.m_size) <= hkAddByteOffsetConst(alloc.m_start, alloc.m_size) )
			{
				// This must be it
				return &alloc;
			}
			else
			{
				// The structure of the allocations is not consistent with thi block, return.
				return HK_NULL;
			}
		}
	}

	return HK_NULL;
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
