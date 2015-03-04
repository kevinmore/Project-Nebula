/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/FreeList/hkFreeListAllocator.h>
#include <Common/Base/Container/MinHeap/hkMinHeap.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>
#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>

// default value for memory scrubbing (might be changed by the determinism util, when performing determinism checks)
hkUint32 hkFreeListAllocator::s_fillReturnedToUser = 0x7ffa110c; // signalling NaN

#ifdef HK_DEBUG

static HK_FORCE_INLINE void MEMORY_SCRUB(void* ptr, int val, int nbytes)
{
	// if you want to use NAN: for (int i =0; i < nbytes/4; i++) ((int*)ptr)[i] = 0x7FC00000;
	if (ptr)
	{
		hkString::memSet4( ptr, val, nbytes/4 );
	}
}

static HK_FORCE_INLINE void MEMORY_SCRUB_BATCH(void** ptrs,int numPtrs, int val, int nbytes)
{
	// if you want to use NAN: for (int i =0; i < nbytes/4; i++) ((int*)ptr)[i] = 0x7FC00000;
	for (int i=0;i<numPtrs;i++)
	{
		void* ptr = ptrs[i];
		if (ptr)
		{
			hkString::memSet4(ptr, val, nbytes/4);
		}
	}
}

#else
#	define MEMORY_SCRUB(PTR, WHAT, NBYTES) /* nothing */
#   define MEMORY_SCRUB_BATCH(PTR,NUM_PTRS, WHAT, NBYTES) /* nothing */
#endif

HK_FORCE_INLINE bool hkFreeListAllocator::_hasMemoryAvailable(hk_size_t size)
{
	if (m_allocatorExtended)
	{
		// We need to work out the total memory used quickly
		hk_size_t sumAllocatedSize = m_allocatorExtended->getApproxTotalAllocated();

		// Work out how much free space there is on freelists
		hk_size_t freeAvailable = m_totalBytesInFreeLists;

		hk_size_t totalUsed = sumAllocatedSize - freeAvailable;

		m_peakInUse = hkMath::max2( totalUsed, m_peakInUse );

		bool hasMemory = (totalUsed < m_softLimit - size);
		return hasMemory;
	}
	else
	{
		// Can't tell -> so just assume there is enough memory
		return true;
	}
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

						  hkFreeListAllocator

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

hkFreeListAllocator::hkFreeListAllocator(hkMemoryAllocator* server, hkMemoryAllocator::ExtendedInterface* allocatorExtended, const Cinfo* info)
:	m_criticalSection(4000),
	m_allocator(server),
	m_allocatorExtended(allocatorExtended),
	m_numFreeLists(0)
{
	_construct(info);
}

hkFreeListAllocator::hkFreeListAllocator()
:	m_criticalSection(4000),
	m_allocator(HK_NULL),
	m_allocatorExtended(HK_NULL),
	m_numFreeLists(0)
{
}

void hkFreeListAllocator::init(hkMemoryAllocator* allocator, hkMemoryAllocator::ExtendedInterface* allocatorExtended, const Cinfo* info)
{
	HK_ASSERT(0x423432aa, allocator && m_allocator == HK_NULL);
	HK_ASSERT(0x423432aa, m_numFreeLists == 0);

	m_allocator = allocator;
	m_allocatorExtended = allocatorExtended;

	_construct(info);
}

void hkFreeListAllocator::_construct(const Cinfo* infoIn)
{
	m_incrementalFreeListIndex = 0;
	m_totalBytesInFreeLists = 0;
	m_peakInUse = 0;

	setMemorySoftLimit(SOFT_LIMIT_MAX);

	// Set up info
	Cinfo defaultInfo;
	if (!infoIn)
	{
		setFixedSizeCinfo(8 * 1024, defaultInfo);
		infoIn = &defaultInfo;
	}
	const Cinfo& info = *infoIn;

	{
		const hk_size_t blockSize  = (info.m_preferedBlockSize == 0) ? 1024 : info.m_preferedBlockSize;
		// Set up the block allocator
		m_blockAllocator.init(sizeof(hkFreeList::Block), HK_ALIGN_OF(hkFreeList::Block), blockSize, m_allocator);
	}

	m_topFreeList = m_freeListMemory;
	m_lastFreeList = m_topFreeList + MAX_UNIQUE_FREELISTS;

	// Zero them all first
	for (int i = 0; i < MAX_FREELISTS;i++)
	{
		m_sizeToFreeList[i] = HK_NULL;
	}

	
	for (int i = 0; i < info.m_numInfos; i++)
	{
		const FreeListCinfo& freeListInfo = info.m_infos[i];
		hkFreeList* freeList = _newFreeList(freeListInfo.m_elementSize, freeListInfo.m_alignment, freeListInfo.m_blockSize);
		HK_ASSERT(0x32423423, freeList != HK_NULL);
		m_sizeToFreeList[freeListInfo.m_elementSize >> FREELIST_SHIFT] = freeList;
	}

	// Fix any of the gaps
	{
		hkFreeList* last = m_sizeToFreeList[HK_COUNT_OF(m_sizeToFreeList) - 1];
		HK_ASSERT(0x322a43a2, last != HK_NULL);

		for (int i = HK_COUNT_OF(m_sizeToFreeList) - 1; i >= 0; i--)
		{
			if (m_sizeToFreeList[i])
			{
				last = m_sizeToFreeList[i];
			}
			else
			{
				m_sizeToFreeList[i] = last;
			}
		}
	}

#if 0
	// Set up known sizes
	m_sizeToFreeList[512 >> FREELIST_SHIFT] = _newFreeList(512, 128, 64*1024);
	//
	m_sizeToFreeList[256 >> FREELIST_SHIFT] = _newFreeList(256, 32, 32*1024);
	// For zero sized allocs
	m_sizeToFreeList[0] = _newFreeList(0,sizeof(void*),256);

	// 512 + 32 aligned 128 boundaries is 640 bytes
	hkFreeList* freeList640 = _newFreeList(512 + 32, 128, 640 * 32);
	HK_ASSERT(0x23a32a23, freeList640->getElementSize() == 640);

	for (int i = 512 + 16; i <= 640; i++)
	{
		m_sizeToFreeList[(i) >> FREELIST_SHIFT] = freeList640;
	}

	// Set all of the simple sizes
	// Go from the largest size downward so we create the one with the largest block first
	for (int i = MAX_FREELIST_SIZE; i >= FREELIST_ALIGNMENT; i -= FREELIST_ALIGNMENT)
	{
		if (m_sizeToFreeList[i >> FREELIST_SHIFT])
		{
			continue;
		}

		hk_size_t blockSize = 1024;
		hk_size_t alignment = 16;

		if (i>0)
		{
			// By default each block has at least 16 elements
			blockSize = i * 16;
		}

		// Min block size is 4k
		if (blockSize < 4096)
		{
			blockSize = 4096;
		}
		if (i >= 64)
		{
			alignment = 32;
		}

		// Create all of the freelists
		m_sizeToFreeList[i >> FREELIST_SHIFT] = _newFreeList(i, alignment, blockSize);
	}
#endif
}

hkFreeListAllocator::~hkFreeListAllocator()
{
	// delete all of the freelists
	for (int i = 0; i < m_numFreeLists; i++)
	{
		hkFreeList* freeList = m_freeLists[i];
		if (freeList)
		{
			_deleteFreeList(freeList);
			// Remove any other references
			for (int j = i + 1; j < m_numFreeLists; j++)
			{
				if (m_freeLists[j] == freeList)
				{
					m_freeLists[j] = HK_NULL;
				}
			}
		}
	}

	// Free all memory used by the block allocator
	m_blockAllocator.m_freeList.freeAllMemory();
}

bool hkFreeListAllocator::canAllocTotal(int size)
{
	// We don't need a critical section here - as the worst that can happen
	// is the amount of memory that is available will be slightly off.
	// In order for this to work the hkFreeList + hkLargeBlock allocators have to be
	// complicit in having methods that are 'thread safe' (in this case the
	// safeness is because the method just return a contained member variable,
	// and reads and writes from that variable are atomic on all known
	// platforms)
	//m_criticalSection.enter();
	bool res = _hasMemoryAvailable( hk_size_t(size) );
	//m_criticalSection.leave();
	return res;
}

hk_size_t hkFreeListAllocator::getApproxTotalAllocated() const
{
	if (m_allocatorExtended)
	{
		// Work out how much free space there is on freelists
		hk_size_t freeAvailable = m_totalBytesInFreeLists;
		hk_size_t totalUsed = m_allocatorExtended->getApproxTotalAllocated() - freeAvailable;

		return totalUsed;
	}
	else
	{
		return 0;
	}
}

void hkFreeListAllocator::setScrubValues(hkUint32 allocValue, hkUint32)
{
	
	m_criticalSection.enter();
	// set the static value for allocation scrubbing.
	s_fillReturnedToUser = allocValue;
	// ignore the free value
	m_criticalSection.leave();
}

int hkFreeListAllocator::Cinfo::findInfoIndex(hk_size_t elementSize) const
{
	for (int i = 0; i < m_numInfos; i++)
	{
		const FreeListCinfo& info = m_infos[i];
		if (elementSize == info.m_elementSize)
		{
			return i;
		}
	}
	return -1;
}

void hkFreeListAllocator::Cinfo::add(hk_size_t elementSize, hk_size_t alignment, hk_size_t blockSize)
{
	// Element sizes must be the alignment sizes
	HK_ASSERT(0x344a2a34, (elementSize & (FREELIST_ALIGNMENT - 1)) == 0);

	// Search for element with that size
	int index = findInfoIndex(elementSize);
	if (index < 0)
	{
		HK_ASSERT(0x234a2652, m_numInfos < int(HK_COUNT_OF(m_infos)));
		index = m_numInfos++;
	}
	FreeListCinfo& info = m_infos[index];

	info.m_elementSize = elementSize;
	info.m_alignment = alignment;
	info.m_blockSize = blockSize;
}

void hkFreeListAllocator::Cinfo::removeAt(int index)
{
	HK_ASSERT(0x3223432, index >= 0 && index < m_numInfos);
	if (index != m_numInfos - 1)
	{
		m_infos[index] = m_infos[m_numInfos - 1];
	}
	m_numInfos--;
}

/* static */void hkFreeListAllocator::setFixedSizeCinfo(hk_size_t blockSize, Cinfo& info)
{
	// Minimum block size is 1k
	HK_ASSERT(0x2342aaa4, blockSize >= 1024);

	info.m_preferedBlockSize = blockSize;

	info.add(512, 128, blockSize);
	info.add(256,  32, blockSize);
	info.add(0, sizeof(void*), blockSize);

	{
		// From (512, ] should use this
		// 640

		hk_size_t size = 512 + 32;			//hkpRigidBody
		hk_size_t align = 128;
		size = (size + align - 1) & ~(align - 1);

		info.add(size, 128, blockSize);

		// If this doesn't compile, means will need some alterations for the freelists above 640
		HK_ASSERT(0x324a2433, MAX_FREELIST_SIZE == size);
	}

	// Set all of the simple sizes
	// Go from the largest size downward so we create the one with the largest block first
	for (int i = 512; i >= FREELIST_ALIGNMENT; i -= FREELIST_ALIGNMENT)
	{
		hk_size_t elementSize = hk_size_t(i);

		if (elementSize >= 256)
		{
			// Remove 2 bits
			elementSize = hkClearBits(elementSize, (FREELIST_ALIGNMENT << 2) - 1);
		}
		else if (elementSize > 128)
		{
			elementSize = hkClearBits(elementSize, (FREELIST_ALIGNMENT << 1) - 1);
		}

		// Work out block size and alignment
		hk_size_t alignment = FREELIST_ALIGNMENT;
		if (elementSize >= 64)
		{
			alignment = 32;
		}

		// Work out the element size with the alignment
		elementSize = (elementSize + alignment - 1) & ~(alignment - 1);

		// See if there is already an entry
		if (info.findInfoIndex(elementSize) >= 0)
		{
			continue;
		}

		// Set
		info.add(elementSize, alignment, blockSize);
	}

}

/* static */void hkFreeListAllocator::setDefaultCinfo(Cinfo& info)
{
	info.add(512, 128, 16 * 1024);
	info.add(256,  32, 16 * 1024);
	info.add(0, sizeof(void*), 256);

	{
		// From (512, ] should use this
		// 640

		hk_size_t size = 512 + 32;			//hkpRigidBody
		hk_size_t align = 128;
		size = (size + align - 1) & ~(align - 1);

		info.add(size, 128, 16 * 1024);
	
		// If this doesn't compile, means will need some alterations for the freelists above 640
		HK_ASSERT(0x324a2433, MAX_FREELIST_SIZE == size);
	}

	// Set all of the simple sizes
	// Go from the largest size downward so we create the one with the largest block first
	for (int i = 512; i >= FREELIST_ALIGNMENT; i -= FREELIST_ALIGNMENT)
	{
		hk_size_t elementSize = hk_size_t(i);

		if (elementSize >= 256)
		{
			// Remove 2 bits
			elementSize = hkClearBits(elementSize, (FREELIST_ALIGNMENT << 2) - 1);
		}
		else if (elementSize > 128)
		{
			elementSize = hkClearBits(elementSize, (FREELIST_ALIGNMENT << 1) - 1);
		}

		// Work out block size and alignment
		hk_size_t blockSize = 1024;
		hk_size_t alignment = FREELIST_ALIGNMENT;

		if (elementSize > 0)
		{
			// By default each block has at least 16 elements
			blockSize = i * 16;
		}

		// Min block size is 4k
		if (blockSize < 4096)
		{
			blockSize = 4096;
		}
		if (elementSize >= 64)
		{
			alignment = 32;
		}

		// Work out the element size with the alignment
		elementSize = (elementSize + alignment - 1) & ~(alignment - 1);

		// See if there is already an entry
		if (info.findInfoIndex(elementSize) >= 0)
		{
			continue;
		}

		// Set
		info.add(elementSize, alignment, blockSize);
	}
}

void hkFreeListAllocator::getMemoryStatistics(MemoryStatistics& stats) const
{
	m_criticalSection.enter();

	// Get the large memory statistics
	m_allocator->getMemoryStatistics(stats);

	hkMemoryAllocator::MemoryStatistics freeTotals;
	freeTotals.m_allocated = 0;
	freeTotals.m_inUse = 0;
	freeTotals.m_available = 0;

	for (int i = 0; i < m_numFreeLists + 1; i++)
	{
		hkMemoryAllocator::MemoryStatistics freeStats;
		const hkFreeList* list = HK_NULL;
		if (i == m_numFreeLists)
		{
			// The block allocator freelist
			list = &m_blockAllocator.m_freeList;
		}
		else
		{
			list = m_freeLists[i];
		}

		list->getMemoryStatistics(freeStats);

		if (list->isFreeElementAvailable() && list->getElementSize() > unsigned(stats.m_largestBlock) )
		{
			stats.m_largestBlock = list->getElementSize();
		}

		freeTotals.m_available += freeStats.m_available;
		freeTotals.m_inUse += freeStats.m_inUse;
		freeTotals.m_allocated += freeStats.m_allocated;
	}

	stats.m_available += freeTotals.m_available;
	stats.m_inUse = stats.m_inUse - freeTotals.m_allocated + freeTotals.m_inUse;
	stats.m_peakInUse = m_peakInUse;

	m_criticalSection.leave();
}


void hkFreeListAllocator::resetPeakMemoryStatistics()
{
	hkMemoryAllocator::MemoryStatistics s;
	getMemoryStatistics(s);
	m_peakInUse = s.m_inUse;
}

hkBool hkFreeListAllocator::isOk() const
{
	m_criticalSection.enter();

	hkBool ok = true;
	for (int i = 0; i < m_numFreeLists; i++)
	{
		if (!m_freeLists[i]->isOk())
		{
			ok = false;
			break;
		}
	}
	m_criticalSection.leave();
	return ok;
}

hkFreeList* hkFreeListAllocator::_newFreeList(hk_size_t elementSize, hk_size_t alignment, hk_size_t blockSize)
{
	if (m_topFreeList >= m_lastFreeList)
	{
		HK_ASSERT2(0x32432423, 0, "Too many freelists have been allocated -> MAX_UNIQUE_FREELISTS isn't big enough to cope");
		HK_BREAKPOINT(0);
	}

	hkFreeList* list = m_topFreeList++;
	new (list) hkFreeList( elementSize, alignment, blockSize, m_allocator, &m_blockAllocator);

	// See if we already have one of the same element size
	for (int i = 0; i < m_numFreeLists; i++)
	{
		if (m_freeLists[i]->getElementSize() == list->getElementSize())
		{
			_deleteFreeList(list);
			return m_freeLists[i];
		}
	}

	m_freeLists[m_numFreeLists++] = list;
	return list;
}

void hkFreeListAllocator::_deleteFreeList(hkFreeList* freeList)
{
	// Call the destructor
	freeList->~hkFreeList();
	if (freeList + 1 == m_topFreeList)
	{
		m_topFreeList--;
	}
}

int hkFreeListAllocator::getAllocatedSize( const void *p, int nbytes ) const
{
	if (nbytes <= MAX_FREELIST_SIZE)
	{
		if ( nbytes == 0 )
		{
			return 0;
		}
		hkFreeList* list = m_sizeToFreeList[(nbytes+FREELIST_ALIGNMENT-1)>>FREELIST_SHIFT];
		if (list)
		{
			return int(list->getElementSize());
		}
	}
	// Else its coming from the large memory allocator
	return m_allocator->getAllocatedSize(p, nbytes);
}

void* hkFreeListAllocator::bufAlloc(int& reqNumInOut)
{
	m_criticalSection.enter();

	void* ret;
	if (reqNumInOut <= MAX_FREELIST_SIZE)
	{
		hkFreeList* list = m_sizeToFreeList[(reqNumInOut+FREELIST_ALIGNMENT-1)>>FREELIST_SHIFT];
		hk_size_t oldNumFree = list->m_numFreeElements;
		ret = list->alloc();
		hk_size_t newNumFree = list->m_numFreeElements;
		m_totalBytesInFreeLists += (newNumFree-oldNumFree) * list->getElementSize();
		reqNumInOut = (int)list->getElementSize();
	}
	else
	{
		ret = m_allocator->blockAlloc(reqNumInOut);
	}

#if defined (HK_ENABLE_MEMORY_EXCEPTION_UTIL)
	if (!hkMemoryExceptionTestingUtil::isMemoryAvailable(0)) 
	{
		hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OUT_OF_MEMORY );
	}
#endif
	if ( !_hasMemoryAvailable(0) )
	{
		hkSetOutOfMemoryState(hkMemoryAllocator::MEMORY_STATE_OUT_OF_MEMORY);
	}

	MEMORY_SCRUB(ret, s_fillReturnedToUser, reqNumInOut);

	m_criticalSection.leave();
	return ret;
}

void* hkFreeListAllocator::blockAlloc(int nbytes)
{
	return bufAlloc(nbytes);
}


void hkFreeListAllocator::blockAllocBatch( void** blocksOut, int nblocks, int nbytes )
{
	m_criticalSection.enter();

	if (nbytes <= MAX_FREELIST_SIZE)
	{
		hkFreeList* list = m_sizeToFreeList[(nbytes + FREELIST_ALIGNMENT - 1) >> FREELIST_SHIFT];
		hk_size_t oldNumFree = list->m_numFreeElements;
		list->allocBatch(blocksOut,nblocks);
		hk_size_t newNumFree = list->m_numFreeElements;
		m_totalBytesInFreeLists += (newNumFree-oldNumFree) * list->getElementSize();

		MEMORY_SCRUB_BATCH(blocksOut,nblocks, s_fillReturnedToUser, nbytes);
	}
	else
	{
		for (int i = 0; i < nblocks; i++)
		{
			void* ret = m_allocator->blockAlloc(nbytes);
			MEMORY_SCRUB(ret, s_fillReturnedToUser, nbytes);
			blocksOut[i] = ret;
		}
	}
#if defined (HK_ENABLE_MEMORY_EXCEPTION_UTIL)
	if (!hkMemoryExceptionTestingUtil::isMemoryAvailable(0)) 
	{
		hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OUT_OF_MEMORY );
	}
#endif

	if ( !_hasMemoryAvailable(0) )
	{
		hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OUT_OF_MEMORY );
	}

	m_criticalSection.leave();
}

void  hkFreeListAllocator::blockFree(void* p, int nbytes)
{
	if(p)
	{
		m_criticalSection.enter();
		
		if (nbytes <= MAX_FREELIST_SIZE)
		{
			hkFreeList* list = m_sizeToFreeList[(nbytes+FREELIST_ALIGNMENT-1)>>FREELIST_SHIFT];
			hk_size_t oldNumFree = list->m_numFreeElements;
			list->free(p);
			hk_size_t newNumFree = list->m_numFreeElements;
			m_totalBytesInFreeLists += (newNumFree-oldNumFree) * list->getElementSize();
		}
		else
		{
			m_allocator->blockFree(p, nbytes);
		}
		m_criticalSection.leave();
	}
}

void hkFreeListAllocator::blockFreeBatch(void** blocks, int nblocks, int nbytes )
{
	m_criticalSection.enter();
	if (nbytes <= MAX_FREELIST_SIZE)
	{
		hkFreeList* list = m_sizeToFreeList[(nbytes+FREELIST_ALIGNMENT-1)>>FREELIST_SHIFT];
		hk_size_t oldNumFree = list->m_numFreeElements;
		list->freeBatch(blocks,nblocks);
		hk_size_t newNumFree = list->m_numFreeElements;
		m_totalBytesInFreeLists += (newNumFree-oldNumFree) * list->getElementSize();
	}
	else
	{
		for( int i = 0; i < nblocks; ++i )
		{
			void* mem = blocks[i];
			if (mem)
			{
				m_allocator->blockFree(mem, nbytes);
			}
		}
	}
	m_criticalSection.leave();
}

struct WalkInfo
{
	WalkInfo(hkMemoryAllocator* allocator, hkFreeList* freeLists[], int numFreeLists, hkMemoryAllocator::MemoryWalkCallback callback, void* param):
		m_allocator(allocator),
		m_allocs(HK_NULL),
		m_numAllocs(0),
		m_callback(callback),
		m_param(param)
	{
		// Work out the number of freelist blocks - so we can allocate a memory to hold the pointers to allocations
		for (int i = 0; i < numFreeLists; i++)
		{
			hkFreeList* freeList = freeLists[i];
			m_numAllocs += hkFreeList::calcNumBlocks(freeList->getActiveBlocks());
			m_numAllocs += hkFreeList::calcNumBlocks(freeList->getFreeBlocks());
		}

		// Allocate using the allocator
		m_allocs = (void**)allocator->blockAlloc(int(m_numAllocs * sizeof(void*)));
		HK_ASSERT(0x3243b2a2, m_allocs);
		if (m_allocs)
		{
			// Find and add all of the allocations, as seen by the underlying allocator
			void** cur = m_allocs;
			for (int i = 0; i < numFreeLists; i++)
			{
				cur = _addBlockAllocs(cur, freeLists[i]->getActiveBlocks());
				cur = _addBlockAllocs(cur, freeLists[i]->getFreeBlocks());
			}

			// All slots should be full
			HK_ASSERT(0x3242a323, cur == m_allocs + m_numAllocs);

			// Order so can be searched more quickly (with a binary chop)
			hkSort(m_allocs, m_numAllocs);
		}
	}
	~WalkInfo()
	{
		if (m_allocs)
		{
			m_allocator->blockFree(m_allocs, int(sizeof(void*) * m_numAllocs));
		}
	}

	static void HK_CALL _Callback(void* block, hk_size_t size, bool allocated, int pool, void* param)
	{
		((WalkInfo*)param)->handleBlock(block, size, allocated);
	}

	int _findAllocIndex(void* ptr) const
	{
		// Search via binary chop
		void** allocs = m_allocs;
		int size = m_numAllocs;

		while ( size > 0)
		{
			int middle = size / 2; 
			void* cur = allocs[middle];
			if (ptr < cur)
			{
				// Must be in bottom half
				size = middle; 
			}
			else if (ptr > cur)
			{
				// Must be in top half
				size = size - (middle + 1);
				allocs += middle + 1;
			}
			else
			{
				// This must be it
				return (int)(allocs - m_allocs) + middle;
			}
		}
		
		// Didn't find it
		return -1;
	}
	int _findAllocIndexSlow(void* ptr) const
	{
		// Linearly search for the entry
		for (int i = 0; i < m_numAllocs; i++)
		{
			if (m_allocs[i] == ptr)
			{
				return i;
			}
		}
		return -1;
	}
	void handleBlock(void* block, hk_size_t size, hkBool allocated)
	{
		if (allocated)
		{
			// Only if it's allocated can it hold freelist data. 
			// Search to see if the block is one a freelist has allocated. If found, ignore.
			const int index = _findAllocIndex(block);
			HK_ASSERT(0x23432432, index == _findAllocIndexSlow(block));
			// If it's in list then we are done
			if ( index >= 0)
			{
				return;
			}
		}
		
		// Do the callback
		m_callback(block, size, allocated, 0, m_param);
	}

	static void** _addBlockAllocs(void** cur, const hkFreeList::Block* block)
	{
		for (; block; block = block->m_next)
		{
			*cur++ = (block->m_elementsAlloc) ?  block->m_elementsAlloc : (void*)block;
		}
		return cur;
	}

	hkMemoryAllocator* m_allocator;			///< Allocator used to allocate m_allocs (so can be freed in dtor)
	void** m_allocs;						///< Ordered for fast lookup. Allocations by held by the freelist (allocated on m_allocator).
	int m_numAllocs;						///< Total number of freelist allocations
	hkMemoryAllocator::MemoryWalkCallback m_callback;	///< The callback
	void* m_param;
};


hkResult hkFreeListAllocator::walkMemory(MemoryWalkCallback callback,void* param)
{
	// The difficulty in doing a memory walk is that the freelists will make large allocations 
	// to the underlying allocator, but we don't want to return the freelists allocations (otherwise
	// we'll have the same sections of memory appearing more than once).
	// So to stop this we work out all of the allocations from freelists going to the underlying allocator
	// and put in an ordered list. Then when the underlying allocator has it's walkMemory called, we'll 
	// only return blocks that are not in that list.

	if (!m_allocatorExtended)
	{
		return HK_FAILURE;
	}

	hkCriticalSectionLock lock(&m_criticalSection);

	// Create a list of all freelists, including the block allocator freelist.
	hkFreeList* freeLists[MAX_FREELISTS + 1];
	hkString::memCpy(freeLists, m_freeLists, m_numFreeLists * sizeof(hkFreeList*));
	freeLists[m_numFreeLists] = &m_blockAllocator.m_freeList;

	{
		// Set up walk info -> works out allocations made by the freelists, so they can be avoided
		WalkInfo info(hkMallocAllocator::m_defaultMallocAllocator, freeLists, m_numFreeLists + 1, callback, param);

		// Walk the underlying allocators memory, through the WalkInfo, which will filter out FreeList Block allocations
		hkResult res = m_allocatorExtended->walkMemory(WalkInfo::_Callback, &info);
		if (res != HK_SUCCESS)
		{
			return res;
		}
	}

	// Walk the contents of the freelist
	for (int i = 0; i < m_numFreeLists; i++)
	{
		hkFreeList* list = m_freeLists[i];
		list->walkMemory(callback, i + 1, param);
	}

	// Do the walk on the block allocations
	m_blockAllocator.m_freeList.walkMemory(callback, m_numFreeLists + 1, param);

	return HK_SUCCESS;
}

void hkFreeListAllocator::garbageCollect()
{
	m_criticalSection.enter();
	// We can garbage collect freelists
	// We are going to keep running around this list until we've wrung the last chunk of free memory out of them
	hkBool memoryFreed = false;
	hkBool collectFailed = false;
	int totalFreed = 0;
	do
	{
		memoryFreed = false;
		for (int i = 0; i < m_numFreeLists; i++)
		{
			hkFreeList* list = m_freeLists[i];
			// If it has free blocks then there will be blocks freed
			// Look for garbage
			hk_size_t oldNumFree = list->m_numFreeElements;
			if (list->findGarbage()<0)
			{
				collectFailed = true;
			}
			// If we have some free blocks then some memory was found
			if (list->hasFreeBlocks())
			{
				memoryFreed = true;
			}
			// Free all of the blocks
			totalFreed += list->freeAllFreeBlocks();
			hk_size_t newNumFree = list->m_numFreeElements;
			m_totalBytesInFreeLists += (newNumFree-oldNumFree) * list->getElementSize();
		}
	}    while (memoryFreed&&collectFailed);

	if (m_allocatorExtended)
	{
		// Collect the large block allocator
		m_allocatorExtended->garbageCollect();
	}
	
	m_criticalSection.leave();
}


void hkFreeListAllocator::incrementalGarbageCollect(int numBlocksIn)
{
	m_criticalSection.enter();

	// Presumably some collection is wanted... so do a min of one block
	int numBlocks = hkMath::max2(1, numBlocksIn);

	const int firstFreeListIndex = m_incrementalFreeListIndex;
	int freeListIndex = firstFreeListIndex;

	do 
	{
		hkFreeList* list = m_freeLists[freeListIndex];

		int numCollected;
		// Do an incremental collection on the freelist
		hk_size_t oldNumFree = list->m_numFreeElements;
		if (list->incrementalFindGarbage(numBlocks, numCollected))
		{
			// Step onto next one as we've hit the end
			freeListIndex++;
			if (freeListIndex >= m_numFreeLists)
			{
				freeListIndex = 0;
			}
		}

		// Free any free blocks
		list->freeAllFreeBlocks();
		// Remove any processed
		numBlocks -= numCollected;
		hk_size_t newNumFree = list->m_numFreeElements;
		m_totalBytesInFreeLists += (newNumFree-oldNumFree) * list->getElementSize();
	} while(numBlocks > 0 && freeListIndex != firstFreeListIndex);

	// Save off, for next incremental call
	m_incrementalFreeListIndex = freeListIndex;

	if (m_allocatorExtended)
	{
		// If there is an extended inteface, some incremental collection on it may be needed
		m_allocatorExtended->incrementalGarbageCollect(numBlocksIn);
	}

	m_criticalSection.leave();
}


hkResult hkFreeListAllocator::setMemorySoftLimit(hk_size_t maxMemory)
{
	hkCriticalSectionLock lock(&m_criticalSection);
	m_softLimit = maxMemory;

	return HK_SUCCESS;
}

hk_size_t hkFreeListAllocator::getMemorySoftLimit() const
{
	return m_softLimit;
}

int hkFreeListAllocator::addToSnapshot( hkMemorySnapshot& snap, int parentId )
{
	hkCriticalSectionLock lock( &m_criticalSection );

	if( m_allocatorExtended )
	{
		int heapId = m_allocatorExtended->addToSnapshot(snap, parentId);
		if( heapId == -1 )
		{
			return -1;
		}

		int freeListId = snap.addProvider("hkFreeListAllocator", heapId);
		for (int i = 0; i < m_numFreeLists; i++)
		{
			hkFreeList* list = m_freeLists[i];
			list->addToSnapshot(snap, hkMemorySnapshot::STATUS_USED, freeListId);
		}

		// TODO: Do the walk on the block allocations
// 		hkString::snprintf(name, HK_COUNT_OF(name), "hkBlockAllocator", m_blockAllocator.getElementSize());
// 		cbparam.providerId  = snap.addProvider(name, systemId);
// 		m_blockAllocator.m_freeList.walkMemory(_addToSnapshot, HK_NULL, &cbparam );

		m_blockAllocator.m_freeList.addToSnapshot(snap, hkMemorySnapshot::STATUS_OVERHEAD, freeListId);
		return freeListId;
	}
	return -1;
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
