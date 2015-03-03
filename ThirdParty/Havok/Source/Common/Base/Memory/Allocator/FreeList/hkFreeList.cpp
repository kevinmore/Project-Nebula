/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/FreeList/hkFreeList.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

// This is hkSort::sortList except the comparison operation changed.
hkFreeList::Block* _sortByAllocationAddress(hkFreeList::Block* headPtr)
{
	typedef hkFreeList::Block ListElement;

	// Mergesort For Linked Lists. Algorithm described at
	// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html

	if( headPtr == HK_NULL )
	{
		return HK_NULL;
	}

	for( int sortSize = 1; true; sortSize *= 2 )
	{
		int numMerges = 0;

		ListElement* p = headPtr;
		ListElement preHead; preHead.m_next = HK_NULL;
		ListElement* tail = &preHead;

		while( p )
		{
			numMerges += 1;

			ListElement* q = p;
			int psize;
			for( psize = 0; psize < sortSize && q != HK_NULL; ++psize )
			{
				q = q->m_next;
			}
			int qsize = sortSize;

			// while current lists not empty
			while( psize>0 && qsize>0 && q != HK_NULL )
			{
				ListElement* next;
				if( p->m_elementsAlloc <= q->m_elementsAlloc ) // THIS IS DIFFERENT FROM THE ORIGINAL
				{
					next = p;
					p = p->m_next;
					psize -= 1;
				}
				else
				{
					next = q;
					q = q->m_next;
					qsize -= 1;
				}
				tail->m_next = next;
				tail = next;
			}
			// one or both lists empty
			while( psize > 0 )
			{
				tail->m_next = p;
				tail = p;
				p = p->m_next;
				psize -= 1;
			}
			while( qsize>0 && q != HK_NULL )
			{
				tail->m_next = q;
				tail = q;
				q = q->m_next;
				qsize -= 1;
			}
			p = q;
		}
		tail->m_next = HK_NULL;

		if( numMerges <= 1 )
		{
			return preHead.m_next;
		}
		else
		{
			headPtr = preHead.m_next;
		}
	}
}

hkFreeList::hkFreeList( hk_size_t elementSize, hk_size_t align, hk_size_t blockSize, hkMemoryAllocator* elementAllocator, hkMemoryAllocator* blockAllocator)
{
	_init(elementSize, align, blockSize, elementAllocator, blockAllocator);
}

hkFreeList::hkFreeList()
{
	m_free = HK_NULL;
	m_activeBlocks = HK_NULL;
	m_freeBlocks = HK_NULL;
	m_blockSize = 0;
	m_align = 0;
	m_top = HK_NULL;
	m_blockEnd = HK_NULL;
	m_numFreeElements = 0;
	m_totalNumElements = 0;

	m_blockAllocator = HK_NULL;
	m_elementAllocator = HK_NULL;

	m_elementSize = 0;
	//m_maxBlockSize = 0;
}

void hkFreeList::_init(hk_size_t elementSize, hk_size_t align, hk_size_t blockSize, hkMemoryAllocator* elementAllocator, hkMemoryAllocator* blockAllocator )
{
	HK_ASSERT(0x32423432, blockSize > 0);

	if (!elementAllocator)
	{
		HK_ASSERT(0x32423423, hkMemoryRouter::getInstancePtr());
		elementAllocator = &hkMemoryRouter::getInstance().heap();
	}

	// There must be an element allocator, but not necessarily a block allocator
	HK_ASSERT(0x324a3432, elementAllocator);

	m_free = HK_NULL;
	m_activeBlocks = HK_NULL;
	m_freeBlocks = HK_NULL;
	m_blockSize = blockSize;
	m_align = align;
	m_top = HK_NULL;
	m_blockEnd = HK_NULL;
	m_numFreeElements = 0;
	m_totalNumElements = 0;

	m_elementAllocator = elementAllocator;
	m_blockAllocator = blockAllocator;

	HK_ASSERT(0x324ab235, align >= sizeof(Element));
	HK_ASSERT(0x324ac243, elementAllocator );

	m_lastIncrementalBlock = HK_NULL;

	// This needs to be true for alignment assumptions to work correctly
	HK_COMPILE_TIME_ASSERT( (sizeof(Block) & 0xf) == 0);

	if (elementSize < align)
	{
		elementSize = align;
	}
	else
	{
		elementSize = (elementSize + align - 1) & ~(align - 1);
	}
	m_elementSize = elementSize;
	m_numBlockElements = blockSize / m_elementSize;

	//m_maxBlockSize = bestBlockSize(4096 - sizeof(Block),m_align);
}

void hkFreeList::init(hk_size_t elementSize, hk_size_t align, hk_size_t blockSize, hkMemoryAllocator* elementAllocator, hkMemoryAllocator* blockAllocator)
{
	freeAllMemory();
	_init(elementSize, align, blockSize, elementAllocator, blockAllocator);
}

/* static */int hkFreeList::calcNumBlocks(const Block* blocks)
{
	int num = 0;
	while (blocks)
	{
		num++;
		blocks = blocks->m_next;
	}
	return num;
}

hkBool hkFreeList::_checkFreeBlocks()
{
	if( _calcNumFreeElements() != m_numFreeElements )
	{
		return false;
	}
	if( _calcTotalNumElements() != m_totalNumElements )
	{
		return false;
	}
	Element* cur = m_free;

	while (cur)
	{
		hkUint8* byteCur = (hkUint8*)cur;
		/// This must be the block
		/// Check the alignment

		if ((hk_size_t(byteCur) & (m_align - 1)) != 0)
		{
			return false;
		}

		hkBool found = false;
		// Find the block it is in
		Block* block = m_activeBlocks;
		while (block)
		{
			hkUint8* start = block->m_elements;
			hk_size_t maxEle = block->m_numElements;
			hkUint8* end = start + m_elementSize * maxEle;

			if (byteCur >= start && byteCur < end)
			{
				// Check its on the boundary correctly
				if ((byteCur - start) % m_elementSize !=0)
				{
					return false;
				}

				// Its found
				found = true;
				break;
			}
			block = block->m_next;
		}
		if (!found)
		{
			return false;
		}

		// Next
		cur = cur->m_next;
	}
	return true;
}

int hkFreeList::_freeBlocks(Block* cur)
{
	int numFreed = 0;
	// Free all the blocks in the list
	while (cur)
	{
		numFreed ++;

		m_totalNumElements -= cur->m_numElements;
		m_numFreeElements -= cur->m_numElements;

		Block* next = cur->m_next;
		
		if (m_blockAllocator)
		{
			// There are two underlying allocators

			// Free the payload
			m_elementAllocator->bufFree(cur->m_elementsAlloc, int(m_blockSize));
			// Free the block itself
			m_blockAllocator->blockFree(cur, int(sizeof(Block)));
		}
		else
		{
			// There is only the element allocator - and so the Block is stored before the payload
			m_elementAllocator->bufFree(cur, int(m_blockSize));
		}

		// Next
		cur = next;
	}
	return numFreed;
}

void hkFreeList::freeAllMemory()
{
	_freeBlocks(m_activeBlocks);
	m_activeBlocks = HK_NULL;
	_freeBlocks(m_freeBlocks);
	m_freeBlocks = HK_NULL;

	m_free = HK_NULL;
	// Reset the block size
	m_blockSize = 0;

	m_numFreeElements = 0;
	m_totalNumElements = 0;

	// There is no current block
	m_top = HK_NULL;
	m_blockEnd = HK_NULL;

	// Reset incremental collection
	m_lastIncrementalBlock = HK_NULL;
}

hkBool32 hkFreeList::_calcBlockFree(Block* block)
{
	// Okay I need to store all of the pointers found on the freelist 
	hkLocalBuffer<Element*> elements(int(block->m_numElements));
	int numEle = 0;

	hkUint8* payLoadStart = block->m_elements;
	hkUint8* payLoadEnd = payLoadStart + block->m_numElements * m_elementSize;

	Element** prev = &m_free;
	Element* ele = m_free;

	while (ele)
	{
		hkUint8* data = (hkUint8*)ele;
		if (data >= payLoadStart && data < payLoadEnd)
		{
			elements[numEle++] = ele;
			// Detatch it
			*prev = ele->m_next;
			// Next
			ele = ele->m_next;
		}
		else
		{
			prev = &ele->m_next;
			ele = ele->m_next;
		}
	}

	if (m_top >= payLoadStart && m_top < payLoadEnd)
	{
		// We are currently looking at the 'top' block, work out how many are remaining
		int onTop = int((m_blockEnd - m_top) / m_elementSize); 
		if (onTop + numEle == int(block->m_numElements))
		{
			// Stop this block being used as an allocation block
			m_top = HK_NULL;
			m_blockEnd = HK_NULL;
			return true;
		}
	}
	else
	{
		// We can free them all
		if (numEle == int(block->m_numElements))
		{
			return true;
		}
	}

	// Reorder the blocks elements to make contiguous
	if (numEle > 0)
	{
		if (numEle > 1)
		{
			hkSort(elements.begin(), numEle, _compareElements);
		}
		
		// Attach together
		Element* prevElem = elements[0];
		for (int i = 1; i < numEle; i++)
		{
			Element* element = elements[i];
			prevElem->m_next = element;
			prevElem = element;
		}

		// Attach to the list
		elements[numEle - 1]->m_next = m_free;
		m_free = elements[0];
	}

	return false;
}

void hkFreeList::_moveTopToFree()
{
	// Add any of the top block elements to the free list
	hkUint8* top = m_top;
	Element* head = m_free;
	while (top < m_blockEnd)
	{
		Element* e = reinterpret_cast<Element*>(top);
		e->m_next = head;
		head = e;
		top += m_elementSize;
	}

	// update top of list
	m_free = head;
	// The blocks are now in the freelist
	m_blockEnd = HK_NULL;
	m_top = HK_NULL;
}

hkBool hkFreeList::incrementalFindGarbage(int numBlocks, int& numBlocksOut)
{
	int numBlocksExamined = 0;
	while (numBlocksExamined < numBlocks)
	{
		if (m_lastIncrementalBlock)
		{
			Block* block = m_lastIncrementalBlock->m_next;
			if (block == HK_NULL)
			{
				numBlocksOut = numBlocksExamined;
				// I've hit the end
				m_lastIncrementalBlock = HK_NULL;
				return true;
			}

			// Examine this block
			if (_calcBlockFree(block))
			{
				// If it is free, I can but this block on the free block list
				m_lastIncrementalBlock->m_next = block->m_next;

				// Attach to the free block list
				block->m_next = m_freeBlocks;
				m_freeBlocks = block; 
			}
			else
			{
				m_lastIncrementalBlock = block;
			}
		}
		else
		{
			if (m_activeBlocks == HK_NULL)
			{
				numBlocksOut = numBlocksExamined;
				// We've hit the end
				m_lastIncrementalBlock = HK_NULL;
				return true;
			}

			Block* block = m_activeBlocks;
			if (_calcBlockFree(block))
			{
				// Remove from used list
				m_activeBlocks = block->m_next;

				// Add to the free list
				block->m_next = m_freeBlocks;
				m_freeBlocks = block;
			}
			else
			{
				m_lastIncrementalBlock = block;
			}
		}
		
		// A block was processed
		numBlocksExamined++;
	}

	numBlocksOut = numBlocksExamined;
	// Didn't hit the end - unless its now reset
	return (m_lastIncrementalBlock == HK_NULL);
}

hk_size_t hkFreeList::_calcNumFreeElements() const
{
	hk_size_t num = 0;
	Element* ele = m_free;
	while (ele)
	{
		num++;
		ele = ele->m_next;
	}
	// Don't forget, we've also got the remains of the 'top block'
	num += (m_blockEnd - m_top) / m_elementSize;

	// Add we count the elements that are in blocks that are free as having 'free elements' too.
	{
		Block* cur = m_freeBlocks;
		while (cur)
		{
			num += cur->m_numElements;
			cur = cur->m_next;
		}
	}
	
	return num;
}

hk_size_t hkFreeList::_calcTotalNumElements(Block* cur)
{
	hk_size_t num = 0;
	while (cur)
	{
		num += cur->m_numElements;
		cur = cur->m_next;
	}
	return num;
}

hk_size_t hkFreeList::_calculateBlockAllocatedSize(Block* cur) const
{
	hk_size_t size = 0;

	if (m_blockAllocator)
	{
		while (cur)
		{
			// The block
			size += m_blockAllocator->getAllocatedSize(cur, int(sizeof(Block)));
			// The payload
			size += m_elementAllocator->getAllocatedSize(cur->m_elementsAlloc, int(m_blockSize));

			// Next
			cur = cur->m_next;
		}
	}
	else
	{
		while (cur)
		{
			// The payload + block
			size += m_elementAllocator->getAllocatedSize(cur, int(m_blockSize));
			// Next
			cur = cur->m_next;
		}
	}
	return size;
}


hk_size_t hkFreeList::_calcTotalNumElements() const
{
	return _calcTotalNumElements(m_activeBlocks) + _calcTotalNumElements(m_freeBlocks);
}

int hkFreeList::findGarbage()
{
	// See if there are no blocks
	if ( m_activeBlocks == HK_NULL )
	{
		return 0;
	}

#if 0
	{
		hk_size_t numElements = getNumElements();
		hk_size_t numFree = getNumFreeElements();
	}
#endif

	// One thing that makes this trickier is we don't want to use hkArray etc, because it may allocate memory
	// from this freelist, and therefore screw things up royally. So we do all work in place.

	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkAlgorithm::ListElement, next) == HK_OFFSET_OF(Block, m_next) );
	m_activeBlocks = _sortByAllocationAddress( m_activeBlocks );

	_moveTopToFree();
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkAlgorithm::ListElement, next) == HK_OFFSET_OF(Element, m_next) );
	m_free = hkAlgorithm::sortList( m_free );

		// avoid double ptr messing and start-of-list checking
		// just fake it so we always have a previous and restore after the loop.
	Block activeBlocksHead; activeBlocksHead.m_next = m_activeBlocks;
	Element freeElementsHead; freeElementsHead.m_next = m_free;
		// lingo: block - the large allocation from a parent allocator
		// element - a slice of a block
	Block* prevBlock = &activeBlocksHead; // SINGLE prev block
	Element* prevElement = &freeElementsHead; // CHAIN of elements

	int numUnusedBlocksFound = 0;
	Element* curElement = m_free;

	for( Block* block = m_activeBlocks; block != HK_NULL; /*advance inside loop*/ )
	{
		hkUint8* cur = block->m_elements;
		hkUint8* end = cur + block->m_numElements * m_elementSize;

		// See if they all match up :)
		while( cur == reinterpret_cast<hkUint8*>(curElement) && cur < end)
		{
			cur += m_elementSize;
			curElement = curElement->m_next;
		}

		if( cur == end )
		{
			// They must have all matched :)
			// So we can add this block to the free blocks list
			numUnusedBlocksFound++;

			// Chain onto the list of free blocks
			prevBlock->m_next = block->m_next;
			block->m_next = m_freeBlocks;
			m_freeBlocks = block;
			// next block
			block = prevBlock->m_next;

			// Remove the Elements from the list, curElement is one past the end
			prevElement->m_next = curElement;
		}
		else
		{
			// Didn't match we need to skip thru all the entries remaining in this block
			while( curElement && reinterpret_cast<hkUint8*>(curElement) < end)
			{
				curElement = curElement->m_next;
			}
			// Catch up the prev ptr
			while( prevElement->m_next != curElement )
			{
				prevElement = prevElement->m_next;
			}

			prevBlock = block;
			block = block->m_next;
		}
	}

	m_free = freeElementsHead.m_next;
	m_activeBlocks = activeBlocksHead.m_next;

	// Reset incremental
	m_lastIncrementalBlock = HK_NULL;

	return numUnusedBlocksFound;
}

void hkFreeList::_walkMemoryBlockList( Block* block, MemoryWalkCallback callback, int pool, void* param )
{
	// Do all the free blocks
	for ( ;block != HK_NULL; block = block->m_next)
	{
		// All these elements are free

		hkUint8* cur = block->m_elements;
		hkUint8* end = cur + block->m_numElements * m_elementSize;
		while (cur < end)
		{
			// This ones free
			callback(cur, m_elementSize, false, pool, param);
			cur += m_elementSize;
		}
	}
}
void hkFreeList::walkMemory( MemoryWalkCallback callback, int pool, void* param )
{
	HK_ASSERT(0x342434,_calcNumFreeElements() == m_numFreeElements && _calcTotalNumElements() == m_totalNumElements);

	_walkMemoryBlockList(m_freeBlocks,callback,pool,param);

	// See if there are no blocks
	if ( m_activeBlocks == HK_NULL && m_free == HK_NULL)
	{
		HK_ASSERT(0xf23434,m_top == m_blockEnd);
		HK_ASSERT(0xf23434,m_numFreeElements == m_totalNumElements);
		return;
	}

	// One thing that makes this trickier is we don't want to use hkArray etc, because it may allocate memory
	// from this freelist, and therefore screw things up royally. So we do all the work in place.

	m_activeBlocks = _sortByAllocationAddress( m_activeBlocks );

	_moveTopToFree();
	m_free = hkAlgorithm::sortList( m_free );

	// Okay lets see how we match up
	Element* curElement = m_free; 
	for( Block* block = m_activeBlocks; block != HK_NULL; block = block->m_next )
	{ 
		hkUint8* cur = block->m_elements;
		hkUint8* end = cur + block->m_numElements * m_elementSize;

		// See if they all match up :)
		while (cur < end)
		{
			if (cur == reinterpret_cast<void*>(curElement) )
			{
				// This ones free
				callback(cur, m_elementSize, false, pool, param);
				curElement = curElement->m_next;
			}
			else
			{
				// This ones allocated
				callback(cur, m_elementSize, true, pool, param);
			}
			cur += m_elementSize;
		}
	}
}

int hkFreeList::addToSnapshot(hkMemorySnapshot& snap, hkMemorySnapshot::Status usage, int parentId)
{
	// To report free blocks in a consistent way, we need to understand whether this free list is used
	// to provide allocations to the user of the allocator who owns this (in which case usage will be
	// STATUS_USED and the free blocks will reported as UNUSED memory. If the freelist is used somehow
	// differently, we need to report the free blocks in a consistent way:
	// - if the freelist contains only overhead which is part of the allocator functionality, then
	//   the free blocks in the list will also be marked as OVERHEAD.
	// - if the freelist is used to store memory which is not currently available to the allocator
	//   user but which might be used in the future, then the free blocks in the list are also
	//   marked as UNUSED since they might end up being used by the allocator client.
	hkMemorySnapshot::Status freeStatus = 
		(usage == hkMemorySnapshot::STATUS_USED) ? hkMemorySnapshot::Status(hkMemorySnapshot::STATUS_UNUSED) : usage;
	// Do all the free blocks
	for ( Block* block = m_freeBlocks; block != HK_NULL; block = block->m_next)
	{
		// All these elements are free
		hkUint8* cur = block->m_elements;
		hkUint8* end = cur + block->m_numElements * m_elementSize;
		while (cur < end)
		{
			// This ones free
			snap.addItem( parentId, freeStatus, cur, int(m_elementSize) );
			cur += m_elementSize;
		}
	}

	// See if there are no blocks
	if ( m_activeBlocks == HK_NULL && m_free == HK_NULL)
	{
		HK_ASSERT(0xf23434,m_top == m_blockEnd);
		HK_ASSERT(0xf23434,m_numFreeElements == m_totalNumElements);
		return 0;
	}

	// One thing that makes this trickier is we don't want to use hkArray etc, because it may allocate memory
	// from this freelist, and therefore screw things up royally. So we do all the work in place.

	m_activeBlocks = _sortByAllocationAddress( m_activeBlocks );

	_moveTopToFree();
	m_free = hkAlgorithm::sortList( m_free );

	// Okay lets see how we match up
	Element* curElement = m_free; 
	for( Block* block = m_activeBlocks; block != HK_NULL; block = block->m_next )
	{ 
		//if( int ret = callback(block, sizeof(Block), hkMemoryWalk::MEM_OVERHEAD, poolId, param) ){return ret; }
		hkUint8* cur = block->m_elements;
		hkUint8* end = cur + block->m_numElements * m_elementSize;

		// See if they all match up :)
		while (cur < end)
		{
			if (cur == reinterpret_cast<void*>(curElement) )
			{
				snap.addItem(parentId, freeStatus, cur, int(m_elementSize) );
				curElement = curElement->m_next;
			}
			else
			{
				snap.addItem(parentId, usage, cur, int(m_elementSize) );
			}
			cur += m_elementSize;
		}
	}
	return 0;
}

int hkFreeList::freeAllFreeBlocks()
{
	// Free it all
	int numFreed = _freeBlocks(m_freeBlocks);
	m_freeBlocks = HK_NULL;
	return numFreed;
}

void* hkFreeList::addSpace()
{
	if (m_freeBlocks)
	{
		Block* block = m_freeBlocks;
		m_freeBlocks = block->m_next;

		// We have a block!
		_addBlockElements(block);

		// Attach to the list of active blocks
		block->m_next = m_activeBlocks;
		m_activeBlocks = block;

		// One less free
		m_numFreeElements--;
		// Return an allocation from the top
		void* data = (void*)m_top;
		m_top += m_elementSize;

		return data;
	}

	// we need to work out how big a block we are going to make
	if (m_blockSize <= 0)
	{
		// The element needs to have some size
		HK_ASSERT2(0x324a2434, m_elementSize > 0, "The freelist probably has not been initialized - either in init or ctor.");

		// Make 256 the minimum allocation
		hk_size_t numElements = 256 / m_elementSize;
		// We need at least one element
		numElements = (numElements < 1 )? 1 : numElements;
		// Work out the size taking into account alignment etc
		m_blockSize = bestBlockSize(numElements * m_elementSize, m_align);
	}

#if 0
	// No longer are blocks able to be different sizes, they are all the same size
	else
	{
		if (m_blockSize < m_maxBlockSize)
		{
			// We may need to make the block size bigger
			hk_size_t numElements = m_blockSize / m_elementSize;
			if (numElements < 8)
			{
				numElements = 8;
			}
			else
			{
				numElements = numElements + (numElements >> 1);
			}

			m_blockSize = bestBlockSize(numElements * m_elementSize,m_align);
		}
	}
#endif

	// Allocate the block

	Block* block;

	if (m_blockAllocator)
	{	
		// There is a block allocator
		block = reinterpret_cast<Block*>(m_blockAllocator->blockAlloc(int(sizeof(Block))));
		// Ouch allocation failed....
		if( block == HK_NULL )
		{
			return HK_NULL;
		}

		// Warning - if the amount of memory wasted on the 'Block' allocation is large. If so may be better to 
		// have an allocator, which farms off the allocations to a freelist.
	#ifdef HK_DEBUG
		{
			const int blockAllocSize = m_blockAllocator->getAllocatedSize(block, int(sizeof(Block)));
			if (blockAllocSize >= int(sizeof(Block)) * 3)
			{
				HK_WARN_ONCE(0x343a4324, "Allocator is wasting significant memory, for 'Block' allocations");
			}
		}
	#endif

		int elementsSize = int(m_blockSize);
		hkUint8* elements = reinterpret_cast<hkUint8*>(m_elementAllocator->bufAlloc(elementsSize));
		if (elements == HK_NULL)
		{
			m_blockAllocator->blockFree(block, int(sizeof(Block)));
			return HK_NULL;
		}
			
		block->m_elementsAlloc = elements;
		hkUint8* alignedElementsAlloc = reinterpret_cast<hkUint8*>((((hk_size_t)(elements)) + m_align - 1)&(~hk_size_t(m_align - 1)));
		
		if (alignedElementsAlloc != elements || elementsSize != int(m_blockSize))
		{
			// Need to work out the amount of elements
			block->m_numElements = ((elements + elementsSize) - alignedElementsAlloc) / m_elementSize;
			block->m_elements = alignedElementsAlloc;
		}
		else
		{
			// Its aligned and the alloc size was the same as requested -> so just use, and I already know the amount of elements
			block->m_numElements = m_numBlockElements;
			block->m_elements = elements;
		}
	}
	else
	{
		int allocSize = int(m_blockSize);
		hkUint8* allocPtr = reinterpret_cast<hkUint8*>(m_elementAllocator->bufAlloc(allocSize));
		if (allocPtr == HK_NULL)
		{
			return HK_NULL;
		}

		block = (Block*)allocPtr;

		hkUint8* elements = (hkUint8*)(block + 1);
		hkUint8* alignedElementsAlloc = reinterpret_cast<hkUint8*>((((hk_size_t)(elements)) + m_align - 1) & (~hk_size_t(m_align - 1)));

		// Need to work out the amount of elements

		// Don't need because there isn't an elements alloc - its part of the block alloc
		block->m_elementsAlloc = HK_NULL;				
		block->m_numElements = ((allocPtr + allocSize) - alignedElementsAlloc) / m_elementSize;
		block->m_elements = alignedElementsAlloc;
	}

	// Add all of the elements
	_addBlockElements(block);

	// attach the 'used' block list. This is a fair assumption because presumably the request
	// for more space means, that one of the freeblocks we just added is going to be used
	block->m_next = m_activeBlocks;
	m_activeBlocks = block;

	// Fix up the counts
	m_totalNumElements += block->m_numElements;
	m_numFreeElements += block->m_numElements - 1;

	// Return an allocation from the top
	void* data = (void*)m_top;
	m_top += m_elementSize;
	return data;
}

hk_size_t hkFreeList::bestBlockSize(hk_size_t elementSpace,hk_size_t align)
{
	if (align <= 16) return elementSpace + sizeof(Block);
		/// Okay this should be the max amount I should have to align with otherwise
	return sizeof(Block) + (align - 16) + elementSpace;
}

void hkFreeList::freeAll()
{
	// All the elements are now free
	m_numFreeElements = m_totalNumElements;

	// No active blocks means nothing could be allocated
	if (m_activeBlocks==HK_NULL)
	{
		return;
	}

	// First mark all the blocks
	m_free = HK_NULL;

	// We need to add the blocks to the free blocks
	Block* cur = m_activeBlocks;
	while (cur->m_next != HK_NULL)
	{
		cur = cur->m_next;
	}

	// Concat the left free blocks
	cur->m_next = m_freeBlocks;

	// The blocks are now the free blocks
	m_freeBlocks = m_activeBlocks;
	m_activeBlocks = HK_NULL;

	// There is no top
	m_top = HK_NULL;
	m_blockEnd = HK_NULL;

	// Reset incremental
	m_lastIncrementalBlock = HK_NULL;
}

void hkFreeList::getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& stats ) const
{
	// Total amount of allocated space
	stats.m_allocated = _calculateBlockAllocatedSize(m_activeBlocks) + _calculateBlockAllocatedSize(m_freeBlocks);

	hk_size_t numFreeElements = getNumFreeElements();
	hk_size_t numElements = getTotalNumElements();

	//HK_ASSERT(0x324234,numFreeElements==_calcNumFreeElements());
	//HK_ASSERT(0x324234,numElements==_calcTotalNumElements());

	// The amount thats available
	stats.m_available = numFreeElements * m_elementSize;
	// The amount thats used
	stats.m_inUse = (numElements - numFreeElements) * m_elementSize;

	stats.m_largestBlock = m_elementSize;
	stats.m_totalAvailable = stats.m_available;
}


int hkFreeList::_getSortedBlockHeads( const Block** heads, int numHeads )
{
	HK_ASSERT(0x715ba367, numHeads == 2);
	m_activeBlocks = hkAlgorithm::sortList(m_activeBlocks);
	m_freeBlocks = hkAlgorithm::sortList(m_freeBlocks);
	int nh = 0;
	if( m_activeBlocks )
	{
		heads[nh++] = m_activeBlocks;
	}
	if( m_freeBlocks )
	{
		heads[nh++] = m_freeBlocks;
	}
	return nh;
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
