/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

// this
#include <Common/Base/Memory/Allocator/FreeList/hkFixedSizeAllocator.h>

void hkFixedSizeAllocator::init(hk_size_t blockSize, hk_size_t align, hk_size_t allocatorBlockSize, hkMemoryAllocator* allocator)
{
	// Set up to use the underlying allocator
	m_freeList.init(blockSize, align, allocatorBlockSize, allocator, HK_NULL);
}

void* hkFixedSizeAllocator::blockAlloc( int numBytes )
{
	HK_ASSERT(0x323a2432, numBytes == int(m_freeList.getElementSize()));
	return m_freeList.alloc();
}

void hkFixedSizeAllocator::blockFree( void* p, int numBytes )
{
	HK_ASSERT(0x343a2432, numBytes == int(m_freeList.getElementSize()));
	return m_freeList.free(p);
}

void* hkFixedSizeAllocator::bufAlloc( int& reqNumBytesInOut )
{
	HK_ASSERT(0x343a2432, reqNumBytesInOut == int(m_freeList.getElementSize()));
	return m_freeList.alloc();
}

void hkFixedSizeAllocator::bufFree( void* p, int numBytes )
{
	HK_ASSERT(0x343a2432, numBytes == int(m_freeList.getElementSize()));
	m_freeList.free(p);
}

void* hkFixedSizeAllocator::bufRealloc( void* pold, int oldNumBytes, int& reqNumBytesInOut )
{
	HK_ASSERT(0x343a2432, oldNumBytes <= int(m_freeList.getElementSize()) && reqNumBytesInOut <= int(m_freeList.getElementSize()));
	return pold;
}

void hkFixedSizeAllocator::blockAllocBatch(void** ptrsOut, int numPtrs, int blockSize)
{
	HK_ASSERT(0x3243a432, blockSize == int(m_freeList.getElementSize()));
	m_freeList.allocBatch(ptrsOut, numPtrs);
}

void hkFixedSizeAllocator::blockFreeBatch(void** ptrsIn, int numPtrs, int blockSize)
{
	HK_ASSERT(0x32432432, blockSize == int(m_freeList.getElementSize()));
	m_freeList.freeBatch(ptrsIn, numPtrs);
}

void hkFixedSizeAllocator::getMemoryStatistics( MemoryStatistics& u ) const
{
	m_freeList.getMemoryStatistics(u);
}

int hkFixedSizeAllocator::getAllocatedSize(const void* obj, int nbytes) const
{
	HK_ASSERT(0x32432432, nbytes <= int(m_freeList.getElementSize()));
	return int(m_freeList.getElementSize());
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
