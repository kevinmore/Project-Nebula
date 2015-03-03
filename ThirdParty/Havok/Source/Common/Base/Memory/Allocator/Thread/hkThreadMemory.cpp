/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Thread/hkThreadMemory.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

void* hkThreadMemory::bufAlloc( int& reqNumInOut )
{
	return blockAlloc(reqNumInOut);
}
void hkThreadMemory::bufFree( void* p, int numElem )
{
	return blockFree(p, numElem);
}
void* hkThreadMemory::bufRealloc( void* pold, int oldNum, int& reqNumInOut )
{
	void* pnew = hkThreadMemory::blockAlloc(reqNumInOut);
	if( pnew )
	{
		hkMemUtil::memCpy( pnew, pold, hkMath::min2(reqNumInOut,oldNum));
	}
	hkThreadMemory::blockFree(pold, oldNum);
	return pnew;
}
void hkThreadMemory::blockAllocBatch(void** ptrsOut, int numPtrs, int blockSize)
{
	if( blockSize <= MEMORY_MAX_SIZE_LARGE_BLOCK)
	{
		for( int i = 0; i < numPtrs; ++i )
		{
			ptrsOut[i] = blockAlloc(blockSize);
		}
	}
	else
	{
		m_memory->blockAllocBatch(ptrsOut, numPtrs, blockSize );
	}
}
void hkThreadMemory::blockFreeBatch(void** ptrsIn, int numPtrs, int blockSize)
{
	if( blockSize <= MEMORY_MAX_SIZE_LARGE_BLOCK)
	{
		for( int i = 0; i < numPtrs; ++i )
		{
			blockFree(ptrsIn[i], blockSize);
		}
	}
	else
	{
		m_memory->blockFreeBatch(ptrsIn, numPtrs, blockSize );
	}
}

HK_ON_DEBUG( hkCriticalSection s_debugLock(0) );

hkThreadMemory::hkThreadMemory()
{
	initLookupTables();
	setMemory(HK_NULL);
}

hkThreadMemory::hkThreadMemory(hkMemoryAllocator* memoryInstance)
{
	initLookupTables();
	setMemory(memoryInstance);
}
	

void hkThreadMemory::getMemoryStatistics( hkThreadMemory::MemoryStatistics& u ) const
{
	u.m_allocated = -1;
	u.m_inUse = -1;
	u.m_peakInUse = -1;
	hkLong available = 0;
	for( int i = 0; i < MEMORY_MAX_ALL_ROW; ++i )
	{
		const FreeList& f = m_free_list[i];
		available += f.m_numElem * m_row_to_size_lut[i];
	}
	u.m_available = available;
	u.m_totalAvailable = -1;
	u.m_largestBlock = -1;
}


void hkThreadMemory::initLookupTables()
{
	//XXX make shared readonly
	int i;
	for(i = 0; i <= (MEMORY_MAX_SIZE_SMALL_BLOCK >> MEMORY_SMALL_BLOCK_RSHIFT_BITS); ++i )
	{
		const int size = i << MEMORY_SMALL_BLOCK_RSHIFT_BITS;
		int row = constSizeToRow(size);
		m_small_size_to_row_lut[ i ] = static_cast<char>(row);
		m_row_to_size_lut[row] = size;
	}
	for(i = 0; i < (MEMORY_MAX_SIZE_LARGE_BLOCK >> MEMORY_LARGE_BLOCK_RSHIFT_BITS); ++i)
	{
		int size = (i+1) << MEMORY_LARGE_BLOCK_RSHIFT_BITS;
		int row = constSizeToRow(size);
		m_large_size_to_row_lut[i] = row;
		m_row_to_size_lut[row] = size;
	}
}

hk_size_t hkThreadMemory::getCachedSizeUnchecked()
{
	int sum = 0;
	for (int i =0; i < MEMORY_MAX_ALL_ROW;i++)
	{
		int size = rowToSize(i);
		int numFree = m_free_list[i].m_numElem;
		sum += size * numFree;
	}
	return sum;
}

void hkThreadMemory::setMemory( hkMemoryAllocator* memoryInstance, int maxNumElemsOnFreeList )
{
	if (memoryInstance)
	{
		m_memory = memoryInstance;
		m_maxNumElemsOnFreeList = maxNumElemsOnFreeList;
	}
	else
	{
		m_memory = HK_NULL;
	}
}


void hkThreadMemory::releaseCachedMemory()
{
	for(int rowIndex = MEMORY_MAX_ALL_ROW-1; rowIndex >= 0; --rowIndex )
	{
		if( m_free_list[rowIndex].m_numElem )
		{
			HK_ON_CPU( clearRow(rowIndex)); //CLEAR_WHOLE_ROW
			HK_ON_SPU( onRowFull(rowIndex, HK_NULL));
		}
	}
}

void* hkThreadMemory::onRowEmpty(int rowIndex)
{
	if (m_maxNumElemsOnFreeList==0)
	{
		// If no caching, just return the allocation
		return m_memory->blockAlloc( m_row_to_size_lut[rowIndex] );
	}
	else
	{
		HK_COMPILE_TIME_ASSERT( BATCH_SIZE >= 1 );
		void* ptrs[BATCH_SIZE];

		int size = m_maxNumElemsOnFreeList<BATCH_SIZE?m_maxNumElemsOnFreeList:BATCH_SIZE;

		m_memory->blockAllocBatch(ptrs, size, m_row_to_size_lut[rowIndex] );
		FreeList& flist = m_free_list[rowIndex];
		for( int i = 1; i < size; ++i )
		{
			flist.put(ptrs[i]);
		}
		return ptrs[0];
	}
}

void hkThreadMemory::onRowFull(int rowIndex, void* p)
{
	int blockSize = m_row_to_size_lut[rowIndex];

	if (m_maxNumElemsOnFreeList == 0)
	{
		/// No caching
		m_memory->blockFree(p,blockSize);
	}
	else
	{
		FreeList& flist = m_free_list[rowIndex];
		int numBlocks = flist.m_numElem;

		int rowNeeded =  m_maxNumElemsOnFreeList / 2;
		while( numBlocks > rowNeeded )
		{
			void* ptrs[BATCH_SIZE];
			int n = BATCH_SIZE  < numBlocks-rowNeeded ? BATCH_SIZE : numBlocks-rowNeeded;
			int i;
			for( i = 0; i < n; ++i )
			{
				ptrs[i] = flist.get();
			}
			numBlocks -= n;
			m_memory->blockFreeBatch(ptrs, n, blockSize);
		}

		/// Put p on the list
		m_free_list[rowIndex].put(p);
	}
}

void hkThreadMemory::clearRow(int rowIndex)
{
	int blockSize = m_row_to_size_lut[rowIndex];
	FreeList& flist = m_free_list[rowIndex];
	int numBlocks = flist.m_numElem;

	while( numBlocks > 0 )
	{
		void* ptrs[BATCH_SIZE];
		int n = BATCH_SIZE  < numBlocks ? BATCH_SIZE : numBlocks;
		int i;
		for( i = 0; i < n; ++i )
		{
			ptrs[i] = flist.get();
		}
		numBlocks -= n;
		m_memory->blockFreeBatch(ptrs, n, blockSize);
	}
}

void* hkThreadMemory::blockAlloc(int nbytes)
{
    if ( nbytes <= MEMORY_MAX_SIZE_LARGE_BLOCK)
	{
		int row = getRow(nbytes);
		if( void* p = m_free_list[row].get() )
		{
			return p;
		}
		return onRowEmpty( row );
	}
	else
	{
		return m_memory->blockAlloc( nbytes );
	}
}

void  hkThreadMemory::blockFree(void* p, int nbytes)
{
	if (p)
	{
        if ( nbytes <= MEMORY_MAX_SIZE_LARGE_BLOCK )
		{
			int row = getRow(nbytes);
			if ( m_free_list[row].m_numElem >= m_maxNumElemsOnFreeList )
			{
				onRowFull(row,p);
			}
            else
            {
				m_free_list[row].put(p);
            }
		}
		else
		{
			m_memory->blockFree(p, nbytes );
		}
	}
}

int hkThreadMemory::getAllocatedSize(const void* obj, int numBytes) const
{
	if ( numBytes <= MEMORY_MAX_SIZE_LARGE_BLOCK )
	{
		int row = getRow(numBytes);
		numBytes = m_row_to_size_lut[row];
	}
	return m_memory->getAllocatedSize( obj, numBytes );
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
