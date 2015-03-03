/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/hkMemoryAllocator.h>

void HK_CALL HK_ASSERT_OBJECT_SIZE_OK_FUNC(hk_size_t nbytes);
void HK_CALL HK_ASSERT_OBJECT_SIZE_OK_FUNC(hk_size_t nbytes)
{
	// object size must be representable in 15 bits.
	HK_ASSERT2(0x13aa0880,  nbytes <= (hkUint16(-1)>>1), "Object is too big");
}

hkMemoryAllocator::~hkMemoryAllocator()
{
}

void* hkMemoryAllocator::bufAlloc( int& reqBytesInOut )
{
	return blockAlloc(reqBytesInOut);
}
void hkMemoryAllocator::bufFree( void* p, int numBytes)
{
	return blockFree(p, numBytes);
}
void* hkMemoryAllocator::bufRealloc( void* pold, int oldNumBytes, int& reqBytesInOut )
{
	void* pnew = bufAlloc(reqBytesInOut);
	if ( pnew )
	{
		hkMemUtil::memCpy(pnew, pold, hkMath::min2(oldNumBytes,reqBytesInOut));
	}
	bufFree(pold, oldNumBytes);
	return pnew;
}

void hkMemoryAllocator::blockAllocBatch(void** ptrsOut, int numPtrs, int blockSize)
{
	for( int i = 0; i < numPtrs; ++i )
	{
		ptrsOut[i] = blockAlloc(blockSize);
	}
}
void hkMemoryAllocator::blockFreeBatch(void** ptrsIn, int numPtrs, int blockSize)
{
	for( int i = 0; i < numPtrs; ++i )
	{
		if( ptrsIn[i] )
		{
			blockFree(ptrsIn[i], blockSize);
		}
	}
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
