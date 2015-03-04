/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Checking/hkPaddedAllocator.h>

hkPaddedAllocator::hkPaddedAllocator()
	: m_next(HK_NULL)
{
}

void hkPaddedAllocator::init( hkMemoryAllocator* next, Cinfo* cinfoPtr )
{
	m_next = next;
	Cinfo defaultCinfo;
	m_cinfo = cinfoPtr ? *cinfoPtr : defaultCinfo;
	HK_ASSERT(0x7f05fa8d, m_cinfo.m_numQuadsPad >= 1 );
}

void hkPaddedAllocator::quit()
{
	m_next = HK_NULL;
}

void* hkPaddedAllocator::blockAlloc(int numBytes)
{
	// example numBytes = 11, pad = 16:
	// - numBytesAligned = 16
	// - allocSize = 16 + 2*16 = 48
	// - numBodyWords = (11+4-1)/4 = 3 
	// pad and scrub:
	// [XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX] -> [PPPP PPPP PPPP PPPP XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX]
	// [PPPP PPPP PPPP PPPP XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX] -> [PPPP PPPP PPPP PPPP BBBB BBBB BBBB XXXX XXXX XXXX XXXX XXXX]
	// [PPPP PPPP PPPP PPPP BBBB BBBB BBBB XXXX XXXX XXXX XXXX XXXX] -> [PPPP PPPP PPPP PPPP BBBB BBBB BBBA AAAA XXXX XXXX XXXX XXXX]
	// [PPPP PPPP PPPP PPPP BBBB BBBB BBBA AAAA XXXX XXXX XXXX XXXX] -> [PPPP PPPP PPPP PPPP BBBB BBBB BBBA AAAA PPPP PPPP PPPP PPPP]
	int numBytesAligned = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, numBytes);
	unsigned pad = sizeof(hkQuadReal) * m_cinfo.m_numQuadsPad; // one side
	int allocSize = numBytesAligned + 2*pad;
	m_allocated += allocSize;
	m_inUse += numBytes;
	void* p = m_next->blockAlloc( allocSize );
	int numBodyWords = (numBytes+sizeof(hkUint32)-1) / sizeof(hkUint32);
	// pad and scrub
	hkString::memSet4( p, m_cinfo.m_padPattern, pad/sizeof(hkUint32) );
	hkString::memSet4( hkAddByteOffset(p,pad), m_cinfo.m_bodyPattern, numBodyWords );
	hkString::memSet( hkAddByteOffset(p,pad+numBytes), m_cinfo.m_alignPattern, numBytesAligned-numBytes );
	hkString::memSet4( hkAddByteOffset(p,pad+numBytesAligned), m_cinfo.m_padPattern + 1, pad/sizeof(hkUint32) );
	return hkAddByteOffset(p,pad);
}

void hkPaddedAllocator::blockFree(void* p, int numBytes)
{
	int pad = sizeof(hkQuadReal) * m_cinfo.m_numQuadsPad;
	int numBytesAligned = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, numBytes);
	// check the aligned part
	hkUint32* ps4 = (hkUint32*)hkAddByteOffset(p, -pad);
	hkUint32* pe4 = (hkUint32*)hkAddByteOffset(p, numBytesAligned);
	{
		for( int i = 0; i < 4*m_cinfo.m_numQuadsPad; ++i )
		{
			if( ps4[i] != m_cinfo.m_padPattern ) HK_BREAKPOINT(0); 
			if( pe4[i] != m_cinfo.m_padPattern + 1 ) HK_BREAKPOINT(0);
		}
	}

	// check the unaligned part between the end and pad
	// they were filled with alignPattern, not padPattern or bodyPattern
	{
		const hkUint8* p8 = static_cast<const hkUint8*>(p);
		for( int i = numBytes; i < numBytesAligned; ++i )
		{
			if( p8[i] != m_cinfo.m_alignPattern )
			{
				HK_BREAKPOINT(0);
			}
		}
	}

	int freeSize = numBytesAligned + 2*pad;
	m_allocated -= freeSize;
	m_inUse -= numBytes;
	hkString::memSet4( ps4, m_cinfo.m_freePattern, 2*4*m_cinfo.m_numQuadsPad); // 2=pre+post 4=4 ints per quad
	m_next->blockFree( ps4, freeSize );
}

hkBool32 hkPaddedAllocator::isOk( const void* p, int numBytes ) const
{
	//
	// todo - this is a copy of the above block with returns instead of breakpoints
	//

	int pad = sizeof(hkQuadReal) * m_cinfo.m_numQuadsPad;
	int numBytesAligned = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, numBytes); 
	// check the aligned part
	const hkUint32* ps4 = (const hkUint32*)hkAddByteOffsetConst(p, -pad);
	const hkUint32* pe4 = (const hkUint32*)hkAddByteOffsetConst(p, numBytesAligned);
	{
		for( int i = 0; i < 4*m_cinfo.m_numQuadsPad; ++i )
		{
			if( ps4[i] != m_cinfo.m_padPattern ) return false; 
			if( pe4[i] != m_cinfo.m_padPattern + 1 ) return false;
		}
	}
	// check the unaligned part between the end and pad
	// they were filled with alignPattern, not padPattern or bodyPattern because of alignment
	{
		const hkUint8* p8 = static_cast<const hkUint8*>(p);
		for( int i = numBytes; i < numBytesAligned; ++i )
		{
			if( p8[i] != m_cinfo.m_alignPattern )
			{
				return false;
			}
		}
	}
	return true;
}

void hkPaddedAllocator::getMemoryStatistics( hkMemoryAllocator::MemoryStatistics& u ) const
{
	m_next->getMemoryStatistics( u );
	u.m_allocated = m_allocated;
	u.m_inUse = m_inUse;
	u.m_available = 0;
}

hkPaddedAllocator::Allocation hkPaddedAllocator::getUnderlyingAllocation(const void* obj, int numBytes) const
{
	int pad = sizeof(hkQuadReal) * m_cinfo.m_numQuadsPad;
	int numBytes16 = HK_NEXT_MULTIPLE_OF(16, numBytes);
	Allocation ret;
	ret.address = hkAddByteOffsetConst(obj, -pad);
	ret.size = numBytes16 + 2*pad;
	return ret;
}

int hkPaddedAllocator::getAllocatedSize( const void* obj, int numBytes ) const
{
	int pad = sizeof(hkQuadReal) * m_cinfo.m_numQuadsPad;
	int numBytesAligned = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, numBytes);
	const void* unpadded = hkAddByteOffsetConst(obj, -pad);
	return m_next->getAllocatedSize( unpadded, numBytesAligned + 2*pad ) - 2*pad;
}

void hkPaddedAllocator::setScrubValues(hkUint32 bodyValue, hkUint32 freeValue)
{
	m_cinfo.m_bodyPattern = bodyValue;
	m_cinfo.m_freePattern = freeValue;
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
