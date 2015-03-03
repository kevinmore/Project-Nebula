/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Allocator/Linear/hkLinearAllocator.h>

// -------------------------------- Methods --------------------------------- //

hkLinearAllocator::hkLinearAllocator(hkUint8* bufferPtr, int bufferSize)
	: m_index(0), m_bufferPtr(bufferPtr), m_bufferSize(bufferSize), m_inUse(0), m_peakInUse(0), m_criticalSection(1000)
{}

hkLinearAllocator::~hkLinearAllocator()
{}

HK_FORCE_INLINE void* hkLinearAllocator::blockAlloc(int numBytes)
{
	hkCriticalSectionLock lock( &m_criticalSection );

	hkUint32 i = m_index;
	int numBytesAlign = HK_NEXT_MULTIPLE_OF(16, numBytes);
	if( m_index + numBytesAlign > m_bufferSize ) 
	{
		// not enough static space
		HK_BREAKPOINT(0);
	}
	// update statistics
	m_inUse += numBytes;
	if(m_inUse > m_peakInUse) 
	{
		m_peakInUse = m_inUse;
	}
	m_index += numBytesAlign;
	return &m_bufferPtr[i];
}

HK_FORCE_INLINE void hkLinearAllocator::blockFree(void* p, int numBytes)
{
	hkCriticalSectionLock lock( &m_criticalSection );

	// Static memory is never freed
	m_inUse -= numBytes;
}

HK_FORCE_INLINE void hkLinearAllocator::getMemoryStatistics( MemoryStatistics& u ) const
{
	hkCriticalSectionLock lock( &m_criticalSection );

	u.m_allocated = m_bufferSize;
	u.m_inUse = m_inUse;
	u.m_peakInUse = m_peakInUse;
	u.m_available = m_bufferSize - m_index;
	u.m_totalAvailable = u.m_available;
	u.m_largestBlock = u.m_available;
}

HK_FORCE_INLINE int hkLinearAllocator::getAllocatedSize(const void* obj, int numBytes) const
{
	return numBytes;
}

// -------------------------------------------------------------------------- //

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
