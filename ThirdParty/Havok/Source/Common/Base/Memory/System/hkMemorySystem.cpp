/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Memory/Allocator/FreeList/hkFreeListAllocator.h>

hkMemorySystem::FrameInfo::FrameInfo(int solverBufferSize)
: m_solverBufferSize(solverBufferSize)
{
}

hkMemorySystem* hkMemorySystem::s_instance;

void HK_CALL hkMemorySystem::replaceInstance( hkMemorySystem* m )
{
	s_instance = m;
}

hkMemorySystem& HK_CALL hkMemorySystem::getInstance()
{
	return *s_instance;
}

hkMemorySystem* HK_CALL hkMemorySystem::getInstancePtr()
{
	return s_instance;
}

hkMemorySystem::~hkMemorySystem()
{
}

static void printDetailedStatistics( const char* what, hkOstream& ostr, hkMemoryAllocator::MemoryStatistics& u, hk_size_t softLimit = hkFreeListAllocator::SOFT_LIMIT_MAX )
{
	hk_size_t allocated = hkMath::max2( u.m_allocated, u.m_inUse );
	int overHead = int(allocated - u.m_inUse - u.m_available);
	hkReal relOvhd = overHead / hkReal(u.m_inUse + u.m_available);

	if ( softLimit == 0 || softLimit == hkFreeListAllocator::SOFT_LIMIT_MAX )
	{
		softLimit = allocated;
	}

	hk_size_t available = hk_size_t((1.0f - relOvhd) * softLimit);
	hk_size_t	reserved = allocated - softLimit;

	hkReal relFactor = 100.0f / hkMath::min2( softLimit, available);

	ostr.printf("\t%s:\n", what);
	ostr.printf("            totalSize:   %10i\n", allocated);
	if ( softLimit < allocated )
	{
		ostr.printf("\n            - reserved:  %10i (%4.1f%%)\n", reserved,  100.0f * reserved  / available);
		ostr.printf("            = softlimit: %10i\n", softLimit);
	}
	ostr.printf("            - overhead:  %10i (%4.1f%%)\n", overHead,  100.0f * overHead / softLimit);
	ostr.printf("            = available: %10i\n", available);

	ostr.printf("              used:      %10i (%4.1f%%)\n",    u.m_inUse,		u.m_inUse * relFactor );
	ostr.printf("              free:      %10i (%4.1f%%)\n",    u.m_available,	u.m_available * relFactor );
	ostr.printf("              peak:      %10i (%4.1f%%)\n\n",  u.m_peakInUse,	u.m_peakInUse * relFactor );
}

void hkMemorySystem::garbageCollectThread(hkMemoryRouter&)
{
}

void hkMemorySystem::garbageCollectShared()
{
}

void hkMemorySystem::garbageCollect()
{
	garbageCollectThread(hkMemoryRouter::getInstance());
	garbageCollectShared();
}

hkResult hkMemorySystem::setHeapSoftLimit(int nbytes)
{
	return HK_FAILURE;
}

int hkMemorySystem::getHeapSoftLimit() const
{
	return -1;
}

bool hkMemorySystem::solverCanAllocSingleBlock( int numBytes )
{
	return true;
}

bool hkMemorySystem::heapCanAllocTotal( int numBytes )
{
	return true;
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
