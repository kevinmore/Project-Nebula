/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Router/hkMemoryRouter.h>

namespace
{
	struct RouterSizeInfo
	{
		void* m_allocator;
		hkInt32 m_size;
		hkInt32 m_alignedOffset; // only used in aligned{Allocate,Deallocate}
		#if HK_POINTER_SIZE==4
			int pad16;
		#endif
	};
	HK_COMPILE_TIME_ASSERT(sizeof(RouterSizeInfo)==16);
}

HK_THREAD_LOCAL(hkMemoryRouter*) hkMemoryRouter::s_memoryRouter;

void HK_CALL hkMemoryRouter::replaceInstance( hkMemoryRouter* a )
{
	HK_THREAD_LOCAL_SET(s_memoryRouter, a);
}

hkMemoryRouter::hkMemoryRouter()
: m_temp(HK_NULL)
, m_heap(HK_NULL)
, m_debug(HK_NULL)
, m_solver(HK_NULL)
, m_refObjLocalStore(0)
{
}

void* HK_CALL hkMemoryRouter::alignedAlloc( Allocator& b, int nbytes, int alignment )
{
	// allocate enough to hold the nbytes, the size info and the alignment window
	char* unaligned = reinterpret_cast<char*>(	b.blockAlloc(alignment + nbytes + hkSizeOf(RouterSizeInfo)) );

	HK_ASSERT2( 0x30a343ed, unaligned != HK_NULL, "Out of memory" );

	// the aligned memory is the nearest aligned block, taking into account that the
	// sizeinfo which is placed just before the returned pointer.
	char* aligned = reinterpret_cast<char*>( HK_NEXT_MULTIPLE_OF( alignment, hkUlong(unaligned+hkSizeOf(RouterSizeInfo))) );

	// store the sizeinfo just before the returned pointer
	{
		RouterSizeInfo* x = reinterpret_cast<RouterSizeInfo*>(aligned) - 1;
		x->m_allocator = &b;
		x->m_size = nbytes + alignment;
		x->m_alignedOffset = (int)(aligned - unaligned);
	}
	return static_cast<void*>(aligned);	
}

void HK_CALL hkMemoryRouter::alignedFree( Allocator& b, void* p )
{
	if(p)
	{
		RouterSizeInfo* x = static_cast<RouterSizeInfo*>(p) - 1;
		x->m_allocator = HK_NULL;
		char* unaligned = reinterpret_cast<char*>(p) - x->m_alignedOffset;
		b.blockFree( static_cast<void*>(unaligned), x->m_size + hkSizeOf(RouterSizeInfo) );
	}
}

void* HK_CALL hkMemoryRouter::easyAlloc( Allocator& b, int nbytes )
{
	RouterSizeInfo* x = static_cast<RouterSizeInfo*>( b.blockAlloc(nbytes + hkSizeOf(RouterSizeInfo)) );
	x->m_allocator = &b;
	x->m_size = nbytes;
	return static_cast<void*>(x+1);
}

/* static */hk_size_t HK_CALL hkMemoryRouter::getEasyAllocSize(Allocator& b, const void* ptr)
{
	const RouterSizeInfo* x = ((const RouterSizeInfo*)ptr) - 1;
	return x->m_size;
}

/* static */const void* HK_CALL hkMemoryRouter::getEasyAllocStartAddress(Allocator& b, const void* ptr)
{
	const RouterSizeInfo* x = ((const RouterSizeInfo*)ptr) - 1;
	return (const void*)x;
}

void HK_CALL hkMemoryRouter::easyFree( Allocator& b, void* p )
{
	if (p) // as we allowed by convention to deallocate NULL
	{
		RouterSizeInfo* x = static_cast<RouterSizeInfo*>(p) - 1;
		x->m_allocator = HK_NULL;
		b.blockFree( static_cast<void*>(x),	x->m_size + hkSizeOf(RouterSizeInfo) );
	}
}

void hkMemoryRouter::resetPeakMemoryStatistics()
{
	m_debug->resetPeakMemoryStatistics();
	m_heap->resetPeakMemoryStatistics();
	m_solver->resetPeakMemoryStatistics();
	m_temp->resetPeakMemoryStatistics();
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
