/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Memory/Allocator/FreeList/hkFreeListAllocator.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>
#include <Common/Base/Memory/Allocator/Pooled/hkPooledAllocator.h>

static void testZeroSizeAlloc(hkMemoryAllocator& a)
{
	{
		void* p = a.blockAlloc(0);
		a.blockFree(p,0);
	}
	{
		int size = 0;
		void* p = a.bufAlloc(size);
		a.bufFree(p,size);
	}
}


int allocators_test()
{
	// test builtin ones to catch any new ones which haven't been added here yet
	{
		hkMemoryRouter& m = hkMemoryRouter::getInstance();
		testZeroSizeAlloc( m.heap() );
		testZeroSizeAlloc( m.temp() );
		testZeroSizeAlloc( m.debug() );
		testZeroSizeAlloc( m.stack() );
	}
	{
		hkMallocAllocator m;
		testZeroSizeAlloc(m);
	}
	{
		hkMallocAllocator mal;
		hkThreadMemory thr;
		thr.setMemory(&mal);
		testZeroSizeAlloc(thr);
	}
	{
		hkMallocAllocator mal;
		hkFreeListAllocator fre(&mal, HK_NULL);
		testZeroSizeAlloc(fre);
	}
	{
		hkMallocAllocator mal;
		hkPooledAllocator p;
		p.init(&mal, &mal, &mal, 2048);
		testZeroSizeAlloc(p);
	}
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(allocators_test, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

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
