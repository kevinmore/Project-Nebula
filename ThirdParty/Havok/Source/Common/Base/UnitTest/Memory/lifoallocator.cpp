/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Memory/Allocator/Checking/hkLeakDetectAllocator.h>

namespace
{
	struct Alloc{ void* p; int s; };

		// make nalloc allocations of various sizes up to maxSize
	static void randomAllocs(hkMemoryAllocator& allocator, hkArray<Alloc>::Temp& allocs, int nalloc, hkPseudoRandomGenerator& prng, int maxSize )
	{
		for( int i = 0; i < nalloc; ++i )
		{
			Alloc& a = allocs.expandOne();
			a.s = prng.getRandInt16(maxSize) + 1;
			a.p = allocator.blockAlloc( a.s );
		}
	}

	void pureLifoTest(hkLifoAllocator& lifo, hkPseudoRandomGenerator& prng, int numAllocs, int maxSize )
	{
		hkArray<Alloc>::Temp allocs;
		randomAllocs(lifo, allocs, numAllocs, prng, maxSize);
		for( int i = allocs.getSize()-1; i >= 0; --i )
		{
			lifo.blockFree( allocs[i].p, allocs[i].s );
		}
	}

		// free blocks in random order, pretty much the worst case
	void nonLifoTest(hkLifoAllocator& lifo, hkPseudoRandomGenerator& prng, int numAllocs, int maxSize )
	{
		hkArray<Alloc>::Temp allocs;
		randomAllocs(lifo, allocs, numAllocs, prng, maxSize);
		while( allocs.getSize() )
		{
			int i = prng.getRandInt16(allocs.getSize());
			lifo.blockFree( allocs[i].p, allocs[i].s );
			allocs.removeAt(i);
		}
		HK_ASSERT(0x18dba008, lifo.isEmpty() );
	}

	static void testRealloc(hkLifoAllocator& lifo)
	{
		int n = 16;
		int newN = 4096;
		const int size = 16;

		int nbytes = n * size;
		int nbytes2 = newN * size;

		void* p = lifo.bufAlloc(nbytes);
		void* newP = lifo.bufRealloc(p, nbytes, nbytes2);

		HK_TEST(p != newP);

		lifo.bufFree(newP, nbytes2);
	}

	static void testSlabSize(hkLifoAllocator& lifo)
	{
		const int size = 16384;
		void* p = lifo.blockAlloc(size);
		lifo.blockFree(p, size);
	}

	static void failTest(const char*s, void* p)
	{
		HK_TEST2(false, s);
	}
}

int lifoallocator_main()
{
	hkLeakDetectAllocator leak;
	leak.init(&hkMemoryRouter::getInstance().heap(), &hkMemoryRouter::getInstance().heap(), failTest, HK_NULL);

	hkLifoAllocator lifo;
	for (int i = 0; i < 5; i++) // repeat a few times to test that the lifo object is reusable
	{
		lifo.init(&leak, &leak, &leak);
		hkPseudoRandomGenerator prng(0);

		// regular lifo pattern
		for( int iter = 0; iter < 100; ++iter )
		{
			pureLifoTest(lifo, prng, 10, 64);
		}

		// larger test to allocate & free slabs
		for( int iter = 0; iter < 100; ++iter )
		{
			pureLifoTest(lifo, prng, 100, 1024);
		}

		// heavy duty random patterns
		for( int iter = 0; iter < 50; ++iter )
		{
			for( int n = 1; n < 50; ++n )
			{
				nonLifoTest(lifo, prng, n, 1024);
			}
		}

		// test realloc > slab size
		testRealloc(lifo);

		// test freeing NULL
		lifo.blockFree(HK_NULL, 20000);

		// test size == slabSize
		testSlabSize(lifo);

		lifo.quit();
	}
	leak.quit();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(lifoallocator_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

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
