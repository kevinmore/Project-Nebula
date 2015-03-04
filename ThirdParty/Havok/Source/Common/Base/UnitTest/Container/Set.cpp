/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/Set/hkSet.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

template<class TypeOfSetElement>
static void test_set()
{
	typedef hkSet<TypeOfSetElement> TestSet;

	hkPseudoRandomGenerator randgen(0);
	int numKeys = randgen.getRandChar(127);


	{
		TestSet set;
		set.reserve( numKeys );
		hkArray<TypeOfSetElement> testArray;

		for(int i = 0; i < numKeys; ++i)
		{
			TypeOfSetElement elem = (TypeOfSetElement) randgen.getRand32();
			hkBool32 inserted = set.insert( elem );
			HK_TEST( inserted );
			inserted = set.insert( elem );
			HK_TEST( !inserted );
			hkResult res = HK_FAILURE;
			inserted = set.tryInsert( elem, res );
			HK_TEST( !inserted );
			HK_TEST( res == HK_SUCCESS );
			testArray.pushBack( elem );
		}


		for(int i = 0; i < numKeys; ++i)
		{
			HK_TEST(  set.contains( testArray[i])  );
			typename TestSet::Iterator iter = set.findElement( testArray[i] );
			HK_TEST( set.getElement(iter) == testArray[i] );
		}

		HK_TEST( set.getSize() == testArray.getSize() );

		hkArray<TypeOfSetElement> testArray2;
		testArray2 = testArray;

		typename TestSet::Iterator iter = set.getIterator();
		for (; set.isValid(iter); iter = set.getNext(iter) )
		{
			TypeOfSetElement elem = set.getElement(iter);

			int idx = testArray2.indexOf(elem);
			HK_TEST(idx >= 0);
			testArray2.removeAt(idx);
		}
		HK_TEST( testArray2.isEmpty() );


		for (int i=0; i<numKeys; i++)
		{
			TypeOfSetElement elem = testArray[i];

			HK_TEST( set.contains(elem) );
			hkResult res = set.remove( elem );
			HK_TEST( !set.contains(elem) );
			HK_TEST( res == HK_SUCCESS );
			res = set.remove( elem );
			HK_TEST( res == HK_FAILURE );
		}
	}

}

static void test_shouldResize()
{
	// Check that we can add elements to an hkSet without triggering a resize
	for (int size = 4; size < 256; size++)
	{
		const int stackSize = size;

		int setSize = hkSet<hkUint32>::getSizeInBytesFor(stackSize);
		hkLocalArray<char> setStorage(setSize);

		hkSet<hkUint32> set( setStorage.begin(), setSize );
		const int initialCapacity = set.getCapacity();
		HK_TEST(set.getCapacity() >= stackSize);
		HK_TEST(set.getSize() == 0);

		while( !set.shouldResize() )
		{
			hkUint32 val = set.getSize(); // doesn't matter what we add, as long as it's distinct
			set.insert( val );
			
			HK_TEST(initialCapacity == set.getCapacity());
		}

		// Make sure we actually added something
		HK_TEST( set.getSize() > 0 );

	}
}

static void test_reduceCapacity()
{
	hkSet<int> set;
	for (int i=0; i<1000; i++)
	{
		set.insert(i);
	}
	int oldCap = set.getCapacity();
	for (int i=10; i<1000; i++)
	{
		set.remove(i);
	}
	set.optimizeCapacity();
	int newCap = set.getCapacity();

	HK_TEST( newCap < oldCap );

	hkArray<int> elems;
	hkSet<int>::Iterator iter;
	for (iter = set.getIterator(); set.isValid(iter); iter = set.getNext(iter) )
	{
		elems.pushBack( set.getElement(iter) );
	}

	hkSort( elems.begin(), elems.getSize() );
	for (int i=0; i<10; i++)
	{
		HK_TEST( i == elems[i] );
	}

	hkSet<int> newSet;
	for (int i=0; i<10; i++)
	{
		newSet.insert(i);
	}

	HK_TEST( newSet.getCapacity() == newCap );

}

int set_main()
{
 	test_set<int>();
 	test_set<hkUint32>();
 	test_set<hkUint64>();
	test_set<hkUlong>();

	test_shouldResize();
	test_reduceCapacity();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(set_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
