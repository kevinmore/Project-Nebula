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

template<class TypeOfMapElement>
static void test_pointermap()
{
	typedef hkPointerMap<TypeOfMapElement, TypeOfMapElement> TestMap;

	hkPseudoRandomGenerator randgen(0);
	TypeOfMapElement numKeys = randgen.getRandChar(127);

	//Testing getKey(),getValue()and hasKey() functionality
	{
		TestMap pmap;

		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			pmap.insert(i, i+100);
		}

		// Testing isOk() functionality
		{
			HK_TEST( pmap.isOk() );
		}

		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			typename TestMap::Iterator it = pmap.findKey(i);
			HK_TEST( pmap.getKey(it) == i );
			HK_TEST( pmap.getValue(it) == (i+100) );
			HK_TEST( pmap.hasKey(i) );
		}

		// Testing get() functionality
		{
			for(TypeOfMapElement i = 0; i < numKeys; ++i )
			{
				TypeOfMapElement out = 0;
				HK_TEST( pmap.get(i, &out) == HK_SUCCESS );
				HK_TEST( out==(i+100) );
			}

			for(TypeOfMapElement j = numKeys; j < numKeys+20; ++j )
			{
				TypeOfMapElement out = 0;
				HK_TEST( pmap.get(j, &out) == HK_FAILURE );
				HK_TEST( out==0 );
			}
		}
	}

	// Testing getWithDefault(), getCapacity(), getSizeInBytesFor
	{
		TestMap pmap;

		const int pairSize = sizeof(typename TestMap::Storage)*2;

		// Testing getSizeInBytesFor() && getCapacity() functionality
		for( int k = 1; k < (int)numKeys; ++k )
		{
			TestMap pmapK;
			int calcSizeInBytes = TestMap::getSizeInBytesFor(k);
			for(TypeOfMapElement i = 0; i < (TypeOfMapElement)k; ++i)
			{
				TypeOfMapElement value = i + 50;
				pmapK.insert(i, value);
			}
			int sizeInBytes = pmapK.getCapacity()*pairSize;
			HK_TEST( sizeInBytes == calcSizeInBytes );
			if( sizeInBytes != calcSizeInBytes )
			{
				HK_REPORT("The real size in bytes does not match the hkPointerMap::getSizeInBytesFor( " << k << " ).");
			}
			if( k > 1 )
			{
				// use external buffer
				{
					hkArray<char> mapBuffer(sizeInBytes);
					TestMap extBufMap(mapBuffer.begin(), sizeInBytes);
					sizeInBytes = extBufMap.getCapacity()*pairSize;
					HK_TEST( sizeInBytes == calcSizeInBytes );
					if( sizeInBytes != calcSizeInBytes )
					{
						HK_REPORT("The real size in bytes of map with external buffer does not match the hkPointerMap::getSizeInBytesFor( " << k << " ).");
					}
				}
				// reserved capacity
				{
					hkMap<TypeOfMapElement,TypeOfMapElement> reservedSizeMap(k);
					sizeInBytes = reservedSizeMap.getCapacity()*pairSize;
					HK_TEST( sizeInBytes == calcSizeInBytes );
					if( sizeInBytes != calcSizeInBytes )
					{
						HK_REPORT("The real size in bytes of map with reserved number of elements does not match the hkPointerMap::getSizeInBytesFor( " << k << " ).");
					}
				}
			}
		}
		// init to test
		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			TypeOfMapElement value = i + 50;
			pmap.insert(i, value);
		}

		// Testing getWithDefault() functionality
		{
			for(TypeOfMapElement i = 0; i < numKeys; ++i)
			{
				TypeOfMapElement def = 0;
				TypeOfMapElement value = i + 50;
				HK_TEST( pmap.getWithDefault(i,def) == value );
				HK_TEST( def == 0 );
			}
			for(TypeOfMapElement i = numKeys; i < numKeys+5; ++i)
			{
				TypeOfMapElement def = 0;
				HK_TEST( pmap.getWithDefault(i,def) == def );
			}
		}
		// Testing setValue() functionality
		{
			for( typename TestMap::Iterator it1 = pmap.getIterator();
				pmap.isValid(it1);
				it1 = pmap.getNext(it1) )
			{
				pmap.setValue(it1, 100);
			}

			for( typename TestMap::Iterator it1 = pmap.getIterator();
				pmap.isValid(it1);
				it1 = pmap.getNext(it1) )
			{
				HK_TEST( pmap.getValue(it1) == 100 );
			}
			HK_TEST( pmap.isOk() );
		}
		// Testing wasReallocateed() functionality
		{
			HK_TEST( pmap.wasReallocated() );
			pmap.clear();
			pmap.reserve(20);
			HK_TEST( pmap.wasReallocated() );
			pmap.insert(1, 1);
			HK_TEST( pmap.wasReallocated() );
		}
	}
	//Testing swap()functionality
	{
		TestMap pmap1;
		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			pmap1.insert(i, i+100);
		}

		TestMap pmap2;
		TypeOfMapElement limit = numKeys+21;

		for(TypeOfMapElement i = numKeys; i < limit; ++i)
		{
			pmap2.insert(i, i+99);
		}

		pmap1.swap(pmap2);

		// Verifying pmap1 with pmap2 values
		{
			HK_TEST( pmap1.getSize() == (int)(limit-numKeys) );
			for( typename TestMap::Iterator it1 = pmap1.getIterator();
				pmap1.isValid(it1);
				it1 = pmap1.getNext(it1) )
			{
				TypeOfMapElement key = pmap1.getKey(it1);
				HK_TEST( numKeys <= key && key < limit  );
				HK_TEST( pmap1.getValue(it1) == (key+99) );
			}
		}

		// Verifying pmap2 with pmap1 values
		{
			HK_TEST( pmap2.getSize() == (int)numKeys );
			for( typename TestMap::Iterator it2 = pmap2.getIterator();
				pmap2.isValid(it2);
				it2 = pmap2.getNext(it2) )
			{
				TypeOfMapElement key = pmap2.getKey(it2);
				HK_TEST( (0 == key || 0 < key) && key < numKeys );
				HK_TEST( pmap2.getValue(it2) == (key+100) );
			}
		}
	}

	// Testing Remove() via iterator.
	{
		TestMap pmap1;
		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			pmap1.insert(i, i+100);
		}

		TypeOfMapElement key = 0;
		typename TestMap::Iterator it1 = pmap1.getIterator();
		{
			int num = pmap1.getSize();
			key = pmap1.getKey(it1);
			pmap1.remove(it1);
			HK_TEST( pmap1.getSize() == num-1 );
		}

		TypeOfMapElement out = 0;
		HK_TEST( pmap1.get(key, &out) == HK_FAILURE );
		HK_TEST(out == 0);
	}

	// Testing findOrInsertKey
	{
		TestMap pmap;
		typename TestMap::Iterator it;

		it = pmap.findOrInsertKey(10,101);
		HK_TEST( pmap.getValue(it) == 101 );
		HK_TEST(pmap.getSize() == 1);
		HK_TEST( !pmap.insert(10,102) );
		HK_TEST(pmap.getSize() == 1);

		it = pmap.findOrInsertKey(10,103);
		HK_TEST( pmap.getValue(it) == 102 );
		HK_TEST(pmap.getSize() == 1);
	}

	// Testing Remove() via Key.
	{
		TestMap pmap1;
		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			pmap1.insert(i, i+100);
		}

		int num = pmap1.getSize();
		HK_TEST( pmap1.remove(7) == HK_SUCCESS );
		HK_TEST( pmap1.getSize() == num-1 );

		TypeOfMapElement out = 0;
		HK_TEST( pmap1.get(7, &out) == HK_FAILURE );
		HK_TEST(out == 0);
	}

	// Testing of clear() functionality
	{
		TestMap pmap1;
		for(TypeOfMapElement i = 0; i < numKeys; ++i)
		{
			pmap1.insert(i, i+100);
		}

		pmap1.clear();
		HK_TEST( pmap1.getSize() == 0 );
	}
}

int pointermap_main()
{
	{
		hkPointerMap<hkUint64, int> pmap64;
		hkPointerMap<hkUint32, int> pmap32;

		// If you get a warning about the ULL suffix, just
		// add an exception to the ifdef
#		if 1
#			define UINT64_CONST(A) A##ULL
#		else
#			define UINT64_CONST(A) A
#		endif

		pmap32.insert( /* 0xff00000001 truncates to */ 1, 10 );
		pmap32.insert( /* 0x0000000001 truncates to */ 1, 10 );
		pmap64.insert( UINT64_CONST(0xff00000001), 10 );
		pmap64.insert( UINT64_CONST(0x0000000001), 10 );
		HK_TEST( pmap32.getSize() == 1 );
		HK_TEST( pmap64.getSize() == 2 );
	}

	test_pointermap<hkUlong>();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(pointermap_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
