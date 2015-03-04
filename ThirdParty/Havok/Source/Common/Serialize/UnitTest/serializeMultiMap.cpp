/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkSerializeMultiMap.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

template<typename T>
static T randChoice( hkPseudoRandomGenerator& gen, hkArray<T>& arr )
{
	int i = gen.getRand32() % arr.getSize();
	return arr[i];
}
template<typename T>
static T popRandChoice( hkPseudoRandomGenerator& gen, hkArray<T>& arr )
{
	int i = gen.getRand32() % arr.getSize();
	T r = arr[i];
	arr.removeAt(i);
	return r;
}

static int SerializeMultiMap()
{
	hkPseudoRandomGenerator gen(1234);

	int ITERATIONS = 1000;
	for( int iteration = 0; iteration < ITERATIONS; ++iteration )
	{
		hkSerializeMultiMap<unsigned, unsigned> map;
		hkArray<int> keys;

		for( int round = 0; round < 50; ++round )
		{
			enum
			{
				NEW_KEY = 1,
				DEL_KEY = 2,
				NEW_VAL = 4,
				DEL_VAL = 6
			};
			int whatToDo = gen.getRandChar( DEL_VAL );
			if( whatToDo < NEW_KEY || keys.getSize() == 0 ) // new key
			{
				int k;
				do
				{
					k = gen.getRand32();
				} while( keys.indexOf(k) != -1 );
				keys.pushBack( k );

				HK_TEST( map.insert(k, 0) ); // make sure the key exists
				map.removeByValue(k, 0);

				unsigned loops = gen.getRandChar( 5 );
				for( unsigned i = 0; i < loops; ++i ) // may loop 0 times
				{
					HK_TEST( !map.insert(k, gen.getRand32()) );
				}
			}
			else if( whatToDo < DEL_KEY )  // delete key
			{
				map.removeKey( popRandChoice(gen, keys) );
			}
			else if( whatToDo < NEW_VAL ) // val insertion
			{
				HK_TEST( !map.insert( randChoice(gen, keys), gen.getRand32() ) );
			}
			else // val deletion by index
			{
				hkArray<unsigned> indices;
				unsigned key = randChoice(gen, keys);
				for( int i = map.getFirstIndex(key); i != -1; i = map.getNextIndex(i) )
				{
					indices.pushBack(i);
				}
				if( indices.getSize() )
				{
					int next = map.removeByIndex( key, randChoice(gen, indices) );
					// make sure the "next" index is actually reachable
					if( next != -1 )
					{
						hkBool32 foundNext = false;
						for( int i = map.getFirstIndex(key); i != -1; i = map.getNextIndex(i) )
						{
							if( i == next )
							{
								foundNext = true;
							}
						}
						HK_TEST( foundNext );
					}
				}
			}
		}
	}
	
	hkSerializeMultiMap<void*, hkUlong> map;
	HK_TEST(map.getFirstIndex( &map ) == -1);

	typedef void* Pointer;
	hkUlong table[][5] =
	{
		{ 0x12345678, 11, 31, 21, 0 },
		{ 0x23450000, 12, 12, 42, 0 },
		{ 0x34560001, 3, 23, 0, 0 },
		{ 0x45670111, 4, 24, 24, 0 },
		{ 0x0, 0, 0, 0, 0 }
	};

	for( int i = 0; table[i][0] != 0; ++i )
	{
		for( int j = 1; table[i][j] != 0; ++j )
		{
			map.insert( Pointer(table[i][0]), table[i][j] );
		}
	}

	const int removal[][2] =
	{
		{ 2,2 },
		{ 3,3 },
		{ 3,2 },
		{ 3,1 },
		{ 0,3 },
	};
	int nrepeat = int(sizeof(removal)/(sizeof(int)*2));
	for( int repeat = 0; repeat <= nrepeat; ++repeat )
	{
		for( int i = 0; table[i][0] != 0; ++i )
		{
			int index = map.getFirstIndex(Pointer(table[i][0]));
			HK_TEST( index != -1 || table[i][1] == 0 );
			for( int j = 1; j < 5; ++j )
			{
				if( table[i][j] == 0 ) continue;
				HK_TEST( index != -1);
				hkUlong val = map.getValue(index);
				int found = 0;
				for( int k = 1; table[i][k] && found==0; ++k )
				{
					found |= ( val == table[i][k] );
				}
				HK_TEST( found != 0 );
				index = map.getNextIndex(index);
			}
			HK_TEST( index == -1);
		}

		if( repeat < nrepeat )
		{
			int row = removal[repeat][0];
			int col = removal[repeat][1];
			hkUlong val = table[row][col];
			map.removeByValue( Pointer(table[row][0]), val );
			table[row][col] = 0;
		}
	}
	for( int i = 0; i < 4; ++i )
	{
		for( unsigned j = 0; j < (gen.getRand32()&7); ++j )
		{
			map.insert( Pointer(table[i][0]), gen.getRand32() );
		}
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(SerializeMultiMap, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
