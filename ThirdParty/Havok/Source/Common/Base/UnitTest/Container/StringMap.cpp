/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/StringMap/hkStorageStringMap.h>

static void stringmap()
{
	hkStringMap<int> string_map;

	hkArray<const char*> name_key(5);
	hkArray<int> age_val(5);
	{
		name_key[0] = "John";
		name_key[1] = "Thomas";
		name_key[2] = "Dave";
		name_key[3] = "Billy";
		name_key[4] = "Jack";

		age_val[0] = 26;
		age_val[1] = 28;
		age_val[2] = 23;
		age_val[3] = 24;
		age_val[4] = 27;

		/// Insert key with associated value.
		for( int i = 0; i < 5; i++)
		{
			string_map.insert(name_key[i],age_val[i]);
		}
	}

	// Testing functionality of findKey
	{
		HK_TEST(string_map.isValid(string_map.findKey("Dave")));
		HK_TEST(!string_map.isValid(string_map.findKey("Gary")));
	}

	// Testing of isValid and hasKey functionality
	{
		for( int i = 0; i < 5; i++)
		{
			HK_TEST(string_map.isValid(string_map.findKey(name_key[i])));
			HK_TEST(string_map.hasKey(name_key[i]));
		}
	}

	// Testing getWithDefault functionality
	{
		for( int i = 0; i < 5; i++)
		{
			int x = 77,y;
			y = string_map.getWithDefault(name_key[i],x);
			HK_TEST(y == age_val[i]);
			y = string_map.getWithDefault("This is not available",x);
			HK_TEST(y == 77);
		}
	}

	// Testing functionality of get function
	{
		for( int i = 0; i < 5; i++)
		{
			int value = 0;
			HK_TEST(string_map.get(name_key[i],&value)==HK_SUCCESS);
			HK_TEST(age_val[i]==value);
		}
	}

	// Testing functionality of getKey
	{
		for(hkStringMap<int>::Iterator itr = string_map.getIterator();
			string_map.isValid(itr);
			itr = string_map.getNext(itr))
		{
			const char *namek = string_map.getKey(itr);
			for( int i = 0; i < 5; ++i)
			{
				if(hkString::strCmp(namek, name_key[i])==0)
				{
					HK_TEST(hkString::strCmp(namek, name_key[i])==0);
					break;
				}
			}
		}
	}

	// Testing remove with iterator functionality
	{
		HK_TEST(string_map.getSize()==5);
		string_map.remove(string_map.findKey(name_key[4]));
		HK_TEST(!string_map.hasKey(name_key[4]));
		HK_TEST(string_map.getSize()==4);
	}

	// Testing remove with given key functionality
	{
		HK_TEST(string_map.getSize()==4);
		HK_TEST(string_map.remove(name_key[3])==HK_SUCCESS);
		HK_TEST(string_map.getSize()==3);
		HK_TEST(!string_map.isValid(string_map.findKey(name_key[3])));
	}

	// Testing getSize and isOk functionality
	{
		HK_TEST(string_map.getSize()==3);
		HK_TEST(string_map.isOk());
	}

	// Testing the functionality of setvalue and getvalue
	{
		for(hkStringMap<int>::Iterator itr = string_map.getIterator();
			string_map.isValid(itr);
			itr = string_map.getNext(itr))
		{
			string_map.setValue(itr,33);
			HK_TEST(string_map.getValue(itr)==33);
		}
	}

	// Testing clear functioanlity
	{
		string_map.clear();
		HK_TEST(string_map.getSize()==0);
	}

	// Testing swap functionality
	{
		hkArray<const char*> key(3);
		hkArray<int> val(3);
		hkStringMap<int> sm1;

		key[0] = "abc";
		key[1] = "acb";
		key[2] = "adc";
		val[0] = 1;
		val[1] = 2;
		val[2] = 3;

		for(int i1 = 0; i1 < 3; i1++)
		{
			sm1.insert(key[i1],val[i1]);
		}

		hkArray<const char*> key2(3);
		hkArray<int> val2(3);
		hkStringMap<int> sm2;

		key2[0] = "wxy";
		key2[1] = "wyx";
		key2[2] = "wzx";
		val2[0] = 4;
		val2[1] = 5;
		val2[2] = 6;
		for(int i2 = 0; i2 < 3; i2++)
		{
			sm2.insert(key2[i2],val2[i2]);
		}

		sm1.swap(sm2);

		// Verifying swapped values of stringmap sm1
		int i = 0;
		for(hkStringMap<int>::Iterator itr = sm1.getIterator();
			sm1.isValid(itr); itr = sm1.getNext(itr))
		{
			hkStringMap<int>::Iterator itr1 = sm1.findKey(key2[i]);
			HK_TEST(sm1.isValid(itr1) );
			HK_TEST(sm1.getValue(itr1) == val2[i]);
			++i;
		}

		// Verifying swapped values of stringmap sm2
		i = 0;
		for(hkStringMap<int>::Iterator itr = sm2.getIterator();
			sm2.isValid(itr); itr = sm2.getNext(itr))
		{
			hkStringMap<int>::Iterator itr2 = sm2.findKey(key[i]);
			HK_TEST(sm2.isValid(itr2) );
			HK_TEST(sm2.getValue(itr2) == val[i]);
			++i;
		}
	}

	// Find Or Insert
	{
		hkStringMap<int> sm;
		HK_TEST(sm.getSize()==0);
		sm.insert("foo", 10);
		HK_TEST(sm.getSize()==1);
		sm.insert("foo", 100);
		HK_TEST(sm.getSize()==1);
		// create key
		{
			hkStringMap<int>::Iterator it = sm.findOrInsertKey("bar", 99);
			HK_TEST( sm.getValue(it) == 99 );
			HK_TEST( sm.getWithDefault("bar",0) == 99 );
			HK_TEST( sm.getSize() == 2 );
		}
		// existing key
		{
			hkStringMap<int>::Iterator it = sm.findOrInsertKey("foo", 50);
			HK_TEST( sm.getValue(it) == 100 );
			HK_TEST( sm.getWithDefault("foo",0) == 100 );
			HK_TEST( sm.getSize() == 2 );
		}
	}
}

void storagestringmap()
{
	hkStorageStringMap<int> map;
	{
		map.insert("hello", 1);
		HK_TEST( map.getWithDefault("hello", -1) == 1);
		map.insert("hello", 2);
		HK_TEST( map.getWithDefault("hello", -1) == 2);
	}
	map.clear();
    {
		HK_TEST( map.getOrInsert("foo", 10) == 10);
		HK_TEST( map.getOrInsert("foo", 10) == 10);
    }
	map.clear();
	{
		map.insert("foo", 100);
		map.insert("xxx", 101);
		map.insert("baz", 102);

		hkStorageStringMap<int> map2;
		map2.insert("foo", 10);
		map2.insert("bar", 11);
		map2.insert("baz", 12);

		map.merge(map2);
		HK_TEST( map.getSize() == 4);
		HK_TEST( map.getWithDefault("foo",0) == 10 );
		HK_TEST( map.getWithDefault("xxx",0) == 101 );
		HK_TEST( map.getWithDefault("baz",0) == 12 );
	}
}

int stringmap_main()
{
	stringmap();
	storagestringmap();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(stringmap_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
