/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

static int stringArrayTest()
{
	const char* testString = "test string";

	HK_COMPILE_TIME_ASSERT(hkSizeOf(hkStringPtr) == hkSizeOf(char*));
	{
		hkStringPtr stringPtr;
		hkUlong& stringValue = *reinterpret_cast<hkUlong*>(&stringPtr);
		HK_TEST(stringValue == 0);

		char* stringOnHeap = hkString::strDup(testString);
		stringValue = hkUlong(stringOnHeap); // as packfile data

		stringPtr = testString; // should make a copy of testString
		HK_TEST(stringValue != hkUlong(stringOnHeap));
		HK_TEST(stringValue != hkUlong(testString));
		HK_TEST((stringValue & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG); // copy of testString

		hkDeallocate<char>(stringOnHeap); // free as packfile data
	}
	
	{
#		define STRING_ARRAY_SIZE 6
		// initialize
		hkArray<hkStringPtr> stringPtrArray;
		int i;
		for( i = 0; i < STRING_ARRAY_SIZE; ++i )
		{
			stringPtrArray.expandOne() = testString;
			const hkUlong& stringValue = *reinterpret_cast<const hkUlong*>(&stringPtrArray[i]);
			HK_TEST((stringValue & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG);
		}
		HK_TEST(stringPtrArray.getSize() == STRING_ARRAY_SIZE);

		// change capacity
		const char* items[STRING_ARRAY_SIZE];
		for( i = 0; i < stringPtrArray.getSize(); ++i )
		{
			items[i] = stringPtrArray[i].cString();
		}
		stringPtrArray.reserve(stringPtrArray.getCapacity() + 1);
		for( i = 0; i < stringPtrArray.getSize(); ++i )
		{
			HK_TEST(items[i] == stringPtrArray[i]);
		}

		// resize
		// make it bigger
#		define STRING_ARRAY_NEW_SIZE 10
		int oldSize = stringPtrArray.getSize();
		HK_ASSERT(0x2f657860, oldSize < STRING_ARRAY_NEW_SIZE);
		stringPtrArray.setSize(STRING_ARRAY_NEW_SIZE);
		HK_TEST(stringPtrArray.getSize() == STRING_ARRAY_NEW_SIZE);
		for( i = oldSize; i < stringPtrArray.getSize(); ++i )
		{
			const hkUlong& stringValue = *reinterpret_cast<const hkUlong*>(&stringPtrArray[i]);
			HK_TEST(stringValue == 0);
		}

		// make it smaller
		stringPtrArray[STRING_ARRAY_NEW_SIZE-1] = testString; // test last item
		const hkUlong& lastStringItemValue = *reinterpret_cast<const hkUlong*>(&stringPtrArray[STRING_ARRAY_NEW_SIZE-1]);
		HK_TEST((lastStringItemValue & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG);
		stringPtrArray.setSize(STRING_ARRAY_SIZE); // make it smaller
		HK_TEST(stringPtrArray.getSize() == STRING_ARRAY_SIZE);
		// test with items[] previously setup
		for( i = 0; i < stringPtrArray.getSize(); ++i )
		{
			HK_TEST(items[i] == stringPtrArray[i]);
		}

#		define STRING_ARRAY_ITEM_INDEX 2
		// remove at and copy
		for( i = 0; i < stringPtrArray.getSize(); ++i )
		{
			items[i] = stringPtrArray[i];
		}
		stringPtrArray.removeAtAndCopy(STRING_ARRAY_ITEM_INDEX);
		HK_TEST(stringPtrArray.getSize() == (STRING_ARRAY_SIZE-1));
		for( i = 0; i < STRING_ARRAY_ITEM_INDEX; ++i )
		{
			HK_TEST(items[i] == stringPtrArray[i]);
		}
		for( i = STRING_ARRAY_ITEM_INDEX; i < stringPtrArray.getSize(); ++i )
		{
			HK_TEST(items[i+1] == stringPtrArray[i]);
		}

		// remove at
		const char* lastStringItem = stringPtrArray[stringPtrArray.getSize()-1];
		for( i = 0; i < stringPtrArray.getSize()-1; ++i )
		{
			items[i] = stringPtrArray[i];
		}
		stringPtrArray.removeAt(STRING_ARRAY_ITEM_INDEX);
		HK_TEST(stringPtrArray.getSize() == (STRING_ARRAY_SIZE-2));
		for( i = 0; i < STRING_ARRAY_ITEM_INDEX; ++i )
		{
			HK_TEST(items[i] == stringPtrArray[i]);
		}
		// test removed item index value against last item
		HK_TEST(lastStringItem == stringPtrArray[STRING_ARRAY_ITEM_INDEX]);
		for( i = STRING_ARRAY_ITEM_INDEX+1; i < stringPtrArray.getSize(); ++i )
		{
			HK_TEST(items[i] == stringPtrArray[i]);
		}

		// copy
		hkArray<hkStringPtr> stringPtrArrayCopy;
		stringPtrArrayCopy = stringPtrArray;
		HK_TEST(stringPtrArrayCopy.getSize() == stringPtrArray.getSize());
		for( i = 0; i < stringPtrArrayCopy.getSize(); ++i )
		{
			const hkUlong& stringValue = *reinterpret_cast<const hkUlong*>(&stringPtrArray[i]);
			const hkUlong& stringValueCopy = *reinterpret_cast<const hkUlong*>(&stringPtrArrayCopy[i]);
			HK_TEST(stringValue != stringValueCopy);
			HK_TEST((stringValueCopy & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG);
			HK_TEST((stringValueCopy & hkStringPtr::OWNED_FLAG) == hkStringPtr::OWNED_FLAG);
		}
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( stringArrayTest, "Fast", "Common/Test/UnitTest/Serialize/", "stringArrayTest" );

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
