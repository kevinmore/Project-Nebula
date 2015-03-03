/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>
#include <Common/Base/Container/RelArray/hkRelArrayUtil.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Serialize/Util/hkObjectInspector.h>
#include <Common/Serialize/UnitTest/SerializeRelArray/serializeRelArray.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Serialize/Util/hkSerializeDeprecated.h>


extern const hkClass SerializeRelArrayTestObjectClass;
extern const hkTypeInfo SerializeRelArrayTestObjectTypeInfo;

const int ARRAY_SIZE = 5;

static void initializeObject(SerializeRelArrayTestObject& object)
{
	// set some starting values
	object.m_array1[0] = 1;
	object.m_array1[1] = 1;
	object.m_array1[2] = 2;
	object.m_array1[3] = 3;
	object.m_array1[4] = 5;
	object.m_array2[0] = 'a';
	object.m_array2[1] = 'b';
	object.m_array2[2] = 'c';
	object.m_array2[3] = 'd';
	object.m_array2[4] = 'e';
}

static void checkObject(const SerializeRelArrayTestObject& object)
{
	// check alignment (after deserialization hkRelArrays should always be 16 bytes aligned).
	HK_TEST(((hkUlong)object.m_array1.begin() & (hkRelArrayUtil::RELARRAY_ALIGNMENT-1)) == 0);
	HK_TEST(((hkUlong)object.m_array2.begin() & (hkRelArrayUtil::RELARRAY_ALIGNMENT-1)) == 0);

	// check the starting values
	HK_TEST(object.m_array1[0] == 1);
	HK_TEST(object.m_array1[1] == 1);
	HK_TEST(object.m_array1[2] == 2);
	HK_TEST(object.m_array1[3] == 3);
	HK_TEST(object.m_array1[4] == 5);
	HK_TEST(object.m_array2[0] == 'a');
	HK_TEST(object.m_array2[1] == 'b');
	HK_TEST(object.m_array2[2] == 'c');
	HK_TEST(object.m_array2[3] == 'd');
	HK_TEST(object.m_array2[4] == 'e');
}

static void saveAndLoadRelArrayTest(const void* obj)
{
	// Buffer used to store and load the object
	hkArray<char> buffer;

	// 1: save and load a binary tagfile
	{
		hkArrayStreamWriter writer(&buffer, hkArrayStreamWriter::ARRAY_BORROW);
		hkSerializeUtil::SaveOptions options;
		options.useBinary(true);
		hkResult res = hkSerializeUtil::saveTagfile( obj, SerializeRelArrayTestObjectClass, &writer, HK_NULL, options);
		if(HK_TEST(res == HK_SUCCESS)) // save failed
		{
			hkMemoryStreamReader reader(buffer.begin(), buffer.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
			hkResource* resource = hkSerializeUtil::load(&reader);
			if(HK_TEST(resource != HK_NULL)) // load failed
			{
				SerializeRelArrayTestObject* loadedObj = resource->getContents<SerializeRelArrayTestObject>();
				checkObject(*loadedObj);
				resource->removeReference();
			}
		}
	}

	buffer.clear();

	// save and load an xml tagfile
	{
		hkArrayStreamWriter writer(&buffer, hkArrayStreamWriter::ARRAY_BORROW);
		hkSerializeUtil::SaveOptions options;
		options.useBinary(false);
		hkResult res = hkSerializeUtil::saveTagfile( obj, SerializeRelArrayTestObjectClass, &writer, HK_NULL, options);
		if(HK_TEST(res == HK_SUCCESS)) // save failed
		{
			hkMemoryStreamReader reader(buffer.begin(), buffer.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
			hkResource* resource = hkSerializeUtil::load(&reader);
			if(HK_TEST(resource != HK_NULL)) // load failed
			{
				SerializeRelArrayTestObject* loadedObj = resource->getContents<SerializeRelArrayTestObject>();
				checkObject(*loadedObj);
				resource->removeReference();
			}
		}
	}

	buffer.clear();

	// save and load a binary packfile
	{
		hkArrayStreamWriter writer(&buffer, hkArrayStreamWriter::ARRAY_BORROW);
		hkSerializeUtil::SaveOptions options;
		options.useBinary(true);
		hkResult res = hkSerializeUtil::savePackfile( obj, SerializeRelArrayTestObjectClass, &writer, hkPackfileWriter::Options(), HK_NULL, options);
		if(HK_TEST(res == HK_SUCCESS)) // save failed
		{
			hkMemoryStreamReader reader(buffer.begin(), buffer.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
			hkResource* resource = hkSerializeUtil::load(&reader);
			if(HK_TEST(resource != HK_NULL)) // load failed
			{
				SerializeRelArrayTestObject* loadedObj = resource->getContents<SerializeRelArrayTestObject>();
				checkObject(*loadedObj);
				resource->removeReference();
			}
		}
	}

	buffer.clear();

	// Save and load an xml packfile
	// Don't bother trying if XML packfile support is not enabled.
	if ( hkSerializeDeprecated::getInstance().isEnabled() )
	{
		hkArrayStreamWriter writer(&buffer, hkArrayStreamWriter::ARRAY_BORROW);
		hkSerializeUtil::SaveOptions options;
		options.useBinary(false);
		hkResult res = hkSerializeUtil::savePackfile( obj, SerializeRelArrayTestObjectClass, &writer, hkPackfileWriter::Options(), HK_NULL, options);
		if(HK_TEST(res == HK_SUCCESS)) // save failed
		{
			hkMemoryStreamReader reader(buffer.begin(), buffer.getSize(), hkMemoryStreamReader::MEMORY_INPLACE);
			hkResource* resource = hkSerializeUtil::load(&reader);
			if(HK_TEST(resource != HK_NULL)) // load failed
			{
				SerializeRelArrayTestObject* loadedObj = resource->getContents<SerializeRelArrayTestObject>();
				checkObject(*loadedObj);
				resource->removeReference();
			}
		}
	}

}

static void inspectRelArrayTest(const void* obj)
{
	hkArray<hkObjectInspector::Pointer>::Temp pointers;
	hkObjectInspector::getPointers(obj, SerializeRelArrayTestObjectClass, pointers);
	HK_TEST(pointers.getSize() == 0); // no pointers contained
}

static int serializeRelArrayTest()
{
	hkBuiltinTypeRegistry::getInstance().addType(
		&SerializeRelArrayTestObjectTypeInfo, 
		&SerializeRelArrayTestObjectClass);

	const int blockSize = sizeof(SerializeRelArrayTestObject) + sizeof(int)*ARRAY_SIZE + sizeof(char)*ARRAY_SIZE;

	// Object being saved and loaded
	// The allocated block will also contain the body of the rel array
	void* block = hkMemoryRouter::getInstance().heap().blockAlloc(blockSize);
	void* array1BufAddr = hkAddByteOffset(block, sizeof(SerializeRelArrayTestObject));
	void* array2BufAddr = hkAddByteOffset(array1BufAddr, sizeof(int)*ARRAY_SIZE);
	SerializeRelArrayTestObject* obj = new(block) SerializeRelArrayTestObject;
	int offset1 = hkGetByteOffsetInt(&obj->m_array1, array1BufAddr);
	HK_ASSERT(0x3e867ee4, offset1 > 0);
	obj->m_array1._setOffset( static_cast<hkUint16>(offset1) );
	obj->m_array1._setSize(ARRAY_SIZE);
	int offset2 = hkGetByteOffsetInt(&obj->m_array2, array2BufAddr);
	HK_ASSERT(0x3e867ee4, offset2 > 0);
	obj->m_array2._setOffset( static_cast<hkUint16>(offset2) );
	obj->m_array2._setSize(ARRAY_SIZE);
	
	initializeObject(*obj);

	// Perform tests
	{
		saveAndLoadRelArrayTest(obj);
		inspectRelArrayTest(obj);
	}

	// Release resources
	hkMemoryRouter::getInstance().heap().blockFree(block, blockSize);
	hkBuiltinTypeRegistry::getInstance().reinitialize();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(serializeRelArrayTest, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__ );

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
