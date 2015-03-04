/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Serialize/ResourceDatabase/hkResourceHandle.h>
#include <Common/Serialize/Util/hkNativePackfileUtils.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>

namespace
{
	enum hkContentsTestType
	{
		HK_CONTENTS_TEST_BINARY,
		HK_CONTENTS_TEST_XML,
		HK_CONTENTS_TEST_TAG
	};

	static hkResult saveReferencedObject(hkContentsTestType testType, hkStreamWriter* writer, void* object, const hkClass& klass)
	{
		hkResult err = HK_FAILURE;
		switch( testType )
		{
			case HK_CONTENTS_TEST_BINARY:
			{
				hkPackfileWriter::Options opt;
				err = hkSerializeUtil::savePackfile(object, klass, writer, opt);
				break;
			}
			case HK_CONTENTS_TEST_XML:
			{
				hkPackfileWriter::Options opt;
				err = hkSerializeUtil::savePackfile(object, klass, writer, opt, HK_NULL);
				break;
			}
			case HK_CONTENTS_TEST_TAG:
			{
				err = hkSerializeUtil::saveTagfile(object, klass, writer);
				break;
			}
		}
		return err;
	}

	static void testContents(hkStreamReader* reader)
	{
		hkRefPtr<hkResource> resource = hkSerializeUtil::load(reader);
		HK_TEST(resource != HK_NULL);
		resource->removeReference(); // hkRefPtr takes responsibility
		hkReferencedObject* refObject = resource->getContents<hkReferencedObject>();
		HK_TEST(refObject != HK_NULL);
		hkReferencedObject* refObjectByBaseName = static_cast<hkReferencedObject*>(resource->getContentsPointer(hkReferencedObjectClass.getName(), hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry()));
		HK_TEST(refObject == refObjectByBaseName);
		void* contentsPtr = resource->getContentsPointer(HK_NULL, hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry());
		HK_TEST(refObject == contentsPtr);
	}
}

static int contentsBinaryTest()
{
	hkArray<char> buffer;
	hkRefPtr<hkMemoryResourceContainer> myRefObject = new hkMemoryResourceContainer();
	HK_ASSERT(0x53b06a31, myRefObject);
	myRefObject->removeReference(); // hkRefPtr takes responsibility
	//
	// binary packfile
	//
	{
		// exact class name
		hkOstream out(buffer);
		hkResult err = saveReferencedObject(HK_CONTENTS_TEST_BINARY, out.getStreamWriter(), myRefObject, hkMemoryResourceContainerClass);
		HK_TEST(err == HK_SUCCESS);
	}
	{
		hkIstream in(buffer.begin(), buffer.getSize());
		testContents(in.getStreamReader());
	}
	{
		const char* name = hkNativePackfileUtils::getContentsClassName(buffer.begin(), buffer.getSize());
		HK_TEST( hkString::strCmp(name, hkMemoryResourceContainer::staticClass().getName()) == 0);
		HK_TEST( hkNativePackfileUtils::getContentsClassName(buffer.begin(), 10) == HK_NULL );
		HK_TEST( hkNativePackfileUtils::getContentsClassName(HK_NULL, 100) == HK_NULL );
		HK_TEST( hkNativePackfileUtils::getContentsClassName(HK_NULL, 0) == HK_NULL );
	}
	buffer.setSize(0);
	{
		// base class name
		hkOstream out(buffer);
		hkResult err = saveReferencedObject(HK_CONTENTS_TEST_BINARY, out.getStreamWriter(), myRefObject, hkReferencedObjectClass);
		HK_TEST(err == HK_SUCCESS);
	}
	{
		hkIstream in(buffer.begin(), buffer.getSize());
		testContents(in.getStreamReader());
	}

	return 0;
}

static int contentsXmlTest()
{
	hkArray<char> buffer;
	hkRefPtr<hkMemoryResourceContainer> myRefObject = new hkMemoryResourceContainer();
	HK_ASSERT(0x53b06a31, myRefObject);
	myRefObject->removeReference(); // hkRefPtr takes responsibility
	//
	// XML packfile
	//
	{
		// exact class name
		hkOstream out(buffer);
		hkResult err = saveReferencedObject(HK_CONTENTS_TEST_XML, out.getStreamWriter(), myRefObject, hkMemoryResourceContainerClass);
		HK_TEST(err == HK_SUCCESS);
	}
	{
		hkIstream in(buffer.begin(), buffer.getSize());
		testContents(in.getStreamReader());
	}
	buffer.setSize(0);
	{
		// base class name
		hkOstream out(buffer);
		hkResult err = saveReferencedObject(HK_CONTENTS_TEST_XML, out.getStreamWriter(), myRefObject, hkReferencedObjectClass);
		HK_TEST(err == HK_SUCCESS);
	}
	{
		hkIstream in(buffer.begin(), buffer.getSize());
		testContents(in.getStreamReader());
	}

	return 0;
}

static int contentsTagTest()
{
	hkArray<char> buffer;
	hkRefPtr<hkMemoryResourceContainer> myRefObject = new hkMemoryResourceContainer();
	HK_ASSERT(0x53b06a31, myRefObject);
	myRefObject->removeReference(); // hkRefPtr takes responsibility
	//
	// Binary tagfile
	//
	{
		// exact class name
		hkOstream out(buffer);
		hkResult err = saveReferencedObject(HK_CONTENTS_TEST_TAG, out.getStreamWriter(), myRefObject, hkMemoryResourceContainerClass);
		HK_TEST(err == HK_SUCCESS);
	}
	{
		hkIstream in(buffer.begin(), buffer.getSize());
		testContents(in.getStreamReader());
	}
	buffer.setSize(0);
	{
		// base class name
		hkOstream out(buffer);
		hkResult err = saveReferencedObject(HK_CONTENTS_TEST_TAG, out.getStreamWriter(), myRefObject, hkReferencedObjectClass);
		HK_TEST(err == HK_SUCCESS);
	}
	{
		hkIstream in(buffer.begin(), buffer.getSize());
		testContents(in.getStreamReader());
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( contentsBinaryTest, "Fast", "Common/Test/UnitTest/Serialize/", "contentsBinaryTest" );
//HK_TEST_REGISTER( contentsXmlTest, "Fast", "Common/Test/UnitTest/Serialize/", "contentsXmlTest" );
HK_TEST_REGISTER( contentsTagTest, "Fast", "Common/Test/UnitTest/Serialize/", "contentsTagTest" );

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
