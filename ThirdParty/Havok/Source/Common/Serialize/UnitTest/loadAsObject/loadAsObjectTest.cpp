/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Reflection/Registry/hkDynamicClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Serialize/UnitTest/loadAsObject/loadAsObjectTest.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Serialize/Version/hkVersionUtil.h>

#ifndef HK_COMPILER_ARMCC // does not like the templates below at all..

static const hkClass* loadAsVariantTestClasses[] =
{
	&TestUnsupportedCstringClass,
	&TestUnsupportedCstringArrayClass,
	&TestUnsupportedVariantClass,
	&TestUnsupportedVariantArrayClass,
	&TestUnsupportedSimpleArrayClass,
	&TestUnsupportedHomogeneousArrayClass,
	&TestMemberOfUnsupportedClassClass,
	&TestSupportedVirtualClass,
	&TestSupportedVirtualSimpleDataClass,
	&TestSupportedNonVirtualClass,
	&TestSimpleStructClass,
	&hkReferencedObjectClass,
	&hkBaseObjectClass
};

extern const hkTypeInfo hkReferencedObjectTypeInfo;
extern const hkTypeInfo hkBaseObjectTypeInfo;
static const hkTypeInfo* loadAsVariantTestTypeInfos[] =
{
	&TestUnsupportedCstringTypeInfo,
	&TestUnsupportedCstringArrayTypeInfo,
	&TestUnsupportedVariantTypeInfo,
	&TestUnsupportedVariantArrayTypeInfo,
	&TestUnsupportedSimpleArrayTypeInfo,
	&TestUnsupportedHomogeneousArrayTypeInfo,
	&TestMemberOfUnsupportedClassTypeInfo,
	&TestSupportedVirtualTypeInfo,
	&TestSupportedVirtualSimpleDataTypeInfo,
	&TestSupportedNonVirtualTypeInfo,
	&TestSimpleStructTypeInfo,
	&hkReferencedObjectTypeInfo,
	&hkBaseObjectTypeInfo
};

HK_COMPILE_TIME_ASSERT(HK_COUNT_OF(loadAsVariantTestClasses) == HK_COUNT_OF(loadAsVariantTestTypeInfos));

static inline void registerClassesForTest(hkDynamicClassNameRegistry& classReg, hkTypeInfoRegistry& typeReg)
{
	for( unsigned int i = 0; i < HK_COUNT_OF(loadAsVariantTestClasses); ++i )
	{
		classReg.registerClass(loadAsVariantTestClasses[i]);
		typeReg.registerTypeInfo(loadAsVariantTestTypeInfos[i]);
		if( loadAsVariantTestClasses[i]->hasVtable() )
		{
			hkVtableClassRegistry::getInstance().registerVtable( loadAsVariantTestTypeInfos[i]->getVtable(), loadAsVariantTestClasses[i] );
		}
		//hkBuiltinTypeRegistry::getInstance().addType(loadAsVariantTestTypeInfos[i], loadAsVariantTestClasses[i]);
	}
}

static inline hkObjectResource* loadFromBuffer(hkArray<char>& buffer, const hkClassNameRegistry& classReg, const hkTypeInfoRegistry& typeReg)
{
	hkSerializeUtil::ErrorDetails err;
	hkIstream in(buffer.begin(), buffer.getSize());
	hkSerializeUtil::LoadOptions options;
	options.useClassNameRegistry(&classReg).useTypeInfoRegistry(&typeReg);
	return hkSerializeUtil::loadOnHeap(in.getStreamReader(), &err, options);
}

template<class T>
static inline hkObjectResource* testPackfile(const hkClass& klass, const hkClassNameRegistry& classReg, const hkTypeInfoRegistry& typeReg)
{
	hkArray<char> buffer;

	T testObject;

	// save packfile
	{
		hkOstream out(buffer);
		hkPackfileWriter::Options options;
		hkSerializeUtil::savePackfile(&testObject, klass, out.getStreamWriter(), options);
	}
	// load packfile
	return loadFromBuffer(buffer, classReg, typeReg);
}

template<class T>
static inline hkObjectResource* testTagfile(const hkClass& klass, const hkClassNameRegistry& classReg, const hkTypeInfoRegistry& typeReg)
{
	hkArray<char> buffer;

	T testObject;

	// save tagfile
	{
		hkOstream out(buffer);
		hkSerializeUtil::saveTagfile(&testObject, klass, out.getStreamWriter());
	}
	// load tagfile
	return loadFromBuffer(buffer, classReg, typeReg);
}

template<class T>
static inline void testUnsupportedDataType(const hkClass& klass, const hkClassNameRegistry& classReg, const hkTypeInfoRegistry& typeReg)
{
	// load packfile
	{
		hkObjectResource* loadedObject = testPackfile<T>(klass, classReg, typeReg);
		HK_TEST(loadedObject == HK_NULL);
		if( loadedObject != HK_NULL )
		{
			HK_WARN_ALWAYS(0x6a8db59e, "Test failed. Expect HK_NULL loading " << klass.getName() << " as hkObjectResource from packfile.");
			loadedObject->removeReference();
		}
	}
	// load tagfile
	{
		hkObjectResource* loadedObject = testTagfile<T>(klass, classReg, typeReg);
		HK_TEST(loadedObject == HK_NULL);
		if( loadedObject != HK_NULL )
		{
			HK_WARN_ALWAYS(0x2ab4a56b, "Test failed. Expect HK_NULL loading " << klass.getName() << " as hkObjectResource from tagfile.");
			loadedObject->removeReference();
		}
	}
}

#define TEST_UNSUPPORTED_DATA_TYPE(CPPTYPE) \
	testUnsupportedDataType<CPPTYPE>(CPPTYPE##Class, classReg, typeReg)

template<class T>
static inline void testSupportedDataType(const hkClass& klass, const hkClassNameRegistry& classReg, const hkTypeInfoRegistry& typeReg)
{
	// load packfile
	{
		hkObjectResource* loadedObject = testPackfile<T>(klass, classReg, typeReg);
		HK_TEST(loadedObject != HK_NULL && loadedObject->getContentsPointer(HK_NULL, HK_NULL) != HK_NULL);
		if( loadedObject == HK_NULL || loadedObject->getContentsPointer(HK_NULL, HK_NULL) == HK_NULL )
		{
			HK_WARN_ALWAYS(0x5fc071c8, "Test failed loading " << klass.getName() << " as hkObjectResource from packfile.");
		}
		else
		{
			loadedObject->removeReference();
		}
	}
	// load tagfile
	{
		hkObjectResource* loadedObject = testTagfile<T>(klass, classReg, typeReg);
		HK_TEST(loadedObject != HK_NULL && loadedObject->getContentsPointer(HK_NULL, HK_NULL) != HK_NULL);
		if( loadedObject == HK_NULL || loadedObject->getContentsPointer(HK_NULL, HK_NULL) == HK_NULL )
		{
			HK_WARN_ALWAYS(0x16d9d9f5, "Test failed loading " << klass.getName() << " as hkObjectResource from tagfile.");
		}
		else
		{
			loadedObject->removeReference();
		}
	}
}

#define TEST_SUPPORTED_DATA_TYPE(CPPTYPE) \
	testSupportedDataType<CPPTYPE>(CPPTYPE##Class, classReg, typeReg)

static int loadAsObjectTest()
{
	hkDynamicClassNameRegistry classReg(hkVersionUtil::getCurrentVersion());
	hkTypeInfoRegistry typeReg;
	registerClassesForTest(classReg, typeReg);

	hkError::getInstance().setEnabled(0x43AE1F6B, false);
	TEST_UNSUPPORTED_DATA_TYPE(TestUnsupportedCstring);
	TEST_UNSUPPORTED_DATA_TYPE(TestUnsupportedCstringArray);
	TEST_UNSUPPORTED_DATA_TYPE(TestUnsupportedVariant);
	TEST_UNSUPPORTED_DATA_TYPE(TestUnsupportedVariantArray);
	TEST_UNSUPPORTED_DATA_TYPE(TestUnsupportedSimpleArray);
	TEST_UNSUPPORTED_DATA_TYPE(TestUnsupportedHomogeneousArray);
	TEST_UNSUPPORTED_DATA_TYPE(TestMemberOfUnsupportedClass);
	hkError::getInstance().setEnabled(0x43AE1F6B, true);

	TEST_SUPPORTED_DATA_TYPE(TestSimpleStruct);
	TEST_SUPPORTED_DATA_TYPE(TestSupportedVirtual);
	TEST_SUPPORTED_DATA_TYPE(TestSupportedNonVirtual);
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
//HK_TEST_REGISTER( loadAsObjectTest, "Fast", "Common/Test/UnitTest/Serialize/", "loadAsObjectTest" );

#endif

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
