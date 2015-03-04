/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Reflection/Registry/hkDefaultClassNameRegistry.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/UnitTest/PatchObjectTest/patchObjectTest.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>

struct DeferredErrorStream : public hkStreamWriter
{
	int write(const void* buf, int nbytes)	{ m_data.insertAt( m_data.getSize(), (char*)buf, nbytes ); return nbytes; }

	hkBool isOk() const { return true; }

	void clear() { m_data.clear(); }

	void dump() 
	{
		m_data.pushBack('\0'); // ensure termination
		hkError::getInstance().message(hkError::MESSAGE_REPORT, 1, m_data.begin(), __FILE__, __LINE__);
	}

	hkArray<char> m_data;
};

#ifdef WRITE_TAGFILE
// Write the file to be loaded
static int patchSerializedObject()
{
	hkDefaultClassNameRegistry::getInstance().registerClass(&PatchObjectTestClass);
	hkDefaultClassNameRegistry::getInstance().registerClass(&PatchObjectTestSubClassClass);
	PatchObjectTest *writeObject = new(PatchObjectTest);

	writeObject->m_firstMember = 25;
	writeObject->m_testStructArray.setSize(4);
	for(int i=0;i<4;i++)
	{
		writeObject->m_testStructArray[i].m_testBigMember = i;
		writeObject->m_testStructArray[i].m_testRealMember = 0.0f;
	}
	writeObject->m_middleMember = 50;
	writeObject->m_embeddedStruct.m_testBigMember = -1;
	writeObject->m_embeddedStruct.m_testRealMember = -1.0f;
	writeObject->m_lastMember = 75;

	hkSerializeUtil::saveTagfile(writeObject, PatchObjectTestClass, hkOstream("Resources/Common/Api/Serialize/PatchObjectTest/patchObjectTest.hkt").getStreamWriter());

	return 0;
}

#else

static void HK_CALL PatchObjectTestSubClass_1_to_2(hkDataObject &obj)
{
	if(obj["testBigMember"].asInt() % 2)
	{
		obj["testAddedMember"] = 1;
	}
	else
	{
		obj["testAddedMember"] = 15;
	}
}

static void HK_CALL PatchObjectTest_0_to_1(hkDataObject &obj)
{
	// Come back to this
}

static void registerTestPatches(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Serialize/UnitTest/PatchObjectTest/patchObjectTestPatches.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
}

extern const hkTypeInfo PatchObjectTestTypeInfo;
extern const hkTypeInfo PatchObjectTestSubClassTypeInfo;
extern const hkTypeInfo PatchObjectTestExtraAddedTypeInfo;

static int patchSerializedObject()
{
	hkDynamicClassNameRegistry newClassNameRegistry;
	hkTypeInfoRegistry newTypeRegistry;

	hkVersionPatchManager *oldVersionPatchManager = &(hkVersionPatchManager::getInstance());
	hkVersionPatchManager::getInstance().addReference();

	hkVersionPatchManager newVersionPatchManager;
	hkVersionPatchManager::replaceInstance(&newVersionPatchManager);
	newVersionPatchManager.addReference();

	newClassNameRegistry.registerClass(&PatchObjectTestClass);
	newClassNameRegistry.registerClass(&PatchObjectTestSubClassClass);
	newClassNameRegistry.registerClass(&PatchObjectTestExtraAddedClass);
	newTypeRegistry.registerTypeInfo(&PatchObjectTestTypeInfo);
	newTypeRegistry.registerTypeInfo(&PatchObjectTestSubClassTypeInfo);
	newTypeRegistry.registerTypeInfo(&PatchObjectTestExtraAddedTypeInfo);
	
	registerTestPatches(hkVersionPatchManager::getInstance());
	newVersionPatchManager.recomputePatchDependencies();

	hkSerializeUtil::LoadOptions loadOptions; loadOptions.useClassNameRegistry(&newClassNameRegistry); loadOptions.useTypeInfoRegistry(&newTypeRegistry);
	hkResource* resource = hkSerializeUtil::loadOnHeap("Resources/Common/Api/Serialize/PatchObjectTest/patchObjectTest.hkt", HK_NULL, loadOptions);
	HK_TEST(resource != HK_NULL);
	if(!resource)
	{
        hkVersionPatchManager::replaceInstance(oldVersionPatchManager);
		return 0;
	}
	PatchObjectTest* contents = resource->getContentsWithRegistry<PatchObjectTest>(&newTypeRegistry);
	HK_TEST(contents != HK_NULL);

	HK_TEST(contents->m_firstMember == 25);
	HK_TEST(contents->m_middleMember == 50);
	HK_TEST(contents->m_lastMember == 75);

	HK_TEST(contents->m_embeddedStruct.m_testBigMember == -1);
	HK_TEST(contents->m_embeddedStruct.m_testAddedMember == 1);
	HK_TEST(contents->m_embeddedStruct.m_testEmbedStructAdd.m_extraInt == 1);

	for(int i=0;i<contents->m_testStructArray.getSize();i++)
	{
		HK_TEST(contents->m_testStructArray[i].m_testBigMember == i);
		if(contents->m_testStructArray[i].m_testBigMember % 2)
		{
			HK_TEST(contents->m_testStructArray[i].m_testAddedMember == 1);
		}
		else
		{
			HK_TEST(contents->m_testStructArray[i].m_testAddedMember == 15);
		}
		HK_TEST(contents->m_testStructArray[i].m_testEmbedStructAdd.m_extraInt == 1);
	}

	resource->removeReference();

	hkVersionPatchManager::replaceInstance(oldVersionPatchManager);
	return 0;
}
#endif

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(patchSerializedObject, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__ );

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
