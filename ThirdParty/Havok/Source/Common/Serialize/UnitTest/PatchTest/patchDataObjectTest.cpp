/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/Registry/hkDynamicClassNameRegistry.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Serialize/Util/hkVersionCheckingUtils.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>

extern const hkClass hkTestAClass;
extern const hkClass hkTestBClass;
extern const hkClass hkTestCClass;
extern const hkClass hkTestPatchObjectClass;

// old hkTestA, hkTestB, hkTestC
namespace hkTestOldVersions
{
	//
	// Class hkTestC
	//
	static hkInternalClassMember hkTestCClass_Members[] =
	{
		{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	extern hkClass hkTestCClass;
	hkClass hkTestCClass(
		"hkTestC",
		HK_NULL, // parent
		0,
		HK_NULL,
		0, // interfaces
		HK_NULL,
		0, // enums
		reinterpret_cast<const hkClassMember*>(hkTestCClass_Members),
		HK_COUNT_OF(hkTestCClass_Members),
		HK_NULL, // defaults
		HK_NULL, // attributes
		0, // flags
		0 // version
		);
	//
	// Class hkTestA
	//
	static hkInternalClassMember hkTestAClass_Members[] =
	{
		{ "object", &hkTestCClass, HK_NULL, hkClassMember::TYPE_POINTER, hkClassMember::TYPE_STRUCT, 0, 0, 0, HK_NULL },
		{ "float", HK_NULL, HK_NULL, hkClassMember::TYPE_REAL, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	extern hkClass hkTestAClass;
	hkClass hkTestAClass(
		"hkTestA",
		HK_NULL, // parent
		0,
		HK_NULL,
		0, // interfaces
		HK_NULL,
		0, // enums
		reinterpret_cast<const hkClassMember*>(hkTestAClass_Members),
		HK_COUNT_OF(hkTestAClass_Members),
		HK_NULL, // defaults
		HK_NULL, // attributes
		0, // flags
		0 // version
		);

	//
	// Class hkTestB
	//
	static hkInternalClassMember hkTestBClass_Members[] =
	{
		{ "string", HK_NULL, HK_NULL, hkClassMember::TYPE_CSTRING, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	extern hkClass hkTestBClass;
	hkClass hkTestBClass(
		"hkTestB",
		HK_NULL, // parent
		0,
		HK_NULL,
		0, // interfaces
		HK_NULL,
		0, // enums
		reinterpret_cast<const hkClassMember*>(hkTestBClass_Members),
		HK_COUNT_OF(hkTestBClass_Members),
		HK_NULL, // defaults
		HK_NULL, // attributes
		0, // flags
		0 // version
		);

	//
	// Class hkTestPatchObject
	//
	static hkInternalClassMember hkTestPatchObjectClass_Members[] =
	{
		{ "intMember", HK_NULL, HK_NULL, hkClassMember::TYPE_INT32, hkClassMember::TYPE_VOID, 0, 0, 0, HK_NULL }
	};
	extern hkClass hkTestPatchObjectClass;
	hkClass hkTestPatchObjectClass(
		"hkTestPatchObject",
		HK_NULL, // parent
		0,
		HK_NULL,
		0, // interfaces
		HK_NULL,
		0, // enums
		reinterpret_cast<const hkClassMember*>(hkTestPatchObjectClass_Members),
		HK_COUNT_OF(hkTestPatchObjectClass_Members),
		HK_NULL, // defaults
		HK_NULL, // attributes
		0, // flags
		0 // version
		);
}

static void hkTestPatchObject_1_to_2(hkDataObject& obj)
{
	hkDataObject structObject = obj["structMember"].asObject();
	HK_TEST(structObject.getImplementation() != HK_NULL);
}

static void registerTestPatches(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Serialize/UnitTest/PatchTest/patchDataObjectTest.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
}

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

static int patchDataObject()
{
	DeferredErrorStream deferred;
	hkOstream output(&deferred);
	hkVersionPatchManager manager;
	registerTestPatches(manager);
	manager.recomputePatchDependencies();
	hkDynamicClassNameRegistry classReg;
	classReg.registerClass(&hkTestAClass);
	classReg.registerClass(&hkTestBClass);
	classReg.registerClass(&hkTestCClass);
	classReg.registerClass(&hkTestPatchObjectClass);
	hkResult res;
	{
		hkDataWorldDict noClassesWorld;
		res = hkVersionCheckingUtils::verifyClassPatches(output, noClassesWorld, classReg, manager, hkVersionCheckingUtils::VERBOSE);
		if( res != HK_SUCCESS )
		{
			deferred.dump();
		}
		HK_TEST(res == HK_SUCCESS);
	}
	hkDataWorldDict world;
	{
		hkDataClassImpl* classAimpl;
		hkDataClassImpl* classBimpl;
		hkDataClassImpl* classCimpl;
		hkDataClassImpl* testPatchObjectimpl;
		// add classes
		{
			HK_TEST(world.findClass(hkTestOldVersions::hkTestCClass.getName()) == HK_NULL);
			classCimpl = world.wrapClass(hkTestOldVersions::hkTestCClass);
			HK_TEST(classCimpl != HK_NULL);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestCClass.getName()) == classCimpl);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestAClass.getName()) == HK_NULL);
			classAimpl = world.wrapClass(hkTestOldVersions::hkTestAClass);
			HK_TEST(classAimpl != HK_NULL);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestAClass.getName()) == classAimpl);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestBClass.getName()) == HK_NULL);
			classBimpl = world.wrapClass(hkTestOldVersions::hkTestBClass);
			HK_TEST(classBimpl != HK_NULL);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestBClass.getName()) == classBimpl);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestPatchObjectClass.getName()) == HK_NULL);
			testPatchObjectimpl = world.wrapClass(hkTestOldVersions::hkTestPatchObjectClass);
			HK_TEST(testPatchObjectimpl != HK_NULL);
			HK_TEST(world.findClass(hkTestOldVersions::hkTestPatchObjectClass.getName()) == testPatchObjectimpl);
			// Need to actually add an object to test function patches
			world.newObject(hkDataClass(testPatchObjectimpl));
		}

		hkDefaultClassWrapper wrapper(&classReg);
		res = manager.applyPatches(world, &wrapper);
		HK_TEST(res == HK_SUCCESS);
		{
			hkVersionPatchManager noPatchesManager;
			res = hkVersionCheckingUtils::verifyClassPatches(output, world, classReg, noPatchesManager, hkVersionCheckingUtils::VERBOSE);
			if( res != HK_SUCCESS )
			{
				deferred.dump();
			}
			HK_TEST(res == HK_SUCCESS);
		}
	}
	
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(patchDataObject, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__ );

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
