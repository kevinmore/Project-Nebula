/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Base/Reflection/Registry/hkDefaultClassNameRegistry.h>
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>
#include <Common/Serialize/Util/hkVersionCheckingUtils.h>
#include <Common/Serialize/Util/hkSerializationCheckingUtils.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Compat/hkHavokVersions.h>

#include <Common/Base/KeyCode.h>


#ifndef HK_PLATFORM_LRB

#ifdef HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE
// If HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE defined, we must force it to the first version (we are verifying the patches).
#undef HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE
#endif
#define HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE HK_HAVOK_VERSION_300

#endif

static hkBool hasPrefix(const char* prefix, const char* s)
{
	if(!s)
		return false;

	const char* str = hkString::strStr(s, prefix);
	if (!str)
		return false;

	int prefixLen = hkString::strLen(prefix);
	char nextLetter = str[ prefixLen ]; // should be capitalized or a number
	return (nextLetter >= 'A' && nextLetter <= 'Z' ) || (nextLetter >= '0' && nextLetter <= '9');
}

static hkBool hasPrefixes(const char** prefix, int numPrefixes, const char* s)
{
	for (int i=0; i<numPrefixes; i++)
	{
		if (hasPrefix(prefix[i], s))
			return true;
	}
	return false;
}

#define HK_INITIAL_ENTRY 0x80000000

static void registerInitialClassPatches(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/hkInitialClassPatches.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
}

static void HK_CALL registerKeycodePatches(hkVersionPatchManager& man)
{
#	include <Common/Compat/Patches/hkRegisterPatches.cxx>
}

static int checkPatchProducts()
{
	{
		const char* prefixes[] = { "hk", "hkx" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}

	{
		const char* prefixes[] = { "hkcd" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkcdAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}

#if defined(HK_FEATURE_PRODUCT_PHYSICS_2012)
	{
		const char* prefixes[] = { "hkp", "hkcd" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkpAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_PHYSICS)
	{
#	if !defined(HK_FEATURE_PRODUCT_PHYSICS_2012)
		const char* prefixes[] = { "hknp" , "hkp" };
#	else
		const char* prefixes[] = { "hknp" };
#	endif
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hknpAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_ANIMATION) 
	{
		const char* prefixes[] = { "hka", "hknpRagdoll" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkaAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_BEHAVIOR) 
	{
		const char* prefixes[] = { "hkb", "hkbp", "hkbnp" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkbAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_CLOTH) 
	{
		const char* prefixes[] = { "hcl" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hclAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_DESTRUCTION_2012)
	{
		const char* prefixes[] = { "hkd" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkdAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_DESTRUCTION)
	{
		const char* prefixes[] = { "hknd" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkndAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_AI) 
	{
		const char* prefixes[] = { "hkai" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkaiAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

#if defined(HK_FEATURE_PRODUCT_MILSIM) 
	{
		const char* prefixes[] = { "hkms" };
#		define HK_PATCHES_FILE <Common/Compat/Patches/Utilities/hkmsAllPatches.cxx>
#		include <Common/Serialize/UnitTest/patchesVersionCheck.cxx>
#		undef HK_PATCHES_FILE
	}
#endif

	return 0;
}

namespace hkHavokCurrentClasses
{
	extern const hkStaticClassNameRegistry hkHavokDefaultClassRegistry;
}

static int patchesVersioningCheck()
{
#if !defined(HK_PLATFORM_LRB) && !defined(HK_REAL_IS_DOUBLE)
	hkSerializationCheckingUtils::DeferredErrorStream deferred;
	hkOstream output(&deferred);

	hkDataWorldDict world;

	hkDefaultClassNameRegistry reg;
	reg.merge(hkHavokCurrentClasses::hkHavokDefaultClassRegistry);

	hkVersionPatchManager manager;
	registerInitialClassPatches(manager);
	registerKeycodePatches(manager);
	manager.recomputePatchDependencies();

	hkResult res = hkVersionCheckingUtils::verifyClassPatches(output, world, reg, manager, hkVersionCheckingUtils::NONE);
	if( res != HK_SUCCESS )
	{
		deferred.dump();
	}
	HK_TEST( res == HK_SUCCESS );

#else
	HK_REPORT("hkSerialize tests that use hkCompat are not run on this platform" );
#endif
	return 0;
}

static int patchesCheckMain()
{
	checkPatchProducts();
	patchesVersioningCheck();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(patchesCheckMain, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__ );

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
