/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Config/hkProductFeatures.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Compat/hkHavokVersions.h>
#include <Common/Base/Config/hkOptionalComponent.h>


// HK_SERIALIZE_MIN_COMPATIBLE_VERSION can be used to define which is the oldest
// compatible version of Havok assets. Removing compatibility with older versions
// that will never be used in a given project can cause a significant code size
// reduction. The list of supported versions is given in 
// Common/Compat/Deprecated/Compat/hkCompatVersions.h.
// The minimum compatible version is specified using just the release number:
// to have compatibility at most with version 650b1 the user should use
// #define HK_SERIALIZE_MIN_COMPATIBLE_VERSION 650b1.
#ifndef HK_SERIALIZE_MIN_COMPATIBLE_VERSION
// If HK_SERIALIZE_MIN_COMPATIBLE_VERSION is not defined, use all versions (300 being the first Havok version).
#define HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE HK_HAVOK_VERSION_300
#else
#define HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE HK_HAVOK_VERSION(HK_SERIALIZE_MIN_COMPATIBLE_VERSION)
#endif

// Register libraries with the memory tracker, if it is enabled.
// This uses HK_FEATURE_PRODUCT_* and HK_EXCLUDE_LIBRARY_* to decide
// which tracker reflection items are registered.
#if defined(HK_MEMORY_TRACKER_ENABLE) && !defined(HK_EXCLUDE_FEATURE_MemoryTracker)
#include <Common/Base/Memory/Tracker/Registration/hkRegisterTrackedClasses.cxx>
#endif

#if !defined(HK_EXCLUDE_FEATURE_RegisterVersionPatches)
static void registerVersionPatches()
{
	hkVersionPatchManager& man = hkVersionPatchManager::getInstance();
	#include <Common/Compat/Patches/hkRegisterPatches.cxx>
}
#endif

void HK_CALL hkProductFeatures::initialize()
{
//
// Common
//
#if 1
	#if !defined(HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700)
		HK_OPTIONAL_COMPONENT_REQUEST(hkSerializeDeprecated);
	#endif

	#if defined(HK_MEMORY_TRACKER_ENABLE) && !defined(HK_EXCLUDE_FEATURE_MemoryTracker)
		hkRegisterMemoryTracker();
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_RegisterVersionPatches)
		if ( hkVersionPatchManager::getInstance().getNumPatches() > 0 )
		{
			hkVersionPatchManager::getInstance().clearPatches();
		}

		registerVersionPatches();

		hkVersionPatchManager::getInstance().recomputePatchDependencies();
	#endif
#endif

//
// Physics
//
#ifdef HK_FEATURE_PRODUCT_PHYSICS_2012
	#if !defined(HK_EXCLUDE_FEATURE_hkpHeightField)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpHeightFieldAgent);
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_hkpSimulation)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpSimulation);
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_hkpContinuousSimulation)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpContinuousSimulation);
	#endif

	#if (HK_CONFIG_THREAD != HK_CONFIG_SINGLE_THREADED)
		#if !defined(HK_EXCLUDE_FEATURE_hkpMultiThreadedSimulation)
			HK_OPTIONAL_COMPONENT_REQUEST(hkpMultiThreadedSimulation);
		#endif
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_hkpAccurateInertiaTensorComputer)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpAccurateInertiaTensorComputer);
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_hkp3AxisSweep)
		HK_OPTIONAL_COMPONENT_REQUEST(hkp3AxisSweep);
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_hkpTreeBroadPhase)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpTreeBroadPhase);
	#endif

	#if !defined(HK_EXCLUDE_FEATURE_hkpTreeBroadPhase32)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpTreeBroadPhase32);
	#endif

	#if defined(HK_EXCLUDE_FEATURE_hkpSampledHeightFieldDdaRayCast) && defined (HK_EXCLUDE_FEATURE_hkpSampledHeightFieldCoarseTreeRayCast)
		// Do nothing, the function pointers are set to null.	
	#elif defined(HK_EXCLUDE_FEATURE_hkpSampledHeightFieldDdaRayCast)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpSampledHeightFieldShape_CoarseCast);
	#elif defined(HK_EXCLUDE_FEATURE_hkpSampledHeightFieldCoarseTreeRayCast)
		HK_OPTIONAL_COMPONENT_REQUEST(hkpSampledHeightFieldShape_DdaCast);	
	#else
		HK_OPTIONAL_COMPONENT_REQUEST(hkpSampledHeightField_AllCasts);
	#endif
#endif // HK_FEATURE_PRODUCT_PHYSICS_2012

//
// Destruction
//
#if defined(HK_FEATURE_PRODUCT_DESTRUCTION_2012) && !defined(HK_EXCLUDE_FEATURE_DestructionRuntime)
	extern void HK_CALL registerDestructionRuntime();
	registerDestructionRuntime();
#endif
}

// Generate class registration. Use default classes file if no custom class file is provided
#if !defined(HK_EXCLUDE_FEATURE_RegisterReflectedClasses)
	#if !defined(HK_CLASSES_FILE)
		#define HK_CLASSES_FILE <Common/Serialize/Classlist/hkKeyCodeClasses.h>
	#endif
	#include <Common/Serialize/Util/hkBuiltinTypeRegistry.cxx>
#endif

// Set up the deprecated serialization and versioning system.
// You can replace HK_COMPAT_FILE with your own file that contains
// only the versioning steps you need.
#if !defined(HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700)
	#ifndef HK_COMPAT_FILE
		#define HK_COMPAT_FILE <Common/Compat/Deprecated/Compat/hkCompatVersions.h>
	#endif
	#include <Common/Compat/Deprecated/Compat/hkCompat_All.cxx>
#else
	// Codewarrior cannot deadstrip this correctly, so declare empty updaters list and class list
	#if defined( HK_COMPILER_MWERKS ) && !defined(HK_EXCLUDE_LIBRARY_hkCompat)
		#include <Common/Compat/Deprecated/Compat/hkCompat_None.cxx>
	#endif
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
