/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>

// Registration function is at the end of the file

static void hkaiNavMeshPathSearchParameters_3_to_4(hkDataObject& searchParams)
{
	int oldEnum = searchParams["outputPathType"].asInt();
	if (oldEnum == 2) // Smooth and project
	{
		searchParams["outputPathFlags"] = 3; // smooth | project
	}
	else
	{
		searchParams ["outputPathFlags"] = oldEnum; // either none or smooth only
	}
}

static void hkaiNavVolumeGenerationSettings_7_to_8(hkDataObject& genSettings)
{
	hkDataObject defaultConstructionInfo = genSettings["defaultConstructionInfo"].asObject();
	defaultConstructionInfo["flags"] = genSettings["defaultConstructionProperties"];
}

static void hkaiNavMeshPathSearchParameters_5_to_6(hkDataObject& searchParams)
{
	// This could probably be done as a rename, but I'm a little nervous about renaming a bool to flags.
	int oldVal = searchParams["checkForLineOfSight"].asInt();
	if (oldVal == 1) // checkLineOfSight was enabled
	{
		searchParams["lineOfSightFlags"] = 1;
	}
	else
	{
		searchParams ["lineOfSightFlags"] = 0;
	}
}


void HK_CALL registerAiPatches_2011_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_2/hkaiPatches_2011_2.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
}

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
