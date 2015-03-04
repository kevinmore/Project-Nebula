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

static void hkaiNavMeshGenerationSettings_5_to_6(hkDataObject& obj)
{
	obj["edgeMatchingMetric"] = 2;
	obj["edgeConnectionIterations"] = 2;
}

static void hkaiCharacter_10_to_11(hkDataObject& obj)
{
	obj["desiredVelocity"] = hkVector4::getZero();
}

static void hkaiNavMeshGenerationSettings_6_to_7(hkDataObject& obj)
{
	obj["carvedMaterial"] = int(0x8fffffff);
	obj["carvedCuttingMaterial"] = int(0x8ffffffe);
}

static void hkaiCharacter_12_to_13(hkDataObject& obj)
{
	obj["currentNavMeshFace"] = -1;
}

static void hkaiWorld_13_to_14(hkDataObject& obj)
{
	obj["numPathRequestsPerJob"] = 16;
	obj["numBehaviorUpdatesPerJob"] = 16;
	obj["numCharactersPerAvoidanceJob"] = 16;
}

static void hkaiNavMeshFace_1_to_2(hkDataObject& obj)
{
	obj["padding"] = 0xcdcd;
}

static void hkaiWorld_14_to_15(hkDataObject& obj)
{
	obj["maxPathSearchEdgesOut"] = 64;
	obj["maxPathSearchSegmentsOut"] = 32;
}

static void hkaiNavMeshCostModifier_0_to_1(hkDataObject& obj)
{
	obj["type"] = 1;
}

static void hkaiWorld_15_to_16(hkDataObject& obj)
{
}

static void hkaiNavMeshGenerationSettings_9_to_10(hkDataObject& obj)
{
	hkDataObject simpSettings = obj["simplificationSettings"].asObject();
	obj["loopReplacementArea"] = simpSettings["holeReplacementArea"].asReal();
}

void HK_CALL registerAiPatches_710(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/710/hkaiPatches_710.cxx>
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
