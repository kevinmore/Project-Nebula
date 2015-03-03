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

static void hkaiCharacter_20_to_21(hkDataObject& obj)
{
	int avoidanceEnabled = obj["avoidanceEnabled"].asInt();
	obj["avoidanceEnabledMask"] = (avoidanceEnabled ? 7 : 0);
}

static void hkaiNavMeshInstance_0_to_1(hkDataObject& obj)
{
	const hkDataWorld* world = obj.getClass().getWorld();
	hkDataClass referenceFrameClass( world->findClass("hkaiReferenceFrame") );
	hkDataObject referenceFrame( world->newObject( referenceFrameClass ) );
	referenceFrame["transform"] = obj["transform"];
	obj["referenceFrame"] = referenceFrame;
}

static void hkaiNavMeshGenerationSettings_18_to_19(hkDataObject& obj)
{
	hkDataArray src = obj["materialSettingsMap"].asArray();
	hkDataArray dst = obj["localSettings"].asArray();

	const int srcSize = src.getSize();
	const int dstSize = dst.getSize();
	int newSize = srcSize + dstSize;
	dst.setSize(newSize);
	for (int i = 0; i < srcSize; i++)
	{
		hkDataObject localSetting = dst[i+dstSize].asObject();
		hkDataObject materialSetting = src[i].asObject();

		//localSetting["volume"].setNull();
		localSetting["material"] = materialSetting["materialIndex"];
		localSetting["simplificationSettings"] = materialSetting["simplificationSettings"];

		// inherit these from the global settings
		localSetting["maxWalkableSlope"] = obj["maxWalkableSlope"];
		localSetting["edgeMatchingParams"] = obj["edgeMatchingParams"];
	}
}

static void hkaiNavMeshInstance_1_to_2(hkDataObject& navMeshInstance)
{
	hkDataArray ownedEdges = navMeshInstance["ownedEdges"].asArray();
	hkDataArray cuttingInfo = navMeshInstance["cuttingInfo"].asArray();

	int numOwnedEdges = ownedEdges.getSize();
	cuttingInfo.setSize( numOwnedEdges );

	for (int i=0; i<numOwnedEdges; i++)
	{
		hkDataObject edge = ownedEdges[i].asObject();
		int oldVal = edge["cutInfo"].asInt();

		// If old max, use new max
		cuttingInfo[i] = (oldVal == 255) ? 0xFFFF : oldVal;
	}
}

void HK_CALL registerAiPatches_2011_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_1/hkaiPatches_2011_1.cxx>
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
