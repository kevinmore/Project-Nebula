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

static void hclSimClothSetupObject_2_to_3(hkDataObject& obj)
{
	// We added a vertex selection (of pinched enabled particles).
	// The default is incorrect, because the vertex selection type gets set to 0, which corresponds to VERTEX_SELECTION_ALL,
	// but we want no particle pinch-detection enabled by default.
	hkDataObject pinchDetectionEnabledVertexSelInput = obj["pinchDetectionEnabledParticles"].asObject();
	pinchDetectionEnabledVertexSelInput["type"] = 1; // because hclVertexSelectionInput::VERTEX_SELECTION_NONE = 1	
}

static void hclSimClothData_4_to_5(hkDataObject& obj)
{
	const hkUint32 numStaticCollidables = obj["perInstanceCollidables"].asArray().getSize();

	hkDataArray collidablePinchingDatas = obj["collidablePinchingDatas"].asArray();
	collidablePinchingDatas.setSize(numStaticCollidables);

	for (hkUint32 i=0; i<numStaticCollidables; ++i)
	{
		hkDataObject collidablePinchingData = collidablePinchingDatas[i].asObject();
		collidablePinchingData["pinchDetectionEnabled"] = false;
		collidablePinchingData["pinchDetectionPriority"] = 0;
		collidablePinchingData["pinchDetectionRadius"] = 0.0f;
	}
}

static void hclSimClothData_6_to_7(hkDataObject& obj)
{
	const int numParticles = obj["particleDatas"].asArray().getSize();

	hkDataArray pinchEnabledFlags = obj["perParticlePinchDetectionEnabledFlags"].asArray();
	pinchEnabledFlags.setSize(numParticles);
	for (int i=0; i<numParticles; ++i)
	{
		pinchEnabledFlags[i] = 0;
	}
}

static void hclSkinOperatorBoneInfluence_0_to_1(hkDataObject& obj)
{
	hkUint8 boneIndex = static_cast<hkUint8> (obj["old_boneIndex"].asInt());
	obj["boneIndex"] = static_cast<hkUint16> (boneIndex);
}

static void hclSkinOperatorBoneInfluence_1_to_2(hkDataObject& obj)
{
	hkUint16 boneIndex = static_cast<hkUint16> (obj["old_boneIndex"].asInt());
	obj["boneIndex"] = static_cast<hkUint8> (boneIndex);
}

static void hclBufferDefinition_0_to_1(hkDataObject& obj)
{
	hkDataObject layout = obj["bufferLayout"].asObject();
	layout["numSlots"]=hkUint8(255);
}

static void hclLocalRangeSetupObject_0_to_1(hkDataObject& obj)
{
	obj["stiffness"] = 1.0f;
	obj["useMaxNormalDistance"] = true;	 	
	obj["useMinNormalDistance"] = true;
}

static void hclClothState_0_to_1(hkDataObject& obj)
{
	hkDataArray oldUsedTransformSets = obj["old_usedTransformSets"].asArray();
	const int numEntries = oldUsedTransformSets.getSize();

	hkDataArray newUsedTransformSets = obj["usedTransformSets"].asArray();
	newUsedTransformSets.setSize(numEntries);

	for (int i=0; i<numEntries; ++i)
	{
		hkDataObject entry = newUsedTransformSets[i].asObject();
		entry["transformSetIndex"] = oldUsedTransformSets[i].asInt();
		hkDataArray flags = entry["transformSetUsage"].asObject()["perComponentFlags"].asArray();
		flags[0] = 1; // Transforms read
		flags[1] = 0; // Inverse Transposes not used (90% of cases)
	}
}

void HK_CALL registerClothPatches_2010_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_1/hclPatches_2010_1.cxx>
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
