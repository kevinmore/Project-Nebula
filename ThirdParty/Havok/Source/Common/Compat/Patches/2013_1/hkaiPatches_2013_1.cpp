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

extern void HK_CALL upgradeBitField_2013_1(hkDataObject& dstBitField, hkDataObject& srcBitField);

static void hkaiSilhouetteRecorderSilhouettesSteppedEvent_2_to_3(hkDataObject& obj)
{
	hkDataObject srcBitField = obj["old_updatedSections"].asObject();
	hkDataObject dstBitField = obj["updatedSections"].asObject();
	upgradeBitField_2013_1(dstBitField, srcBitField);
}

static void hkaiCuttingGeometryInfo_0_to_1(hkDataObject& obj)
{
	hkDataObject srcBitField = obj["old_cuttingTriangles"].asObject();
	hkDataObject dstBitField = obj["cuttingTriangles"].asObject();
	upgradeBitField_2013_1(dstBitField, srcBitField);
}

static void setBufferSizes(hkDataObject& bufferSizes, int openSetSize, int searchStateSize)
{
	bufferSizes["maxOpenSetSizeBytes"] = openSetSize;
	bufferSizes["maxSearchStateSizeBytes"] = searchStateSize;

}

static void hkaiNavMeshPathSearchParameters_9_to_10(hkDataObject& obj)
{
	int maxOpenSetSizeBytes = obj["maxOpenSetSizeBytes"].asInt();
	int maxSearchStateSizeBytes = obj["maxSearchStateSizeBytes"].asInt();
	hkDataObject bufferSizes = obj["bufferSizes"].asObject();
	setBufferSizes(bufferSizes, maxOpenSetSizeBytes, maxSearchStateSizeBytes);
	
	int maxHierarchyOpenSetSizeBytes = obj["maxHierarchyOpenSetSizeBytes"].asInt();
	int maxHierarchySearchStateSizeBytes = obj["maxHierarchySearchStateSizeBytes"].asInt();
	hkDataObject hierarchyBufferSizes = obj["hierarchyBufferSizes"].asObject();
	setBufferSizes(hierarchyBufferSizes, maxHierarchyOpenSetSizeBytes, maxHierarchySearchStateSizeBytes);
}

static void hkaiNavVolumePathSearchParameters_3_to_4(hkDataObject& obj)
{
	int maxOpenSetSizeBytes = obj["maxOpenSetSizeBytes"].asInt();
	int maxSearchStateSizeBytes = obj["maxSearchStateSizeBytes"].asInt();
	hkDataObject bufferSizes = obj["bufferSizes"].asObject();
	setBufferSizes(bufferSizes, maxOpenSetSizeBytes, maxSearchStateSizeBytes);
}

static void hkaiGraphPathSearchParameters_1_to_2(hkDataObject& obj)
{
	int maxOpenSetSizeBytes = obj["maxOpenSetSizeBytes"].asInt();
	int maxSearchStateSizeBytes = obj["maxSearchStateSizeBytes"].asInt();
	hkDataObject bufferSizes = obj["bufferSizes"].asObject();
	setBufferSizes(bufferSizes, maxOpenSetSizeBytes, maxSearchStateSizeBytes);
}
static void hkaiStreamingSetGraphConnection_2_to_3(hkDataObject& obj)
{
	obj["edgeCost"] = -1.0f;
}


void HK_CALL registerAiPatches_2013_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2013_1/hkaiPatches_2013_1.cxx>
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
