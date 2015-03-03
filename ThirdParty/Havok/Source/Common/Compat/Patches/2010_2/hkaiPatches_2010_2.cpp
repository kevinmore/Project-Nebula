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

static void hkaiNavMeshCutter_7_to_8(hkDataObject& obj)
{
	hkDataArray src = obj["old_meshInfos"].asArray();
	hkDataArray dst = obj["meshInfos"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

// Create m_searchParameters struct inside hkaiPathfindingUtil::FindPathInput
static void hkaiPathfindingUtilFindPathInput_3_to_4(hkDataObject& obj)
{
	// Get the parameter values from the old version.
	const hkVector4 up = obj["up"].asVector4();
	const int outputPathType = obj["outputPathType"].asInt();
	const int checkForDirectPath = obj["checkForDirectPath"].asInt();
	const int useHierarchicalHeuristic = obj["useHierarchyInfo"].asInt();
	const hkReal heuristicWeight = obj["heuristicWeight"].asReal();
	const hkReal simpleRadiusThreshold = obj["simpleRadiusThreshold"].asReal();

	hkDataObject searchParameters = obj["searchParameters"].asObject();
	searchParameters["up"] = up;
	searchParameters["outputPathType"] = outputPathType;
	searchParameters["checkForDirectPath"] = checkForDirectPath;
	searchParameters["useHierarchicalHeuristic"] = useHierarchicalHeuristic;
	searchParameters["heuristicWeight"] = heuristicWeight;
	searchParameters["simpleRadiusThreshold"] = simpleRadiusThreshold;
}

// Pushing memory limit parameters into m_searchParameters struct
static void hkaiPathfindingUtilFindPathInput_5_to_6(hkDataObject& obj)
{
	// Get the parameter values from the old version.
	const int maxOpenSetSizeBytes = obj["maxOpenSetSizeBytes"].asInt();
	const int maxSearchStateSizeBytes = obj["maxSearchStateSizeBytes"].asInt();
	const int maxHierarchyOpenSetSizeBytes = obj["maxHierarchyOpenSetSizeBytes"].asInt();
	const int maxHierarchySearchStateSizeBytes = obj["maxHierarchySearchStateSizeBytes"].asInt();

	hkDataObject searchParameters = obj["searchParameters"].asObject();
	searchParameters["maxOpenSetSizeBytes"] = maxOpenSetSizeBytes;
	searchParameters["maxSearchStateSizeBytes"] = maxSearchStateSizeBytes;
	searchParameters["maxHierarchyOpenSetSizeBytes"] = maxHierarchyOpenSetSizeBytes;
	searchParameters["maxHierarchySearchStateSizeBytes"] = maxHierarchySearchStateSizeBytes;
}

// Create m_searchParameters struct inside hkaiVolumePathfindingUtil::FindPathInput
static void hkaiVolumePathfindingUtilFindPathInput_2_to_3(hkDataObject& obj)
{
	// Get the parameter values from the old version.
	const hkVector4 up = obj["up"].asVector4();
	const int checkForDirectPath = obj["checkForDirectPath"].asInt();
	const hkReal heuristicWeight = obj["heuristicWeight"].asReal();
	const int maxOpenSetSizeBytes = obj["maxOpenSetSizeBytes"].asInt();
	const int maxSearchStateSizeBytes = obj["maxSearchStateSizeBytes"].asInt();

	hkDataObject searchParameters = obj["searchParameters"].asObject();
	searchParameters["up"] = up;
	searchParameters["checkForDirectPath"] = checkForDirectPath;
	searchParameters["heuristicWeight"] = heuristicWeight;
	searchParameters["maxOpenSetSizeBytes"] = maxOpenSetSizeBytes;
	searchParameters["maxSearchStateSizeBytes"] = maxSearchStateSizeBytes;
}

// Fix default value of unset section UID.
static void hkaiDirectedGraphExplicitCost_2_to_3(hkDataObject& obj)
{
	int sectionUid = obj["sectionUid"].asInt();
	if ( sectionUid == -1 )
		obj["sectionUid"] = 0;
}

// Fix default value of unset section UID.
static void hkaiNavVolume_6_to_7(hkDataObject& obj)
{
	int sectionUid = obj["sectionUid"].asInt();
	if ( sectionUid == -1 )
		obj["sectionUid"] = 0;
}

static void hkaiNavMeshGenerationSettings_16_to_17(hkDataObject& obj)
{
	hkDataObject pruneSettings = obj["regionPruningSettings"].asObject();
	pruneSettings["minRegionArea"] = obj["minRegionArea"].asReal();
	pruneSettings["minDistanceToSeedPoints"] = obj["minDistanceToSeedPoints"].asReal();
	pruneSettings["regionSeedPoints"] = obj["regionSeedPoints"].asArray();
}

void HK_CALL registerAiPatches_2010_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_2/hkaiPatches_2010_2.cxx>
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
