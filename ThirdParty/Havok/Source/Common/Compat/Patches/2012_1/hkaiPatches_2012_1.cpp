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

static void _copyDataToStridedArray( hkDataArray& srcArray, hkDataArray& dstArray )
{
	dstArray.setSize( srcArray.getSize() );

	const int n = srcArray.getSize();
	for (int i=0; i<n; i++)
	{
		hkDataObject src = srcArray[i].asObject();
		dstArray[i] = src["data"].asInt();
	}
}

static void _copyArrays( hkDataObject& obj, const char* oldArrayName, const char* newArrayName )
{
	hkDataArray oldArray = obj[oldArrayName].asArray();
	hkDataArray newArray = obj[newArrayName].asArray();
	_copyDataToStridedArray( oldArray, newArray );
}

// Registration function is at the end of the file
static void hkaiNavMesh_13_to_14(hkDataObject& navMeshObj)
{
	// Assume that old striding was 1
	_copyArrays(navMeshObj, "faces", "faceData");
	_copyArrays(navMeshObj, "edges", "edgeData");
}

static void hkaiNavMeshInstance_3_to_4(hkDataObject& instanceObj)
{
	_copyArrays(instanceObj, "ownedFaces", "ownedFaceData");
	_copyArrays(instanceObj, "ownedEdges", "ownedEdgeData");
	_copyArrays(instanceObj, "instancedFaces", "instancedFaceData");
	_copyArrays(instanceObj, "instancedEdges", "instancedEdgeData");
}

namespace
{
	hkReal convertWallFollowingFactorToAngle(hkReal wallFollowingFactor)
	{
		return hkMath::acos(hkMath::clamp(hkReal(1-wallFollowingFactor/2), 0, 1));
	}
}

static void hkaiAvoidanceSolverAvoidanceProperties_9_to_10(hkDataObject& obj)
{
	hkDataObject aabb = obj["localSensorAabb"].asObject();
	hkVector4 vMin; vMin.setAll(-hkSimdReal_5);
	hkVector4 vMax; vMax.setAll(hkSimdReal_5);
	aabb["min"] = vMin;
	aabb["max"] = vMax;

	hkReal wallFollowingFactor = obj["wallFollowingFactor"].asReal();
	hkReal wallFollowingAngle = convertWallFollowingFactorToAngle(wallFollowingFactor);
	obj["wallFollowingAngle"] = wallFollowingAngle;
}

static void hkaiLocalSteeringInput_4_to_5(hkDataObject& obj)
{
	hkReal wallFollowingFactor = obj["wallFollowingFactor"].asReal();
	hkReal wallFollowingAngle = convertWallFollowingFactorToAngle(wallFollowingFactor);
	obj["wallFollowingAngle"] = wallFollowingAngle;
}

static void hkaiEdgePathEdge_0_to_1(hkDataObject& obj)
{
	hkInt32 oldFaceKey = obj["face"].asInt();
	hkDataObject persistentKey = obj["facePersistent"].asObject();
	persistentKey["key"] = oldFaceKey;
	persistentKey["offset"] = -1;
}
// static void hkaiRigidBodySilhouetteGenerator_0_to_1(hkDataObject& silGen)
// {
// 
// }

void HK_CALL registerAiPatches_2012_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2012_1/hkaiPatches_2012_1.cxx>
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
