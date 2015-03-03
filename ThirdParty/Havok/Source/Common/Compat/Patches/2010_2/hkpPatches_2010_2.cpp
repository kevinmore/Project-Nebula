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

static void hkpConvexVerticesShape_3_to_4(hkDataObject& obj)
{
	hkDataArray oldVerts = obj["rotatedVertices"].asArray();
	hkDataArray newVerts = obj["rotatedVerticesNew"].asArray();
	const int numVerts = oldVerts.getSize();
	newVerts.setSize(numVerts);

	for (int i = 0; i < numVerts; ++i)
	{
		// Get old vertex
		hkDataObject oldVtx = oldVerts[i].asObject();

		// Get new vertex
		hkDataObject newVtx = newVerts[i].asObject();
		hkDataArray newVtxValues = newVtx["vertices"].asArray();

		// Convert from the old to the new format
		newVtxValues[0] = oldVtx["x"].asVector4();
		newVtxValues[1] = oldVtx["y"].asVector4();
		newVtxValues[2] = oldVtx["z"].asVector4();
	}
}

static void hkpCompressedMeshShape_10_11(hkDataObject& obj)
{
	{
		hkDataArray src = obj["old_chunks"].asArray();
		hkDataArray dst = obj["chunks"].asArray();

		const int size = src.getSize();
		dst.setSize(size);
		for (int i = 0; i < size; i++)
		{
			dst[i] = src[i].asObject();
		}
	}

	{
		hkDataArray src = obj["old_convexPieces"].asArray();
		hkDataArray dst = obj["convexPieces"].asArray();

		const int size = src.getSize();
		dst.setSize(size);
		for (int i = 0; i < size; i++)
		{
			dst[i] = src[i].asObject();
		}
	}
}

static void hkpExtendedMeshShape_3_to_4(hkDataObject& obj)
{
	hkDataArray src = obj["old_shapesSubparts"].asArray();
	hkDataArray dst = obj["shapesSubparts"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkpSimpleContactConstraintDataInfo_1_to_2(hkDataObject& obj)
{
	HK_ASSERT2(0xad83433, false, "Patching of hkpSimpleContactConstraintDataInfo not implemented.");
}

static void hkpWorldCinfo_13_to_14(hkDataObject& obj)
{
	/*
	Convert from broad phase flags to enum:
	BROADPHASE_TYPE_SAP = 0
	BROADPHASE_TYPE_TREE = 1
	BROADPHASE_TYPE_HYBRID = 2
	BROADPHASE_TYPE_SAP_AND_KD_TREE_DEPRECATED = 3
	*/

	hkBool useKdTree			= obj["useKdTree"].asInt();
	hkBool useMultipleTree		= obj["useMultipleTree"].asInt();
	hkBool useHybridBroadphase	= obj["useHybridBroadphase"].asInt();
	hkBool standaloneBroadphase	= obj["standaloneBroadphase"].asInt();

	obj["broadPhaseType"] = 0;
	if( useHybridBroadphase && standaloneBroadphase )
	{
		obj["broadPhaseType"] = 1;
	}
	else if( useHybridBroadphase )
	{
		obj["broadPhaseType"] = 2;
	}
	else if( useKdTree )
	{
		obj["broadPhaseType"] = 3;
	}
}

void HK_CALL registerPhysicsPatches_2010_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_2/hkpPatches_2010_2.cxx>
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
