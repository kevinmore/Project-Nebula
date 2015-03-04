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

static void hkaiCharacter_4_to_5(hkDataObject& obj)
{
	const hkDataWorld* world = obj.getClass().getWorld();
	hkDataClass steeringClass( world->findClass("hkaiDefaultLocalSteering") );
	hkDataObject steering( world->newObject( steeringClass ) );
	steering["pathFollowingProperties"] = obj["pathFollowingProperties"];
	steering["avoidanceProperties"] = obj["avoidanceProperties"];
	obj["localSteering"] = steering;
}

static void hkaiNavMesh_1_to_2(hkDataObject& obj)
{
	hkDataObject aabb = obj["aabb"].asObject();

	hkDataArray meshVerts = obj["vertices"].asArray();
	hkVector4 vMin, vMax;
	vMin.setAll(HK_REAL_MAX);
	vMax.setAll(-HK_REAL_MAX);

	for (int i=0; i<meshVerts.getSize(); i++)
	{
		hkVector4 v = meshVerts[i].asVector4();
		vMin.setMin(vMin, v);
		vMax.setMax(vMax, v);
	}

	aabb["min"] = vMin;
	aabb["max"] = vMax;
}

void HK_CALL registerAiPatches_660(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/660/hkaiPatches_660.cxx>
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
