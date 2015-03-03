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

static void hkpBreakableShape_0_to_1(hkDataObject& obj)
{
	obj["physicsShape"] = obj["oldPhysicsShape"];
}

static void hkpWorldCinfo_16_to_17(hkDataObject& obj)
{
	// Convert (removed) BROADPHASE_TYPE_SAP_AND_KD_TREE_DEPRECATED to BROADPHASE_TYPE_TREE
	if( obj["broadPhaseType"].asInt() == 3 )	// BROADPHASE_TYPE_SAP_AND_KD_TREE_DEPRECATED
	{
		obj["broadPhaseType"] = 2;	// BROADPHASE_TYPE_HYBRID
	}
}

static void hkpConvexTransformShape_1_to_2(hkDataObject& obj)
{
	// As it is not possible to obtain the child shapes's aabb here to compute the extra scale we will set both 
	// extra scale and cached aabb center (stored in W components) to zero. This will make the loaded shape behave
	// as in version 1, that is with no convex radius scaling.
	hkQsTransform transform = obj["transform"].asQsTransform();
	transform.m_translation.zeroComponent<3>();
	transform.m_scale.zeroComponent<3>();
	obj["transform"] = transform;
	obj["extraScale"] = hkVector4::getZero();
}

void HK_CALL registerPhysicsPatches_2011_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_2/hkpPatches_2011_2.cxx>
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
