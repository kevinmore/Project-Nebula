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

static void hkpStiffSpringConstraintAtom_0_to_1(hkDataObject& obj)
{
	hkReal minLength = obj["length"].asReal();
	obj["maxLength"] = minLength;
}

static void hkpStiffSpringConstraintDataAtoms_0_to_1(hkDataObject& obj)
{
	hkDataObject setupStabilization = obj["setupStabilization"].asObject();
	setupStabilization["type"] = 23;
	setupStabilization["enabled"] = true;
}

static void hkpBvCompressedMeshShape_1_to_2(hkDataObject& obj)
{	
	HK_WARN_ALWAYS(0x7bdf87fa, "hkpBvCompressedMeshShape instances previous to version 2 had compression artifacts and must be reexported");
}

void HK_CALL registerPhysicsPatches_2011_3(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_3/hkpPatches_2011_3.cxx>
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
