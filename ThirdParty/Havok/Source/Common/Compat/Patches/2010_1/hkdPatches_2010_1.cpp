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

static void hkdRandomSplitFracture_0_to_1(hkDataObject& obj)
{
	// member init to behave like version 0
	obj["randomSeed1"] = obj["randomSeed"].asInt();
	obj["randomSeed2"] = 0;
	obj["randomSeed3"] = 0;
	obj["randomSeed4"] = 0;
	obj["randomRange"] = 0.5f;
}

static void hkdShapeInstanceInfo_3_to_4(hkDataObject& obj)
{
	const hkQuaternion& q = obj["rotation"].asQuaternion();
	const hkVector4& t = obj["translation"].asVector4();

	const hkDataWorld* world = obj.getClass().getWorld();
	hkDataClass qtDataClass( world->findClass("hkQTransform") );
	hkDataObject qtObj = world->newObject( qtDataClass );

	qtObj["translation"] = t;
	qtObj["rotation"] = q;
	obj["transform"] = qtObj;
}

void HK_CALL registerDestructionPatches_2010_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_1/hkdPatches_2010_1.cxx>
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
