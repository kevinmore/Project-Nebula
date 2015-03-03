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

static void _copyArray(hkDataObject& obj, const char* srcName, const char* dstName)
{
	hkDataArray src = obj[srcName].asArray();
	hkDataArray dst = obj[dstName].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkdDeformableBreakableShapePhysicsSkinShape_0_to_1(hkDataObject& obj)
{
	_copyArray(obj, "old_sections", "sections");
}

static void hkdDecorateFractureFaceActionShapeDecorationInfo_1_to_2(hkDataObject& obj)
{
	//_copyArray(obj, "old_sharedData", "sharedData");
	_copyArray(obj, "old_connectionDecorations", "connectionDecorations");
}

static void hkdDecalMapActionDecalMapData_0_to_1(hkDataObject& obj)
{
	_copyArray(obj, "old_decals", "decals");
	_copyArray(obj, "old_decors", "decors");
	_copyArray(obj, "old_decorInfo", "decorInfo");
}

static void hkdBreakableBodyShapeKeyToChildMap_0_to_1(hkDataObject& obj)
{
	_copyArray(obj, "old_shapeKeyToChild", "shapeKeyToChild");
}

void HK_CALL registerDestructionPatches_2010_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_2/hkdPatches_2010_2.cxx>
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
