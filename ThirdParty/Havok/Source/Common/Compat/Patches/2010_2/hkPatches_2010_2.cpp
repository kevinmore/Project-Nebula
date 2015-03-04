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

static void hkxMaterial_1_to_2(hkDataObject& obj)
{
	obj["extraData"] = obj["old_extraData"].asObject();
}

static void hkMemoryResourceHandle_2_to_3(hkDataObject& obj)
{
	obj["variant"] = obj["old_variant"].asObject();
}

static void hkxMaterialTextureStage_0_to_1(hkDataObject& obj)
{
	obj["texture"] = obj["old_texture"].asObject();
}

static void hkxMeshSection_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_userChannels"].asArray();
	hkDataArray dst = obj["userChannels"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkRootLevelContainerNamedVariant_0_to_1(hkDataObject& obj)
{
	obj["variant"] = obj["old_variant"].asObject();
}

static void hkxAttribute_0_to_1(hkDataObject& obj)
{
	obj["value"] = obj["old_value"].asObject();
}

static void hkxNode_1_to_2(hkDataObject& obj)
{
	obj["object"] = obj["old_object"].asObject();
}

static void hkxAttributeHolder_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_attributeGroups"].asArray();
	hkDataArray dst = obj["attributeGroups"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkIndexedTransformSet_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_indexMappings"].asArray();
	hkDataArray dst = obj["indexMappings"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkClass_0_1(hkDataObject& obj)
{
	hkDataArray src = obj["old_declaredEnums"].asArray();
	hkDataArray dst = obj["declaredEnums"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkHalf8_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["quad"].asArray();
	hkDataArray dst = obj["halfs"].asArray();

	// We have to upgrade all 8 components
	for (int i = 0; i < 8; i++)
	{
		hkHalf h; h.setReal<false>(src[i].asReal());
		dst[i] = h;
	}
}

void HK_CALL registerCommonPatches_2010_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_2/hkPatches_2010_2.cxx>
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
