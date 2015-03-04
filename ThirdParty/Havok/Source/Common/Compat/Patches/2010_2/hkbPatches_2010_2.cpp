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

static void hkbVariableValueSet_0_to_1(hkDataObject& obj)
{
	hkDataArray src = obj["old_variantVariableValues"].asArray();
	hkDataArray dst = obj["variantVariableValues"].asArray();

	const int size = src.getSize();
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		dst[i] = src[i].asObject();
	}
}

static void hkbBehaviorGraphData_2_to_3(hkDataObject& obj)
{
	hkDataArray mins = obj["wordMinVariableValues"].asArray();
	hkDataArray maxs = obj["wordMaxVariableValues"].asArray();
	hkDataArray bounds = obj["variableBounds"].asArray();

	bounds.setSize( mins.getSize() );

	for( int i = 0; i < mins.getSize(); ++i )
	{
		bounds[i].asObject()["min"] = mins[i].asObject();
		bounds[i].asObject()["max"] = maxs[i].asObject();
	}
}

static void hkbCharacterControllerModifier_0_to_1(hkDataObject& obj)
{
	hkDataObject src = obj["old_controlData"].asObject();
	hkDataObject dst = obj["controlData"].asObject();

	dst["verticalGain"] = src["verticalGain"].asReal();
	dst["horizontalCatchUpGain"] = src["horizontalCatchUpGain"].asReal();
	dst["maxVerticalSeparation"] = src["maxVerticalSeparation"].asReal();
	dst["maxHorizontalSeparation"] = src["maxHorizontalSeparation"].asReal();
}

static void hkbCharacterStringData_7_to_8(hkDataObject& obj)
{
	hkDataArray oldSkinFilenames = obj["deformableSkinNames"].asArray();	
	hkDataArray newSkinFilenames = obj["skinNames"].asArray();
	newSkinFilenames.setSize(oldSkinFilenames.getSize());

	for( int i = 0; i < oldSkinFilenames.getSize(); ++i )
	{
		newSkinFilenames[i].asObject()["fileName"] = oldSkinFilenames[i].asString();
		newSkinFilenames[i].asObject()["meshName"] = hkStringPtr("*");
	}

	hkDataArray oldBoneAttachmentFilenames = obj["rigidSkinNames"].asArray();	
	hkDataArray newBoneAttachmentFilenames = obj["boneAttachmentNames"].asArray();
	newBoneAttachmentFilenames.setSize(oldBoneAttachmentFilenames.getSize());

	for( int i = 0; i < oldBoneAttachmentFilenames.getSize(); ++i )
	{
		newBoneAttachmentFilenames[i].asObject()["fileName"] = oldBoneAttachmentFilenames[i].asString();
		newBoneAttachmentFilenames[i].asObject()["meshName"] = hkStringPtr("*");
	}
}

void HK_CALL registerBehaviorPatches_2010_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_2/hkbPatches_2010_2.cxx>
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
