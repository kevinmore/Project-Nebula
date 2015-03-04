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

static void hkaBoneAttachment_1_to_2(hkDataObject& obj)
{
	obj["attachment"] = obj["old_attachment"];
}

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

static void hkaMeshBinding_2_to_3(hkDataObject& obj)
{
	_copyArray(obj, "old_mappings", "mappings");
}

static void hkaAnimation_1_to_2(hkDataObject& obj)
{	
	hkInt32 animationType = obj["type"].asInt();

	HK_ASSERT2( 0x4561b83b, animationType != 2, "Delta compressed animations have been removed and can no longer be loaded" );
	HK_ASSERT2( 0x4561b83c, animationType != 3, "Wavelet compressed animations have been removed and can no longer be loaded" );

	if ( animationType > 3 )
	{
		animationType -= 2;
	}

	obj["type"] = animationType;
}

static void hkaAnimation_2_to_3(hkDataObject& obj)
{
	_copyArray(obj, "old_annotationTracks", "annotationTracks");	
}

static void hkaPredictiveCompressedAnimation_0_to_1(hkDataObject& obj)
{
	hkDataArray old_intArrayOffsets = obj["old_intArrayOffsets"].asArray();
	hkDataArray intArrayOffsets = obj["intArrayOffsets"].asArray();
	intArrayOffsets[0] = old_intArrayOffsets[0].asInt();
	intArrayOffsets[1] = 0;
	intArrayOffsets[2] = old_intArrayOffsets[1].asInt();
	intArrayOffsets[3] = old_intArrayOffsets[2].asInt();
	intArrayOffsets[4] = old_intArrayOffsets[3].asInt();
	intArrayOffsets[5] = old_intArrayOffsets[4].asInt();
	intArrayOffsets[6] = old_intArrayOffsets[5].asInt();
	intArrayOffsets[7] = old_intArrayOffsets[6].asInt();
	intArrayOffsets[8] = old_intArrayOffsets[7].asInt();

	// this tells the system that LOD is not supported
	obj["firstFloatBlockScaleAndOffsetIndex"] = -1;
}

void HK_CALL registerAnimationPatches_2010_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_2/hkaPatches_2010_2.cxx>
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
