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

static void hkdShape_1_to_2(hkDataObject& obj)
{
	// HKD-97: Type QUALITY_FIXED has been inserted in 2nd position in the ENUM field.
	//         Therefore all values from 2nd position and up need to be shifted by one.
	int bodyQualityType = obj["bodyQualityType"].asInt();
	if ( bodyQualityType >= 1 )
	{
		bodyQualityType++;
	}
	obj["bodyQualityType"] = bodyQualityType;
}

static void hkdWoodFracture_0_to_1(hkDataObject& obj)
{
	obj["flattenHierarchy"] = obj["old_flattenHierarchy"].asInt();
}

static void hkdSplitInHalfFracture_0_to_1(hkDataObject& obj)
{
	obj["flattenHierarchy"] = obj["old_flattenHierarchy"].asInt();
}

static void hkdSliceFracture_0_to_1(hkDataObject& obj)
{
	obj["childFracture"] = obj["old_childFracture"].asObject();
}


static void hkdDeformableBreakableShape_addChildKeys(hkDataObject& obj)
{
	// HKD-276 now stores the actual shape keys and does not assume childIdx == shapeKey anymore
	const int size = obj["origChildTransforms"].asArray().getSize();
	hkDataArray keyArray = obj["childKeys"].asArray();
	keyArray.setSize(size);

	// as the former versions required a matching childIdx/shapeKey, we can assume this asset did work ok :)
	for (int i=0; i<size; ++i)
	{
		keyArray[i] = i;
	}
}


static void hkdBreakableShapeConnection_0_to_1(hkDataObject& obj)
{
	hkDataObject d(HK_NULL);
	obj["contactAreaDetails"] = d;
}

void HK_CALL registerDestructionPatches_Legacy(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/Legacy/hkdPatches_Legacy.cxx>
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
