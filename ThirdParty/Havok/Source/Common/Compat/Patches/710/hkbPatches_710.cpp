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

static void hkbStateMachine_2_to_3(hkDataObject& obj)
{
	// we removed the value 1 from the enum, so all but zero need to be decremented
	int startStateMode = obj["startStateMode"].asInt();

	if ( startStateMode > 0 )
	{
		startStateMode--;
	}

	obj["startStateMode"] = startStateMode;
}

static void hkbSceneModifierList_0_to_1(hkDataObject& obj)
{
	// Inserted two new scene modifiers after the first scene modifiers so all but zero needs to be incremented by 2.
	hkDataArray sceneModifierEntries = obj["sceneModifierEntries"].asArray();

	const int size = sceneModifierEntries.getSize();
	for ( int i = 0; i < size; ++i )
	{
		hkDataObject sceneModifierEntry = sceneModifierEntries[i].asObject();

		int sceneModifier = sceneModifierEntry["sceneModifier"].asInt();

		if ( sceneModifier > 0 )
		{
			sceneModifier += 2;
		}

		sceneModifierEntry["sceneModifier"] = sceneModifier;
	}
}

static void hkbBehaviorGraph_0_to_1(hkDataObject& obj)
{
	// removed VARIABLE_MODE_MAINTAIN_MEMORY_WHEN_INACTIVE at index 1
	if ( obj["variableMode"].asInt() > 0 )
	{
		obj["variableMode"] = obj["variableMode"].asInt() - 1;
	}
}

void HK_CALL registerBehaviorPatches_710(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/710/hkbPatches_710.cxx>
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
