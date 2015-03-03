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

#if defined(HK_FEATURE_PRODUCT_PHYSICS_2012)

static void hkbCharacterData_9_to_10(hkDataObject& obj)
{
	const hkDataObject oldCharacterControllerInfo = obj["characterControllerInfo"].asObject();
	hkDataObject newCharacterControllerSetup = obj["characterControllerSetup"].asObject();

	// Setup the controller
	{
		hkDataObject newRigidBodySetup = newCharacterControllerSetup["rigidBodySetup"].asObject();
		newRigidBodySetup["collisionFilterInfo"] = oldCharacterControllerInfo["collisionFilterInfo"].asInt();
		newRigidBodySetup["type"] = 0; // ignored anyway

		hkDataObject newShapeSetup = newRigidBodySetup["shapeSetup"].asObject();
		newShapeSetup["capsuleHeight"] = oldCharacterControllerInfo["capsuleHeight"].asReal();
		newShapeSetup["capsuleRadius"] = oldCharacterControllerInfo["capsuleRadius"].asReal();
		newShapeSetup["type"] = 0;

		hkDataObject characterControllerCinfo = oldCharacterControllerInfo["characterControllerCinfo"].asObject();
		newCharacterControllerSetup["controllerCinfo"] = characterControllerCinfo;
	}
}

static void hkbRigidBodyRagdollControlData_1_to_2(hkDataObject& obj)
{
	const hkDataObject oldControlData = obj["keyFrameHierarchyControlData"].asObject();
	hkDataObject newControlData = obj["keyFrameControlData"].asObject();

	for(hkDataObject::Iterator oldIter = oldControlData.getMemberIterator();
		oldControlData.isValid(oldIter);
		oldIter = oldControlData.getNextMember(oldIter) )
	{
		const char* oldMemberName = oldControlData.getMemberName(oldIter);

		HK_ASSERT(0x22440107, newControlData.hasMember(oldMemberName));

		if (newControlData.hasMember(oldMemberName))
		{
			HK_ASSERT(0x22440108, newControlData[oldMemberName].getType()->isReal() && oldControlData[oldMemberName].getType()->isReal());

			newControlData[oldMemberName] = oldControlData[oldMemberName].asReal();
		}
	}
}

#endif

void HK_CALL registerBehaviorPatches_2012_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2012_2/hkbPatches_2012_2.cxx>
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
