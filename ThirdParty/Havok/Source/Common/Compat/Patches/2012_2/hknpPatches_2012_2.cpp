/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/Reflection/hkClassMember.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/KeyCode.h>

// Registration function is at the end of the file

#define HK_INITIAL_ENTRY 0x80000000

static void notimplemented(hkDataObject& obj)
{
	HK_ASSERT(0x548932c2, 0);
}

static void hknpShape_0_to_1(hkDataObject& obj)
{
	// Size of m_flags increased from 8 to 16 bits
	hkUint8 oldFlags = static_cast<hkUint8> (obj["old_flags"].asInt());
	obj["flags"] = static_cast<hkUint16> (oldFlags);
}

static void hknpPhysicsSceneData_0_to_1(hkDataObject& obj)
{
	// Array of SystemInstance structures changed to array of hknpPhysicsSystemData ptrs

	hkDataArray systemInstances = obj["systemInstances"].asArray();
	const int numSystemInstances = systemInstances.getSize();

	hkDataArray systemDatas = obj["systemDatas"].asArray();
	systemDatas.setSize( numSystemInstances );

	for (int i=0; i<numSystemInstances; ++i)
	{
		hkDataObject systemInstance = systemInstances[i].asObject();
		systemDatas[i] = systemInstance["systemData"].asObject();
	}
}


void HK_CALL registerNewPhysicsPatches_2012_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2012_2/hknpPatches_2012_2.cxx>
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
