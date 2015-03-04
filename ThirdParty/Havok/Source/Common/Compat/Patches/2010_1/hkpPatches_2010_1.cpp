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

//
//	Upgrades either the ball & socket, hinge, limited hinge or ragdoll constraint to use the fast solver by default

static void commonConstraintDataAtomsStabilityUpgrade(hkDataObject& obj)
{
	hkDataObject setupStabilization = obj["setupStabilization"].asObject();
	hkDataObject ballSocket = obj["ballSocket"].asObject();

	// Disable stabilization by default
	setupStabilization["enabled"]	= false;
	setupStabilization["maxAngle"]	= HK_REAL_HIGH;

	//  We must also set the type of the setupStabilization atom
	setupStabilization["type"]	= (hkInt16)23;	// hkpConstraintAtom::TYPE_SETUP_STABILIZATION

	if ( ballSocket.hasMember("solvingMethod") )
	{
		ballSocket["solvingMethod"] = (hkUint8)1;	// hkpConstraintAtom::METHOD_FAST;
	}

	// Save changes
	obj["setupStabilization"] = setupStabilization;
	obj["ballSocket"] = ballSocket;
}

static void hkpDisplayBindingDataRigidBody_1_to_2(hkDataObject& obj)
{
	obj["displayObjectPtr"] = obj["displayObject"].asObject();
}

static void hkpSimpleContactConstraintDataInfo_0_to_1(hkDataObject& obj)
{
	HK_ASSERT2(0xad83433, false, "Patching of hkpSimpleContactConstraintDataInfo not implemented.");
}

void HK_CALL registerPhysicsPatches_2010_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_1/hkpPatches_2010_1.cxx>
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
