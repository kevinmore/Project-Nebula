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

static void hkpCompressedMeshShape_8_to_9(hkDataObject& obj)
{
	hkDataArray transforms = obj["transforms"].asArray();

	if( obj.hasMember("transforms") && transforms.getSize()!=0 )
	{
		hkDataArray newtransforms (obj["newtransforms"].asArray().getImplementation());
		newtransforms.setSize( transforms.getSize() );

		for ( int i = 0; i < transforms.getSize(); ++i )
		{
			const hkTransform& tr = transforms[i].asTransform();
			hkQsTransform qs; qs.setFromTransform(tr);
			newtransforms[i] = qs;
		}
	}	
}

static void hkpExtendedMeshShape_2_to_3(hkDataObject& obj)
{
	hkVector4 scale = obj["scaling"].asVector4();
	hkDataArray triSubparts = obj["trianglesSubparts"].asArray();
	for ( int i = 0; i < triSubparts.getSize(); ++i )
	{
		hkDataObject triSub = triSubparts[i].asObject();
		hkQsTransform transform = triSub["transform"].asQsTransform();
		transform.setScale( scale );
		triSub["transform"] = transform;
		triSubparts[i] = triSub;
	}
}

void HK_CALL registerPhysicsPatches_710(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/710/hkpPatches_710.cxx>
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
