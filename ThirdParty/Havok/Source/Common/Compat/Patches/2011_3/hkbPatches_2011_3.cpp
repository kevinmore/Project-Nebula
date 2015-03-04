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


static void hkbCharacterStringData_8_to_9(hkDataObject& obj)
{
	hkDataArray oldAnimationNames = obj["animationNames"].asArray();
	hkDataArray oldAnimationFilenames = obj["animationFilenames"].asArray();

	if ( oldAnimationNames.getSize() == 0 )
	{
		// empty string datas shouldn't create any bundle info
		return;
	}
	
	hkDataArray newDefaultAnimationBundleNameData = obj["animationBundleNameData"].asArray();
	newDefaultAnimationBundleNameData.setSize( 1 );
	hkDataArray newDefaultAnimationBundleFilenameData = obj["animationBundleFilenameData"].asArray();
	newDefaultAnimationBundleFilenameData.setSize( 1 );
	hkDataArray newDefaultAnimationNames = newDefaultAnimationBundleNameData[0].asObject()["assetNames"].asArray();
	newDefaultAnimationNames.setSize( oldAnimationNames.getSize() );
	hkDataArray newDefaultAnimationFilenames = newDefaultAnimationBundleFilenameData[0].asObject()["assetNames"].asArray();
	newDefaultAnimationFilenames.setSize( oldAnimationFilenames.getSize() );

	HK_ASSERT2(0x1e5af05a, oldAnimationNames.getSize() >= oldAnimationFilenames.getSize(), "Old assets should have at least as many animation names as filenames.");

	bool characterPropertyUsed = false;
	for( int i = 0; i < oldAnimationNames.getSize(); i++ )
	{
		newDefaultAnimationNames[i] = oldAnimationNames[i].asString();
		// If any of the animation names are overridden by a filename then it means
		// we're using character properties (as otherwise the animation name would be the filename)
		if ( i < oldAnimationFilenames.getSize() && oldAnimationFilenames[i].asString() != HK_NULL )
		{
			newDefaultAnimationFilenames[i] = oldAnimationFilenames[i].asString();
			characterPropertyUsed = true;
		}
	}

	if ( !characterPropertyUsed )
	{
		// Needed to setSize to zero explicitly rather than calling clear
		// for proper ArrayArrayImplementation cleanup (see COM-1893)
		newDefaultAnimationBundleFilenameData.setSize(0);
	}
}

// Registration function is at the end of the file

void HK_CALL registerBehaviorPatches_2011_3(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_3/hkbPatches_2011_3.cxx>
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
