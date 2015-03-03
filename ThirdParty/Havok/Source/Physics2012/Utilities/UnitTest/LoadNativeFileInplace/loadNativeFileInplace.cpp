/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Serialize/Util/hkNativePackfileUtils.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Physics2012/Utilities/Serialize/hkpPhysicsData.h>
#include <Physics2012/Utilities/Serialize/hkpHavokSnapshot.h>

static int loadNativeFileInplace()
{
	// Disabled 
	hkError::getInstance().setEnabled( 0x651F7AA5, false ); // removing deprecated object
	hkError::getInstance().setEnabled( 0x9FE65234, false ); //Unsupported simulation type, setting to SIMULATION_TYPE_CONTINUOUS

	// Load asset as an hkResource and save it in native binary packfile format in buf
	hkArray<char> buf;
	{				
		// Load XML tagfile
		hkResource* res = hkSerializeUtil::load("Resources/Common/Api/Serialize/SimpleLoad/simpleTagfile.xml");
		HK_ASSERT( 0x215d080c, res != HK_NULL );

		// Get the top level object in the file
		hkRootLevelContainer* container = res->getContents<hkRootLevelContainer>();
		HK_ASSERT2(0xa6451543, container != HK_NULL, "Could not load root level obejct" );

		// Save resource in the buffer in native binary packfile format
		hkpHavokSnapshot::ConvertListener convertListener;
		HK_ON_DEBUG( hkResult result = )hkSerializeUtil::savePackfile(container, hkRootLevelContainerClass,
			hkOstream(buf).getStreamWriter(),
			hkPackfileWriter::Options(),
			&convertListener,
			hkSerializeUtil::SaveOptions());
		HK_ASSERT(0x45786055, result == HK_SUCCESS );
		
		res->removeReference();
	}

	int bufferSize = hkNativePackfileUtils::getRequiredBufferSize( buf.begin(), buf.getSize() );
	hkArray<char> dataBuffer;
	dataBuffer.reserveExactly(bufferSize);
	HK_ON_DEBUG( hkRootLevelContainer* container = static_cast<hkRootLevelContainer*>)
		(hkNativePackfileUtils::load(buf.begin(), buf.getSize(), dataBuffer.begin(), dataBuffer.getCapacity()));
	HK_ASSERT2(0xa6451543, container != HK_NULL, "Could not load root level obejct" );
	HK_ON_DEBUG(hkpPhysicsData* physicsData = static_cast<hkpPhysicsData*>( container->findObjectByType( hkpPhysicsDataClass.getName() ) ));
	HK_ASSERT2(0xa6451544, physicsData != HK_NULL, "Could not find physics data in root level object" );

	hkNativePackfileUtils::unload(dataBuffer.begin(), dataBuffer.getCapacity());
	hkError::getInstance().setEnabled( 0x651F7AA5, true );
	hkError::getInstance().setEnabled( 0x9FE65234, true );

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(loadNativeFileInplace, "Native", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
