/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>
#include <Common/Serialize/Resource/hkObjectResource.h>
#include <Common/Serialize/ResourceDatabase/hkResourceHandle.h>

int dataObjectUtilTest()
{
	// basic test should work
	{
		hkDataWorldNative world;
		hkMemoryResourceHandle handle;
		world.setContents(&handle, hkMemoryResourceHandleClass);
		hkObjectResource* res = hkDataObjectUtil::toObject( world.getContents(), false );
		HK_TEST(res != HK_NULL);
		delete res;
	}
	// chase a pointer
	{
		hkDataWorldNative world;
		hkMemoryResourceHandle handle1;
		hkMemoryResourceHandle handle2;
		handle1.setObject( &handle2, &hkMemoryResourceHandleClass );
		world.setContents( &handle1, hkMemoryResourceHandleClass);
		hkObjectResource* res = hkDataObjectUtil::toObject( world.getContents(), false );
		HK_TEST(res != HK_NULL);
		delete res;
	}
	// should detect circular ref, return null
	{
		hkDataWorldNative world;
		hkMemoryResourceHandle handle1;
		hkMemoryResourceHandle handle2;
		handle1.setObject( &handle2, &hkMemoryResourceHandleClass );
		handle2.setObject( &handle1, &hkMemoryResourceHandleClass );
		world.setContents(&handle1, hkMemoryResourceHandleClass);
		hkError::getInstance().setEnabled(0x19CA1F60, false);
		hkObjectResource* res = hkDataObjectUtil::toObject( world.getContents(), false );
		hkError::getInstance().setEnabled(0x19CA1F60, true);
		HK_TEST(res == HK_NULL);
	}
	// circular refs are ok if they are broken by a +owned(false)
	{
		hkDataWorldNative world;

		// Mark flags as zero so there are no reference counting issues when destroyed 
		hkSimpleLocalFrame frame1;
		frame1.m_memSizeAndFlags = 0;
		hkSimpleLocalFrame frame2;
		frame2.m_memSizeAndFlags = 0;
		hkSimpleLocalFrame frame3;
		frame3.m_memSizeAndFlags = 0;

		frame1.m_children.pushBack(&frame2);
		frame1.m_children.pushBack(&frame3);

		frame2.setParentFrame(&frame2); // self ref should be ok too
		frame3.setParentFrame(&frame1); // ptr to parent

		world.setContents(&frame1, hkSimpleLocalFrameClass);
		hkObjectResource* res = hkDataObjectUtil::toObject( world.getContents(), false );
		HK_TEST(res != HK_NULL);
		delete res;
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(dataObjectUtilTest, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
