/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>

static int SerializeUtilTest()
{
#if 0
	// hkHalf8 is deprecated
	{
		hkHalf8 h0;
		{
			hkVector4 top; top.set(1,2,3,4);
			hkVector4 bot; bot.set(5,6,7,8);
			h0.packFirst<true>(top); h0.packSecond<true>(bot);
		}

		hkArray<char> buf;
		hkSerializeUtil::save( &h0, hkHalf8Class, hkOstream(buf).getStreamWriter() );

		hkHalf8* h1 = hkSerializeUtil::loadObject<hkHalf8>(buf.begin(), buf.getSize());
		for( int i = 0; i < 8; ++i )
		{
			HK_TEST( h0.getComponent(i) == h1->getComponent(i) );
		}
		delete h1;
	}
#endif
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER( SerializeUtilTest, "Fast", "Common/Test/UnitTest/Serialize/", "SerializeUtilTest" );

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
