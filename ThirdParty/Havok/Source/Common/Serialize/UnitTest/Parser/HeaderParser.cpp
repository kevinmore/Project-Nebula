/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/UnitTest/Parser/HeaderParser.h>
#include <Common/Base/Types/hkTypedUnion.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

int headerParserTest()
{
	extern const hkClass HeaderParserClass;
	int memIdx = HeaderParserClass.getMemberIndexByName("name");
	if( memIdx != -1 )
	{
		hkTypedUnion def;
		HeaderParserClass.getDefault(memIdx, def);
		const char* foo = (char*)def.getStorage().m_ulong;
		++foo;
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(headerParserTest, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
