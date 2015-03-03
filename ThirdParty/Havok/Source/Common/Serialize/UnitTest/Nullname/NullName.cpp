/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileReader.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileWriter.h>
#include <Common/Serialize/UnitTest/Nullname/NullName.h>
#include <Common/Serialize/UnitTest/serializeUtilities.h>

extern const hkClass hkNullNameClass;
extern const hkTypeInfo hkNullNameTypeInfo;

static hkResult compareNullName(const hkNullName& objA, const hkNullName& objB)
{
	hkBool res = ( (hkString::strCmp(objA.m_c1, objB.m_c1)==0) &&
				 (hkString::strCmp(objA.m_c2, objB.m_c2)==0) &&
				 ( objA.m_c3 == objB.m_c3 )
				 );

	return (res)? HK_SUCCESS : HK_FAILURE;
}

int NullNameTest()
{
	hkNullName nullname;

	hkBuiltinTypeRegistry::getInstance().addType( &hkNullNameTypeInfo, &hkNullNameClass );

	serializeTest<hkXmlPackfileReader, hkXmlPackfileWriter, hkNullName>(nullname, hkNullNameClass, hkNullNameTypeInfo, &compareNullName);
	tagfileTest<hkNullName>(nullname, hkNullNameClass, hkNullNameTypeInfo, &compareNullName);

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
//HK_TEST_REGISTER(NullNameTest, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
