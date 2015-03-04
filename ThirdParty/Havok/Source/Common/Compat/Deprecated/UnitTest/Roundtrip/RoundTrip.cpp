/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/Types/hkIgnoreDeprecated.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Compat/Deprecated/Packfile/Binary/hkBinaryPackfileReader.h>
#include <Common/Serialize/Packfile/Binary/hkBinaryPackfileWriter.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileReader.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileWriter.h>
#include <Common/Compat/Deprecated/UnitTest/Roundtrip/RoundTrip.h>
#include <Common/Serialize/UnitTest/serializeUtilities.h>

static hkResult compareRoundTrip(const hkRoundTrip& orig, const hkRoundTrip& copy)
{
	return (hkString::memCmp(&orig, &copy, sizeof(hkRoundTrip)) == 0)? HK_SUCCESS : HK_FAILURE;
}

extern const hkTypeInfo hkRoundTripTypeInfo;
int RoundTrip()
{
	typedef hkRoundTrip Object;

	char origBuf[sizeof(Object)];
	hkString::memSet( origBuf, 0, sizeof(origBuf));
	Object* orig = new (origBuf) Object();

	hkError::getInstance().setEnabled(0x718f5bfe, false); // Propagating bits warning disabled
	hkError::getInstance().setEnabled(0x1b912aea, false); // Unreflected bits warning disabled
	hkError::getInstance().setEnabled(0x555b54ac, false); // Unreflected bits warning disabled

	serializeTest< hkBinaryPackfileReader, hkBinaryPackfileWriter, Object >(*orig, hkRoundTripClass, hkRoundTripTypeInfo, &compareRoundTrip);
	serializeTest< hkXmlPackfileReader, hkXmlPackfileWriter, Object >(*orig, hkRoundTripClass, hkRoundTripTypeInfo, &compareRoundTrip);

	hkError::getInstance().setEnabled(0x718f5bfe, true);
	hkError::getInstance().setEnabled(0x1b912aea, true);
	hkError::getInstance().setEnabled(0x555b54ac, true); 

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(RoundTrip, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
