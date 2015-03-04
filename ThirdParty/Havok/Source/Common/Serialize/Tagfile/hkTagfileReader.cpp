/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Tagfile/hkTagfileReader.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileCommon.h>

hkEnum<hkTagfileReader::FormatType,hkInt32> HK_CALL hkTagfileReader::detectFormat( hkStreamReader* stream )
{
	char buf[16]; // sizeof(hkTagfileHeader)???
	if( stream->peek( buf, sizeof(buf)) != sizeof(buf) )
	{
		return FORMAT_ERROR; // could not read header
	}
	hkUint32* magic = reinterpret_cast<hkUint32*>(buf);
	if( hkBinaryTagfile::isBinaryMagic( magic[0], magic[1] ) )
	{
		return FORMAT_BINARY;
	}
	return FORMAT_UNKNOWN;
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
