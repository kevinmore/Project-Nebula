/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileHeader.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>

hkResult hkPackfileHeader::readHeader(hkStreamReader* stream, hkPackfileHeader& out)
{
	if(stream->read(&out, sizeof(hkPackfileHeader)) < static_cast<int>(sizeof(hkPackfileHeader)))
	{
		return HK_FAILURE;
	}

	hkPackfileHeader validHeader;
	if(out.m_magic[0] != validHeader.m_magic[0] || out.m_magic[1] != validHeader.m_magic[1])
	{
		return HK_FAILURE;
	}

	return HK_SUCCESS;
}

const hkPackfileSectionHeader* hkPackfileHeader::getSectionHeader(const void* packfileData, int i) const
{
	const hkPackfileHeader* packfileHeader = static_cast<const hkPackfileHeader*>(packfileData);

	if(packfileHeader->m_numSections == 0)
	{
		return HK_NULL;
	}

	int bytesToSkip;
	if(packfileHeader->m_fileVersion <= 9)
	{
		bytesToSkip = 0;
	}
	else if(packfileHeader->m_fileVersion <= 11)
	{
		bytesToSkip = 0; 
	}
	else
	{
		HK_ASSERT2(0x26b238fd, false, "Invalid handling of new packfile version, or packfile data corrupted.");
		bytesToSkip = 0;
	}

	// The size of the hkPackfileSectionHeader changed in version 11 of the binary packfile format.
	// While now it is 64 bytes, before it was 48. We need to be able to read both formats.
	// Because the changing fields are just padding at the end of the header, we can simply reinterpret cast the
	// memory.
	int sectionHeaderSize;
	if(packfileHeader->m_fileVersion <= 10)
	{
		sectionHeaderSize = 12*sizeof(hkInt32);
	}
	else if(packfileHeader->m_fileVersion == 11)
	{
		sectionHeaderSize = 16*sizeof(hkInt32);
	}
	else
	{
		HK_ASSERT2(0x2991e55f, false, "Invalid handling of new packfile version, or packfile data corrupted.");
		sectionHeaderSize = 0;
	}

	const hkPackfileSectionHeader* first = static_cast<const hkPackfileSectionHeader*>(
		hkAddByteOffset(packfileData, sizeof(hkPackfileHeader) + bytesToSkip) );

	return static_cast<const hkPackfileSectionHeader*>(
		hkAddByteOffset(first, sectionHeaderSize*i) );
}

hkPackfileSectionHeader* hkPackfileHeader::getSectionHeader(void* packfileData, int i) const
{
	return const_cast<hkPackfileSectionHeader*>(getSectionHeader(const_cast<const void*>(packfileData), i));
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
