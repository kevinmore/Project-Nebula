/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#ifndef HK_BASE_GAMECUBEDVDSTREAMREADER_H
#define HK_BASE_GAMECUBEDVDSTREAMREADER_H

#include <Common/Base/System/Io/Reader/hkStreamReader.h>

#include <dolphin.h>
#include <dolphin/dvd.h>

struct DVDFileInfo;

/// A stream through DVD interface for Wii
class hkWiiDvdStreamReader : public hkSeekableStreamReader
{
public:

	hkWiiDvdStreamReader(const char* fname);
	virtual ~hkWiiDvdStreamReader();
	virtual hkBool isOk() const;
	virtual int read(void* buf, int nbytes);
	virtual hkResult seek( int offset, SeekWhence whence );
	virtual int tell() const;

protected:

	DVDFileInfo m_dvdInfo;
	int m_fileOffset;
	int m_fileLen;
};

#endif //HK_BASE_GAMECUBEDVDSTREAMREADER_H

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
