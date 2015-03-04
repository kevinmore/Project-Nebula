/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/Posix/hkPosixStreamWriter.h>

#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/System/Io/Platform/Posix/hkPosixStreamReader.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#if defined(HK_PLATFORM_LINUX) || defined(HK_PLATFORM_MACPPC) || defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_LRB) || defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_TIZEN) || defined(HK_PLATFORM_PS4)
#	include <fcntl.h>
#	include <unistd.h>
#	include <sys/stat.h>
#ifndef HK_PLATFORM_PS4
#	include <dirent.h>
#endif
#	define HK_OPEN(_fname, flags, mode) ::open(_fname, flags, mode)
#	define HK_CLOSE(_handle) ::close(_handle)
#	define HK_WRITE(_handle, _buf, _nbytes) ::write( _handle, _buf, _nbytes )
#	define HK_SEEK(_handle, _offset, _whence) ::lseek(_handle, _offset, _whence)
#	define HK_FSYNC(_handle) ::fsync(_handle)
// check that we can use these interchangeably
HK_COMPILE_TIME_ASSERT(hkSeekableStreamReader::STREAM_SET==SEEK_SET
					   && hkSeekableStreamReader::STREAM_CUR==SEEK_CUR
					   && hkSeekableStreamReader::STREAM_END==SEEK_END );
#elif defined(HK_PLATFORM_PSP) || defined(HK_PLATFORM_PSVITA)
#if defined(HK_PLATFORM_PSVITA)
#	include <kernel/iofilemgr.h>
#else
#	include <iofilemgr.h>
#endif
#	define HK_OPEN(_fname, flags, mode) sceIoOpen(_fname,  flags, SCE_STM_RWU)
#	define HK_CLOSE(_handle) sceIoClose(_handle)
#	define HK_WRITE(_handle, _buf, _nbytes) sceIoWrite( _handle, _buf, _nbytes )
#	define HK_SEEK(_handle, _offset, _whence) sceIoLseek(_handle, _offset, _whence)
#	define HK_FSYNC(_handle) // nothing
HK_COMPILE_TIME_ASSERT(hkSeekableStreamReader::STREAM_SET==SCE_SEEK_SET
					   && hkSeekableStreamReader::STREAM_CUR==SCE_SEEK_CUR
					   && hkSeekableStreamReader::STREAM_END==SCE_SEEK_END );
#else
#	error unknown platform
#endif

hkPosixStreamWriter* hkPosixStreamWriter::open(const char* name, int flags, int mode)
{
	int handle = HK_OPEN(name, flags, mode);
	if( handle >= 0 )
	{
		return new hkPosixStreamWriter(handle, true);
	}
	return HK_NULL;
}

hkPosixStreamWriter::hkPosixStreamWriter(int handle, hkBool shouldClose)
:	m_handle(handle), m_shouldClose(shouldClose)
{
}

void hkPosixStreamWriter::close()
{
	if(m_handle >= 0 && m_shouldClose)
	{
		HK_CLOSE(m_handle);
		m_handle = -1;
	}
}

hkPosixStreamWriter::~hkPosixStreamWriter()
{
	close();
}

int hkPosixStreamWriter::write( const void* buf, int nbytes)
{
	if( m_handle >= 0 )
	{
		int n = HK_WRITE( m_handle, buf, nbytes );
		if( n <= 0 )
		{
			close();
		}
		return n;
	}
	return 0;
}

void hkPosixStreamWriter::flush()
{
#ifndef HK_PLATFORM_NACL
	if( m_handle >= 0 )
	{
		HK_FSYNC(m_handle);
	}
#endif
}

hkBool hkPosixStreamWriter::isOk() const
{
	return m_handle >= 0;
}

hkBool hkPosixStreamWriter::seekTellSupported() const
{
	return true;
}

hkResult hkPosixStreamWriter::seek( int offset, SeekWhence whence)
{
	return HK_SEEK(m_handle, offset, whence) != -1
		? HK_SUCCESS
		: HK_FAILURE;
}

int hkPosixStreamWriter::tell() const
{
	return HK_SEEK(m_handle, 0, STREAM_CUR);
}

#undef HK_OPEN
#undef HK_CLOSE
#undef HK_WRITE
#undef HK_SEEK
#undef HK_FSYNC

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
