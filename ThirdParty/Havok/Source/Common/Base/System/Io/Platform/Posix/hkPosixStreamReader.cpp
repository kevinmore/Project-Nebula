/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/System/Io/Platform/Posix/hkPosixStreamReader.h>

#if defined(HK_PLATFORM_LINUX)  || defined(HK_PLATFORM_MACPPC) || defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_LRB) || defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_TIZEN) || defined(HK_PLATFORM_PS4)
#	include <fcntl.h>
#	include <unistd.h>
#	include <sys/stat.h>
#ifndef HK_PLATFORM_PS4
#	include <dirent.h>
#endif
#	define HK_OPEN(_fname) ::open(_fname, O_RDONLY, 0666)
#	define HK_CLOSE(_handle) ::close(_handle)
#	define HK_READ(_handle, _buf, _nbytes) ::read( _handle, _buf, _nbytes )
#	define HK_SEEK(_handle, _offset, _whence) ::lseek(_handle, _offset, _whence)
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
#	define HK_OPEN(_fname) sceIoOpen(_fname, SCE_O_RDONLY, 0)
#	define HK_CLOSE(_handle) sceIoClose(_handle)
#	define HK_READ(_handle, _buf, _nbytes) sceIoRead( _handle, _buf, _nbytes )
#	define HK_SEEK(_handle, _offset, _whence) sceIoLseek32(_handle, _offset, _whence)
HK_COMPILE_TIME_ASSERT(hkSeekableStreamReader::STREAM_SET==SCE_SEEK_SET
					   && hkSeekableStreamReader::STREAM_CUR==SCE_SEEK_CUR
					   && hkSeekableStreamReader::STREAM_END==SCE_SEEK_END );
#else
#	error unknown platform
#endif

hkPosixStreamReader* hkPosixStreamReader::open( const char* name )
{
	int handle = HK_OPEN( name );
	if( handle >= 0 )
	{
		return new hkPosixStreamReader(handle);
	}
	return HK_NULL;
}

hkPosixStreamReader::hkPosixStreamReader( int handle )
	: m_handle(handle), m_isOk(handle >= 0)
{
}

void hkPosixStreamReader::close()
{
	if(m_handle >= 0 )
	{
		HK_CLOSE(m_handle);
		m_handle = -1;
		m_isOk = false;
	}
}

hkPosixStreamReader::~hkPosixStreamReader()
{
	close();
}

int hkPosixStreamReader::read( void* buf, int nbytes)
{
	if( m_handle >= 0 && nbytes )
	{
		int nread = HK_READ( m_handle, buf, nbytes );
		if( nread <= 0 ) 
		{
			m_isOk = false;
		}
		return nread;
	}
	return 0; 
}

int hkPosixStreamReader::peek( void* buf, int nbytes)
{
	int nread = HK_READ( m_handle, buf, nbytes );
	if( nread <= 0 || HK_SEEK(m_handle, -nread, STREAM_CUR)<0 )
	{
		m_isOk = false;
	}
	return nread;
}


hkBool hkPosixStreamReader::isOk() const
{
	return m_isOk;
}

hkResult hkPosixStreamReader::seek( int offset, SeekWhence whence)
{
	if(HK_SEEK(m_handle, offset, whence) >= 0)
	{
		m_isOk = true;
		return HK_SUCCESS;
	}
	m_isOk = false;
	return HK_FAILURE;
}

int hkPosixStreamReader::tell() const
{
	return HK_SEEK(m_handle, 0, STREAM_CUR);
}

#undef HK_OPEN
#undef HK_CLOSE
#undef HK_READ
#undef HK_SEEK

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
