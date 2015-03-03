/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/String/hkString.h>
#include <Common/Base/System/Io/Platform/Stdio/hkStdioStreamReader.h>
#include <Common/Base/Container/String/hkUtf8.h>
#include <Common/Base/Fwd/hkcstdio.h>

using namespace std;

hkStdioStreamReader* hkStdioStreamReader::open( const char* nameIn )
{
	#if !defined HK_PLATFORM_WIN32 // doesn't support fopen with utf8 names
		FILE* handle = fopen(nameIn, "rb");
	#else
		FILE* handle = _wfopen( hkUtf8::WideFromUtf8(nameIn).cString(), L"rb" );
	#endif
	if(handle)
	{
		return new hkStdioStreamReader(handle);
	}
	return HK_NULL;
}

hkStdioStreamReader::hkStdioStreamReader( void* handle)
	:	m_handle(handle), m_isOk(handle!=HK_NULL)
{
}

hkStdioStreamReader::~hkStdioStreamReader()
{
	if(m_handle != HK_NULL )
	{
		fclose( (FILE*)m_handle);
	}
}

int hkStdioStreamReader::read( void* buf, int nbytes)
{
	HK_ASSERT2(0x6400412c, m_handle != HK_NULL, "Read from closed file" );
	int nread = static_cast<int>( fread( buf, 1, nbytes, (FILE*)m_handle ) );
	if(nread <= 0)
	{
		m_isOk = false;
	}
	return nread;
}

hkBool hkStdioStreamReader::isOk() const
{
	return m_isOk;
}

int hkStdioStreamReader::peek( void* buf, int nbytes)
{
	FILE* handle = (FILE*)m_handle;
	long nread = static_cast<long>(fread( buf, 1, nbytes, handle ));
	if( (fseek(handle, -nread, SEEK_CUR) < 0) || nread==0 )
	{
		m_isOk = false;
	}
	return int(nread);
}

/* seek, tell */

hkResult hkStdioStreamReader::seek( int offset, SeekWhence whence)
{
	if(fseek((FILE*)m_handle, offset, whence) == 0)
	{
		m_isOk = true;
		return HK_SUCCESS;
	}
	m_isOk = false;
	return HK_FAILURE;
}

int hkStdioStreamReader::tell() const
{
	return ftell((FILE*)m_handle);
}

// check that we can use these interchangeably
HK_COMPILE_TIME_ASSERT(hkSeekableStreamReader::STREAM_SET==SEEK_SET
					   && hkSeekableStreamReader::STREAM_CUR==SEEK_CUR
					   && hkSeekableStreamReader::STREAM_END==SEEK_END );

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
