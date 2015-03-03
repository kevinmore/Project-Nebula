/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/Stdio/hkStdioStreamWriter.h>
#include <Common/Base/Fwd/hkcstdio.h>

using namespace std;

hkStdioStreamWriter* hkStdioStreamWriter::open(const char* name, const char* mode)
{
	if( FILE* f = fopen(name, mode) )
	{
		return new hkStdioStreamWriter(f, true);
	}
	return HK_NULL;
}

hkStdioStreamWriter::hkStdioStreamWriter(void* handle, hkBool shouldClose)
	:	m_handle(handle), m_shouldClose(shouldClose)
{
}

void hkStdioStreamWriter::close()
{
	if(m_handle != HK_NULL && m_shouldClose)
	{
		fclose( (FILE*)m_handle);
	}
	m_handle = HK_NULL;
}

hkStdioStreamWriter::~hkStdioStreamWriter()
{
	close();
}

int hkStdioStreamWriter::write( const void* buf, int nbytes)
{
	if( m_handle != HK_NULL && nbytes > 0 )
	{
		int n = static_cast<int>( fwrite( buf, 1, nbytes, (FILE*)m_handle ) );
		if( n <= 0 )
		{
			close();
		}
		return n;
	}
	return 0;
}

void hkStdioStreamWriter::flush()
{
	if( m_handle != HK_NULL )
	{
		fflush( (FILE*)m_handle );
	}
}

hkBool hkStdioStreamWriter::isOk() const
{
	return m_handle != HK_NULL;
}

hkBool hkStdioStreamWriter::seekTellSupported() const
{
	return true;
}

hkResult hkStdioStreamWriter::seek( int offset, SeekWhence whence)
{
	return fseek((FILE*)m_handle, offset, whence) == 0
		? HK_SUCCESS
		: HK_FAILURE;
}

int hkStdioStreamWriter::tell() const
{
	return ftell((FILE*)m_handle);
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
