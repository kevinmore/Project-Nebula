/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/Win32/hkWin32StreamWriter.h>
#include <Common/Base/Fwd/hkwindows.h>
#include <Common/Base/Container/String/hkUtf8.h>

#if defined(HK_PLATFORM_WINRT) || (_WIN32_WINNT >= 0x0600 )
#	define _USE_EX_FILEIO 1
#endif

hkWin32StreamWriter* hkWin32StreamWriter::open(const char* fname, int dwCreationDisposition )
{
#ifdef _USE_EX_FILEIO
	HANDLE handle = CreateFile2( hkUtf8::WideFromUtf8(fname).cString(), GENERIC_WRITE, FILE_SHARE_READ, dwCreationDisposition, HK_NULL );
#else
	HANDLE handle = CreateFileW( hkUtf8::WideFromUtf8(fname).cString(), GENERIC_WRITE, FILE_SHARE_READ, HK_NULL, dwCreationDisposition, FILE_ATTRIBUTE_NORMAL, HK_NULL );
#endif
	if(handle != INVALID_HANDLE_VALUE)
	{
		return new hkWin32StreamWriter(handle);
	}
	return HK_NULL;
}

hkWin32StreamWriter::hkWin32StreamWriter(HANDLE handle)
	: m_handle(handle)
{
}

void hkWin32StreamWriter::close()
{
	if(m_handle != INVALID_HANDLE_VALUE)
	{
		CloseHandle(m_handle);
	}
}

hkWin32StreamWriter::~hkWin32StreamWriter()
{
	close();
}

int hkWin32StreamWriter::write( const void* buf, int nbytes)
{
	if( (m_handle != INVALID_HANDLE_VALUE) && nbytes > 0 )
	{
		DWORD n;
		WriteFile( m_handle, buf, nbytes, &n, HK_NULL );
		if( n <= 0 )
		{
			close();
		}
		return n;
	}
	return 0;
}

void hkWin32StreamWriter::flush()
{
	if( m_handle != INVALID_HANDLE_VALUE )
	{
		FlushFileBuffers(m_handle);
	}
}

hkBool hkWin32StreamWriter::isOk() const
{
	return m_handle != INVALID_HANDLE_VALUE;
}

hkResult hkWin32StreamWriter::seek( int offset, SeekWhence whence)
{
#ifdef _USE_EX_FILEIO
	DWORD moveType[] = {FILE_BEGIN,FILE_CURRENT, FILE_END};
	LARGE_INTEGER newPos;
	LARGE_INTEGER offsetL; offsetL.QuadPart = offset;
	BOOL ok = SetFilePointerEx(m_handle, offsetL, &newPos, moveType[(int)whence]);
	return ok ? HK_SUCCESS : HK_FAILURE;
#else
	return SetFilePointer(m_handle, offset, HK_NULL, whence) != INVALID_SET_FILE_POINTER ? HK_SUCCESS : HK_FAILURE;
#endif
}

int hkWin32StreamWriter::tell() const
{
#ifdef _USE_EX_FILEIO
	LARGE_INTEGER newPos;
	LARGE_INTEGER offsetL; offsetL.QuadPart = 0;
	BOOL ok = SetFilePointerEx(m_handle, offsetL, &newPos, FILE_CURRENT);
	return ok? (int)( newPos.LowPart ) : 0;
#else
	DWORD off = SetFilePointer(m_handle, 0, HK_NULL, FILE_CURRENT);
	return off != INVALID_SET_FILE_POINTER ? off : -1;
#endif
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
