/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/Win32/hkWin32FileSystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Container/String/hkUtf8.h>
#include <Common/Base/System/Io/Platform/Stdio/hkStdioStreamReader.h>

#if defined(HK_PLATFORM_XBOX360)
	#include <Common/Base/System/Io/Platform/Stdio/hkStdioStreamWriter.h>
	#define GetFileAttributesW GetFileAttributes
	#define FindFirstFileW FindFirstFile
	#define FindNextFileW FindNextFile
	#define DeleteFileW DeleteFile	
	#define CreateDirectoryW CreateDirectory
	#define HK_WIN32_FIND_DATA WIN32_FIND_DATA
	#define HK_WIDEN(X) (X)
	#define HK_NARROW(X) (X)
	#define HK_STRCMP(VAR,CONST_STRING) hkString::strCmp(VAR, CONST_STRING)
	#include <Xtl.h>

// XBox 360 is quite strict on input paths
static void s_convertFileNameToNative(hkStringBuf& filename)
{
	filename.replace('/', '\\');

	hkStringBuf tempBuffer = filename;
	hkArray<const char*>::Temp bits; // Should be ::Temp, but hkStringBuf::split doesn't handle that
	tempBuffer.split('\\', bits);

	int i = 0;
	const char dot = '.';
	while ( i < bits.getSize() )
	{
		if (bits[i][0] == dot)
		{
			if(bits[i][1] == 0) // "." in the path
			{
				bits.removeAtAndCopy(i);
				i--;

			}
			else if (bits[i][1] == dot) // ".." in the path
			{

				HK_WARN_ON_DEBUG_IF(i <= 0, 0x37c8d421, "Can't open file below root directory"); // "../dir1/file.txt" isn't valid
				HK_WARN_ON_DEBUG_IF(!( (bits[i][1] == dot) && (bits[i][2] == 0) ), 0x28bcd825, "Invalid filename"); // "dir1/..asdf" or "dir2/.asdf"
				bits.removeAtAndCopy(i);
				--i;
				if (i >= 0)
				{
					bits.removeAtAndCopy(i);
					--i;
				}
			}
		}
		i++;
	}

	filename.clear();
	for (i=0; i<bits.getSize(); i++)
	{
		filename += bits[i];
		if (i != bits.getSize() - 1)
			filename += "\\";
	}
}

#else
	#include <Common/Base/System/Io/Platform/Win32/hkWin32StreamWriter.h>
	#define HK_WIN32_FIND_DATA WIN32_FIND_DATAW
	#define HK_WIDEN(X) hkUtf8::WideFromUtf8(X).cString()
	#define HK_NARROW(X) hkUtf8::Utf8FromWide(X).cString()
	#define HK_STRCMP(VAR,CONST_STRING) wcscmp(VAR, L ## CONST_STRING)
	#include <windows.h>

static void s_convertFileNameToNative(hkStringBuf& filename)
{
	filename.replace('/', '\\');
}

#endif
#include <io.h>

//Keep me to compute the #100ns since the epoch
// 	SYSTEMTIME st = { 1970,1,0,1,   0,0,0,0 };
// 	FILETIME ft;
// 	SystemTimeToFileTime(&st, &ft);
// 	hkInt64 delta = s_combineHiLoDwords( ft.dwHighDateTime, ft.dwLowDateTime );
//  delta == 116444736000000000UI64 * 100ns from win32 epoch to linux epoch

#define HK_TIMESTAMP_NSEC100_TO_UNIX_EPOCH  116444736000000000UI64

static hkUint64 s_combineHiLoDwords(DWORD high, DWORD low)
{
	return low + (hkUint64(high)<<32);
}

static hkUint64 s_convertWindowsFiletimeToUnixTime(DWORD high, DWORD low)
{
	hkUint64 filetime = low + (hkUint64(high)<<32);
	return (filetime - HK_TIMESTAMP_NSEC100_TO_UNIX_EPOCH) * 100;
}

// Populate an entry from a finddata. Return false on failure (e.g. name==".")
static void s_entryFromFindData( hkFileSystem::Entry& e, hkFileSystem* fs, const char* top, const HK_WIN32_FIND_DATA& f )
{
	e.setAll
	(
		fs,
		hkStringBuf(top).pathAppend( HK_NARROW( f.cFileName ) ),
		f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ? hkFileSystem::Entry::F_ISDIR : hkFileSystem::Entry::F_ISFILE,
		s_convertWindowsFiletimeToUnixTime( f.ftLastWriteTime.dwHighDateTime, f.ftLastWriteTime.dwLowDateTime ),
		s_combineHiLoDwords( f.nFileSizeHigh, f.nFileSizeLow )
	);
}

hkRefNew<hkStreamReader> hkWin32FileSystem::openReader(const char* name, OpenFlags flags)
{
	hkStringBuf sb = name; s_convertFileNameToNative(sb);
	// Stdio stream reader is already buffered so we don't need to buffer again
	return _handleFlags( hkStdioStreamReader::open(sb), OpenFlags(flags&~OPEN_BUFFERED) );
}

hkRefNew<hkStreamWriter> hkWin32FileSystem::openWriter(const char* name, OpenFlags flags)
{
	hkStringBuf sb = name; s_convertFileNameToNative(sb);
	#if defined(HK_PLATFORM_XBOX360)
		const char* mode = ( flags & OPEN_TRUNCATE ) ? "wb" : "r+b";
		return _handleFlags( hkStdioStreamWriter::open( sb, mode), OpenFlags(flags&~OPEN_BUFFERED) );
	#else
		int dwCreationDisposition = (flags & OPEN_TRUNCATE) ? CREATE_ALWAYS : OPEN_ALWAYS;
		return _handleFlags( hkWin32StreamWriter::open(sb, dwCreationDisposition ), flags );
	#endif
}

hkFileSystem::Result hkWin32FileSystem::remove(const char* path)
{
	hkStringBuf sb = path; s_convertFileNameToNative(sb);
	return DeleteFileW( HK_WIDEN(sb) ) ? RESULT_OK : RESULT_ERROR;
}
hkFileSystem::Result hkWin32FileSystem::mkdir(const char* path)
{
	hkStringBuf sb = path; s_convertFileNameToNative(sb);
	return CreateDirectoryW( HK_WIDEN(sb), HK_NULL ) ? RESULT_OK : RESULT_ERROR;
}


hkFileSystem::Result hkWin32FileSystem::stat( const char* path, Entry& entryOut )
{
	hkStringBuf sb = path; s_convertFileNameToNative(sb);
	HK_ASSERT2(0x129e4884, hkString::strChr(path,'*')==0, "Use an iterator for wildcards" );
	HK_WIN32_FIND_DATA findData;
	// FindFirstFileW cannot handle a path that ends with a slash, so we have to remove it.
	while (sb.endsWith("\\"))
	{
		sb.chompEnd(1);
	}
	
	HANDLE h = FindFirstFileW( HK_WIDEN(sb), &findData);
	if( h != INVALID_HANDLE_VALUE )
	{

		entryOut.setAll
		(
			this,
			path,
			findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ? hkFileSystem::Entry::F_ISDIR : hkFileSystem::Entry::F_ISFILE,
			s_convertWindowsFiletimeToUnixTime( findData.ftLastWriteTime.dwHighDateTime, findData.ftLastWriteTime.dwLowDateTime ),
			s_combineHiLoDwords( findData.nFileSizeHigh, findData.nFileSizeLow )
		);
		FindClose(h);
		return RESULT_OK;
	}
	return RESULT_ERROR;
}

namespace
{
	struct Win32Impl : public hkFileSystem::Iterator::Impl
	{
		Win32Impl(hkFileSystem* fs, const char* top, const char* wildcard)
			: m_fs(fs)
			, m_handle(0)
			, m_top(top)
			, m_wildcard(wildcard)
		{
			HK_ASSERT2(0x1e0bb0cd, hkString::strChr(m_top,'*') == HK_NULL, "Path part cannot contain a *");
			HK_ASSERT2(0x47f9de01, wildcard==HK_NULL || hkString::strChr(wildcard,'*'), "Wildcard must be null or contain a *" );
		}

		virtual bool advance(hkFileSystem::Entry& e)
		{
			if( m_handle == 0 ) // first time through
			{
				HK_WIN32_FIND_DATA findData;
				hkStringBuf pattern( m_top );
				pattern.pathAppend( m_wildcard ? m_wildcard : "*");
				s_convertFileNameToNative(pattern);

				m_handle = FindFirstFileW( HK_WIDEN(pattern), &findData );
				if( m_handle == INVALID_HANDLE_VALUE )
				{
					return false;
				}
				if( hkFileSystem::Iterator::nameAcceptable( HK_NARROW(findData.cFileName), HK_NULL) )
				{
					s_entryFromFindData(e, m_fs, m_top, findData);
					return true;
				}
			}
			if( m_handle != INVALID_HANDLE_VALUE )
			{
				HK_WIN32_FIND_DATA findData;
				while( FindNextFileW( m_handle, &findData) )
				{
					if( hkFileSystem::Iterator::nameAcceptable( HK_NARROW(findData.cFileName), HK_NULL ) )
					{
						s_entryFromFindData(e, m_fs, m_top, findData);
						return true;
					}
				}
				// we ran out of entries
				FindClose( m_handle );
				m_handle = INVALID_HANDLE_VALUE;
			}
			return false;
		}

		~Win32Impl()
		{
			if( m_handle != 0 && m_handle != INVALID_HANDLE_VALUE )
			{
				FindClose( m_handle );
			}
		}

		hkFileSystem* m_fs;
		hkStringPtr m_top;
		hkStringPtr m_wildcard;
		HANDLE m_handle;
	};
}

hkRefNew<hkFileSystem::Iterator::Impl> hkWin32FileSystem::createIterator( const char* top, const char* wildcard )
{
	return new Win32Impl(this, top, wildcard);
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
