/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/Posix/hkPosixFileSystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/Reader/Buffered/hkBufferedStreamReader.h>
#include <Common/Base/System/Io/Platform/Posix/hkPosixStreamWriter.h>
#include <Common/Base/System/Io/Platform/Posix/hkPosixStreamReader.h>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define EMULATE_CASE_INSENSITIVE_FS
#ifdef EMULATE_CASE_INSENSITIVE_FS

// Some platforms don't define this
#if !defined(PATH_MAX)
#define PATH_MAX 4096
#endif

	///
	static bool fix_path_case(char* fullpath)
	{
		// If stat succeeds, we're done
		// stat handles doubled slashes etc

		struct stat s;
		if( stat(fullpath,&s) == 0 )
		{
			return true;
		}

		bool fixed = false;
		char* lastslash = strrchr(fullpath,'/');

		// handle any trailing slashes specially e.g. /usr/bar///
		// because they can't easily be split into head and tail

		if( lastslash && lastslash[1] == 0)
		{
			while( lastslash!=fullpath && lastslash[-1]=='/')
			{
				lastslash -= 1;
			}
			*lastslash = 0;
			fixed = fix_path_case(fullpath);
			*lastslash = '/';
			return fixed;
		}

		// split fullpath into its head and tail

		const char* head; // the folder to look in
		char* tail; // the last part we're trying to fix


		if( lastslash == 0 )
		{
			// no slashes e.g. "foo"
			head = ".";
			tail = fullpath;
		}
		else if( lastslash == fullpath )
		{
			// fullpath is fully rooted e.g. "/usr"
			head = "/";
			tail = fullpath + 1;
			lastslash = 0;
		}
		else
		{
			// normal case "a/foo/bar"
			*lastslash = 0; // split fullpath in place
			head = fullpath; // "a/foo"
			tail = lastslash + 1; // "bar"
			if( fix_path_case(fullpath) == false )
			{
				*lastslash = '/';
				return false;
			}
		}

		// here, we know head is ok, tail must be fixed

		if( DIR* dir = opendir(head) )
		{
			while( dirent* dent = readdir(dir) )
			{
				if( strcasecmp(dent->d_name,tail) == 0 )
				{
					fixed = true;
					strcpy(tail, dent->d_name);
					break;
				}
			}
			closedir(dir);
		}

		// restore if we did an inplace split

		if(lastslash)
		{
			*lastslash = '/';
		}
		return fixed;
	}

	struct FIX_PATH_CASE
	{
		FIX_PATH_CASE(const char* s)
		{
			m_buf[PATH_MAX] = 0;
			strncpy(m_buf,s, sizeof(m_buf)-1);
			fix_path_case(m_buf);
		}
		operator const char*() const { return m_buf; }
		char m_buf[PATH_MAX+1];
	};
#else
	#define FIX_PATH_CASE(PATH) PATH
#endif



hkRefNew<hkStreamReader> hkPosixFileSystem::openReader(const char* name, OpenFlags flags)
{
	return _handleFlags( hkPosixStreamReader::open(FIX_PATH_CASE(name)), flags );
}

hkRefNew<hkStreamWriter> hkPosixFileSystem::openWriter(const char* name, OpenFlags flags)
{
	int posixMode =	 O_WRONLY | O_CREAT | ((flags & OPEN_TRUNCATE) ? O_TRUNC : 0);
	return _handleFlags( hkPosixStreamWriter::open(FIX_PATH_CASE(name), posixMode), flags );
}

hkFileSystem::Result hkPosixFileSystem::remove(const char* name)
{
	return ::unlink(FIX_PATH_CASE(name)) >= 0 ? RESULT_OK : RESULT_ERROR;
}

hkFileSystem::Result hkPosixFileSystem::mkdir(const char* name)
{
	return ::mkdir(FIX_PATH_CASE(name), 0755) >= 0 ? RESULT_OK : RESULT_ERROR;
}

static int s_stat( const char* path, hkFileSystem* fs, hkFileSystem::Entry& entryOut )
{
	struct stat st;
	
	if (stat( path, &st ) == 0) 
	{
		entryOut.setAll
		(
			fs,
			path,
			S_ISDIR( st.st_mode ) ? hkFileSystem::Entry::F_ISDIR
				: S_ISREG( st.st_mode ) ? hkFileSystem::Entry::F_ISFILE
				: hkFileSystem::Entry::F_ISUNKNOWN,
			#if defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_PS4)
				hkUint64(st.st_mtimespec.tv_sec * 1000000000L) + st.st_mtimespec.tv_nsec,
			#elif defined(HK_PLATFORM_LINUX) || defined(HK_PLATFORM_TIZEN)
				hkUint64(st.st_mtim.tv_sec * 1000000000L) + st.st_mtim.tv_nsec,
			#else
				hkUint64(st.st_mtime * 1000000000L) + st.st_mtime_nsec,
			#endif
			st.st_size
			);
			return 0;
	}
	return 1;
}

hkFileSystem::Result hkPosixFileSystem::stat( const char* path, hkFileSystem::Entry& entryOut )
{
	if( s_stat(FIX_PATH_CASE(path), this, entryOut) == 0)
	{
		return RESULT_OK;
	}
	else
	{
		return RESULT_ERROR;
	}
}

namespace
{
	struct PosixIter : public hkFileSystem::Iterator::Impl
	{
		PosixIter(hkFileSystem* fs, const char* top, const char* wildcard, DIR* handle)
		: m_fs(fs), m_top(top), m_wildcard(wildcard), m_handle(handle)
		{
		}

		~PosixIter()
		{
			if( m_handle )
			{
				closedir(m_handle);
				m_handle = HK_NULL;
			}
		}
		
		virtual bool advance(hkFileSystem::Entry& e)
		{
			while(m_handle)
			{
				struct dirent* dent = readdir(m_handle);
				if( dent != HK_NULL )
				{
					if( hkFileSystem::Iterator::nameAcceptable(dent->d_name, m_wildcard) )
					{
						if( s_stat( hkStringBuf( m_top ).pathAppend( dent->d_name ), m_fs, e) == 0 )
						{
							return true;
						}
					}
				}
				else
				{
					closedir(m_handle);
					m_handle = HK_NULL;
				}
			}
			return false;
		}
		hkFileSystem* m_fs;
		hkStringPtr m_top;
		hkStringPtr m_wildcard;
		DIR* m_handle;
	};
}

hkRefNew<hkFileSystem::Iterator::Impl> hkPosixFileSystem::createIterator( const char* top, const char* wildcard )
{
	// Linux doesn't like "" to mean current directory
	if(hkString::strLen(top) == 0)
	{
		top = ".";
	}
	if( DIR* handle = opendir( top ) )
	{
		return new PosixIter(this, top, wildcard, handle);
	}
	return HK_NULL;
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
