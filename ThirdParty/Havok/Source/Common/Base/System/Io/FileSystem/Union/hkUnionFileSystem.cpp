/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/FileSystem/Union/hkUnionFileSystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>

namespace
{
	struct MountIterator
	{
		typedef hkUnionFileSystem::Mount Mount;

		MountIterator( hkArray<Mount>& mounts, const char* pathIn )
			: m_mounts(mounts), m_idx(m_mounts.getSize()), m_pathIn(pathIn)
		{
			if(m_pathIn)
			{
				
				
				while( hkString::beginsWith(m_pathIn, "./") )
				{
					m_pathIn += 2;
				}
			}
		}

		bool advance()
		{
			if( m_idx > 0 )
			{
				m_idx -= 1;
				return true;
			}
			return false;
		}

		Mount& current()
		{
			return m_mounts[m_idx];
		}

		const char* path()
		{
			const Mount& m = m_mounts[m_idx];
			m_pathOut = m_pathIn;
			m_pathOut.replace( m.m_srcPath, m.m_dstPath, hkStringBuf::REPLACE_ONE);
			return m_pathOut;
		}

		hkArray<Mount>& m_mounts;
		int m_idx;
		const char* m_pathIn;
		hkStringBuf m_pathOut;
	};
}

hkFileSystem* hkUnionFileSystem::resolvePath( const char* pathIn, hkStringBuf& pathOut )
{
	MountIterator it(m_mounts, pathIn);
	while( it.advance() )
	{
		const Mount& m = it.current();
		Entry entry;
		if( m.m_fs->stat(it.path(), entry) == hkFileSystem::RESULT_OK )
		{
			pathOut = it.path();
			return m.m_fs;
		}
	}
	return HK_NULL;
}


hkRefNew<hkStreamReader> hkUnionFileSystem::openReader(const char* name, OpenFlags flags)
{
	MountIterator it(m_mounts, name);
	while( it.advance() )
	{
		const Mount& m = it.current();
		if( hkStreamReader* r = m.m_fs->openReader( it.path(), flags ).stealOwnership() )
		{
			return r;
		}
	}
	return HK_NULL;
}

// Open a file writer, creating the file directory structure as we go along
static hkRefNew<hkStreamWriter> openWriterCreatingDirectories(hkArray<hkUnionFileSystem::Mount>& mounts, const char* name, hkFileSystem::OpenFlags flags)
{
	MountIterator it(mounts, name);
	
	while( it.advance() )
	{
		const hkUnionFileSystem::Mount& m = it.current();
		if( m.m_writable )
		{
			// Make each directory subpart
			hkStringBuf directoryBits(it.path());
			directoryBits.pathNormalize();
			directoryBits.pathDirname();
			hkArray<const char*>::Temp parts;
			int totalNumParts = directoryBits.split('/', parts);
			for(int numParts=1; numParts<=totalNumParts; numParts++)
			{
				hkStringBuf currentPathPart(m.m_dstPath);
				for(int i=0;i<numParts;i++)
				{
					currentPathPart.pathAppend(parts[i]);
				}
				m.m_fs->mkdir(currentPathPart);
			}
			
			if( hkStreamWriter* w = m.m_fs->openWriter( it.path(), flags ).stealOwnership() )
			{
				return w;
			}
		}
	}
	
	return HK_NULL;
}

hkRefNew<hkStreamWriter> hkUnionFileSystem::openWriter(const char* name, OpenFlags flags)
{
	MountIterator it(m_mounts, name);
	bool dirExistsInReadableFS = false;
	
	while( it.advance() )
	{
		const Mount& m = it.current();
		if( m.m_writable )
		{
			if( hkStreamWriter* w = m.m_fs->openWriter( it.path(), flags ).stealOwnership() )
			{
				return w;
			}
		}
		else if(!dirExistsInReadableFS)
		{
			hkStringBuf pathDirName(it.path());
			pathDirName.pathDirname();
			
			hkFileSystem::Entry dummyEntry;
			dirExistsInReadableFS = m.m_fs->stat(pathDirName, dummyEntry);
		}
	}
	if(dirExistsInReadableFS)
	{
		// If the directory exists in a read-only file system, create a
		// copy of the directory structure in the writeable file system
		return openWriterCreatingDirectories(m_mounts, name, flags);
	}
	return HK_NULL;
}

hkFileSystem::Result hkUnionFileSystem::remove(const char* path)
{
	MountIterator it(m_mounts, path);
	while( it.advance() )
	{
		const Mount& m = it.current();
		if( m.m_writable )
		{
			if( m.m_fs->remove( it.path() ) == hkFileSystem::RESULT_OK )
			{
				return RESULT_OK;
			}
		}
	}
	return RESULT_ERROR;
}

hkFileSystem::Result hkUnionFileSystem::mkdir(const char* path)
{
	MountIterator it(m_mounts, path);
	while( it.advance() )
	{
		const Mount& m = it.current();
		if( m.m_writable )
		{
			if( m.m_fs->mkdir( it.path() ) == hkFileSystem::RESULT_OK )
			{
				return RESULT_OK;
			}
		}
	}
	return RESULT_ERROR;
}

hkFileSystem::Result hkUnionFileSystem::stat( const char* path, Entry& entryOut )
{
	MountIterator it(m_mounts, path);
	while( it.advance() )
	{
		const Mount& m = it.current();
		if( m.m_fs->stat( it.path(), entryOut ) == hkFileSystem::RESULT_OK )
		{
			// reverse map the filename
			hkStringBuf sb = m.m_srcPath.cString();
			sb.pathAppend( entryOut.getPath() + hkString::strLen(m.m_dstPath) );
			entryOut.setPath(this, sb);
			return RESULT_OK;
		}
	}
	return RESULT_ERROR;
}

void hkUnionFileSystem::mount(hkFileSystem* fs, const char* srcPath, const char* dstPath, hkBool writable)
{
	Mount& m = m_mounts.expandOne();
	m.m_fs = fs;
	m.m_srcPath = srcPath;
	m.m_dstPath = dstPath;
	m.m_writable = writable;
}

namespace
{
	struct UnionIterator : public hkFileSystem::Iterator::Impl
	{
		UnionIterator( hkUnionFileSystem* fs, hkArray<hkUnionFileSystem::Mount>& mounts, const char* top, const char* wildcard)
			: m_fs(fs)
			, m_mounts(mounts)
			, m_top(top)
			, m_wildcard(wildcard)
			, m_next(HK_NULL)
			, m_mountIndex(mounts.getSize())
		{
		}

		virtual bool advance(hkFileSystem::Entry& e)
		{
			while(1)
			{
				if( m_next )
				{
					if( m_next->advance(e) )
					{
						const hkUnionFileSystem::Mount& m = m_mounts[m_mountIndex];
						HK_ASSERT(0x33819923, hkString::beginsWith(e.getPath(), m.m_dstPath));
						// reverse map the filename
						hkStringBuf sb = m.m_srcPath.cString();
						sb.pathAppend( e.getPath() + hkString::strLen(m.m_dstPath) );
						e.setPath(m_fs, sb);
						return true;
					}
					m_next = HK_NULL;
				}
				m_mountIndex -= 1;
				if( m_mountIndex >= 0 )
				{
					const hkUnionFileSystem::Mount& m = m_mounts[m_mountIndex];
					if( hkString::beginsWith(m_top, m.m_srcPath) )
					{
						hkStringBuf sb = m.m_dstPath.cString();
						sb.pathAppend( m_top + hkString::strLen(m.m_srcPath) );
						m_next = m.m_fs->createIterator(sb, m_wildcard);
					}
				}
				else
				{
					return false;
				}
			}
		}

		hkUnionFileSystem* m_fs;
		hkArray<hkUnionFileSystem::Mount>& m_mounts;
		hkStringPtr m_top;
		hkStringPtr m_wildcard;
		hkRefPtr<hkFileSystem::Iterator::Impl> m_next;
		int m_mountIndex;
	};
}

hkRefNew<hkFileSystem::Iterator::Impl> hkUnionFileSystem::createIterator( const char* top, const char* wildcard )
{
	return new UnionIterator( this, m_mounts, top, wildcard);
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
