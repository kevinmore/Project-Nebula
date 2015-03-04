/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/Reader/Buffered/hkBufferedStreamReader.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>



void hkFileSystem::TimeStamp::set(hkUint64 nsSinceEpoch)
{
	m_time = nsSinceEpoch;
}

hkUint64 hkFileSystem::TimeStamp::get() const
{
	return m_time;
}

hkBool hkFileSystem::TimeStamp::isValid() const
{
  return m_time != hkFileSystem::TimeStamp::TIMESTAMP_UNAVAILABLE;
}



hkRefNew<hkStreamReader> hkFileSystem::Entry::openReader(OpenFlags flags) const
{
	return m_fs->openReader( m_path, flags );
}

hkRefNew<hkStreamWriter> hkFileSystem::Entry::openWriter(OpenFlags flags) const
{
	return m_fs->openWriter( m_path, flags );
}

void hkFileSystem::Entry::setPath( hkFileSystem* fs, const char* path )
{
	HK_ASSERT(0x8692bcc, hkString::endsWith(path, "/")==false);
	m_fs = fs;
	m_path = path;
}

const char* hkFileSystem::Entry::getName() const
{
	HK_ASSERT(0x73b2865a, m_path.endsWith("/") == hkFalse32);
	if( const char* name = hkString::strRchr(m_path, '/') )
	{
		do
		{
			++name;
		} while( *name=='/' );
		// maybe it was like "foo/bar/"
		if( *name )
		{
			return name;
		}
	}
	// "/" not found, just return the head
	return m_path;
}

void hkFileSystem::Entry::setAll( hkFileSystem* fs, const char* fullPath, Flags flags, TimeStamp mt, hkInt64 sz )
{
	m_fs = fs;
	m_path = fullPath;
	m_flags = flags;
	m_mtime = mt;
	m_size = sz;
}

hkResult hkFileSystem::listDirectory(const char* basePath, DirectoryListing& listingOut)
{
	hkFileSystem::Iterator iter(this, basePath);
	listingOut.setFs(this);
	while( iter.advance() )
	{
		listingOut.addEntry( iter.current() );
	}
	return HK_SUCCESS;
}

hkFileSystem::Iterator::Iterator(hkFileSystem* fs, const char* top, const char* wildcard)
	: m_fs(fs), m_wildcard(wildcard)
{
	HK_ASSERT(0x2b9fb1b7, wildcard==HK_NULL || hkString::strChr(wildcard, '/')==HK_NULL );
	m_todo.pushBack( top );
}

bool hkFileSystem::Iterator::advance()
{
	while( 1 )
	{
		if( m_impl != HK_NULL )
		{
			if( m_impl->advance(m_entry) )
			{
				return true;
			}
			m_impl = HK_NULL;
		}
		if( m_todo.getSize() )
		{
			// try again with the next item
			m_impl = m_fs->createIterator( m_todo.back(), m_wildcard );
			m_todo.popBack();
		}
		else
		{
			// end of this iteration and no more todos, we're done
			return false;
		}
	}
}

bool hkFileSystem::Iterator::nameAcceptable(const char* name, const char* wildcard)
{
	// nonempty, not . nor ..
	if( name[0]==0 ) return false;
	if( name[0]=='.' && name[1]==0 ) return false;
	if( name[0]=='.' && name[1]=='.' && name[2]==0 ) return false;

	if( wildcard == HK_NULL ) return true;

	HK_ASSERT(0x61cee828, hkString::strChr(wildcard, '?') == HK_NULL );
	HK_ASSERT2(0x374596bd, hkString::strChr(name, '*') == HK_NULL, "name cannot contain wildcards" );
	HK_ASSERT2(0x268fada4, wildcard[0] == '*', "only simple patterns currently supported (*.xxx)" );
	HK_ASSERT2(0x7f34a731, hkString::strChr(wildcard+1, '*')==HK_NULL, "only simple patterns currently supported (*.xxx)" );

	return hkString::endsWith(name, wildcard+1);	
}

hkStreamReader* hkFileSystem::_handleFlags(hkStreamReader* sr, OpenFlags flags)
{
	HK_ASSERT(0x5b74fb14, (flags & ~OPEN_BUFFERED) == 0 );
	if( sr )
	{
		if( flags & OPEN_BUFFERED )
		{
			hkBufferedStreamReader* buf = new hkBufferedStreamReader(sr);
			sr->removeReference();
			sr = buf;
		}
	}
	return sr;
}

hkStreamWriter* hkFileSystem::_handleFlags(hkStreamWriter* sw, OpenFlags flags)
{
	if( sw )
	{
		if( flags & OPEN_BUFFERED )
		{
			hkBufferedStreamWriter* buf = new hkBufferedStreamWriter(sw);
			sw->removeReference();
			sw = buf;
		}
	}
	return sw;
}

HK_SINGLETON_MANUAL_IMPLEMENTATION(hkFileSystem);

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
