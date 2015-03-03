/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>

#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>

#ifndef HK_PLATFORM_SPU
HK_REFLECTION_DEFINE_STUB_VIRTUAL_BASE(hkArrayStreamWriter);
#endif

hkMemoryTrack::hkMemoryTrack( int numBytesPerSector  )
{
	m_numBytesLastSector = numBytesPerSector;
	m_numBytesPerSector = numBytesPerSector;
	m_numBytesRead = 0;
	m_numSectorsUnloaded = 0;	
}

hkMemoryTrack::~hkMemoryTrack()
{
	clear();
}

void hkMemoryTrack::clear()
{
	for (int i = 0; i < m_sectors.getSize(); i++)
	{
		hkDeallocateChunk(m_sectors[i], m_numBytesPerSector, HK_MEMORY_CLASS_BASE );
	}
	m_sectors.clearAndDeallocate();
	m_numBytesLastSector = m_numBytesPerSector;
	m_numBytesRead = 0;
	m_numSectorsUnloaded = 0;
}

void hkMemoryTrack::write ( const void* data, int numBytes )
{
	int currentSector = m_sectors.getSize()-1;

	while ( numBytes > 0 )
	{
		int bytesFreeInLastSector = m_numBytesPerSector - m_numBytesLastSector;
		if ( !bytesFreeInLastSector )
		{
			hkUint8* newSector = hkAllocateChunk<hkUint8>(m_numBytesPerSector, HK_MEMORY_CLASS_BASE );
			m_sectors.pushBack( newSector );
			m_numBytesLastSector = 0;
			currentSector++;
			bytesFreeInLastSector = m_numBytesPerSector;
		}

		hkUint8* dest = m_sectors[currentSector] + m_numBytesLastSector;

		if ( numBytes <= bytesFreeInLastSector )
		{
			hkString::memCpy( dest, data, numBytes );
			m_numBytesLastSector += numBytes;
			numBytes = 0;
			break;
		}
		hkString::memCpy( dest, data, bytesFreeInLastSector );
		data = hkAddByteOffsetConst( data, bytesFreeInLastSector );
		numBytes -= bytesFreeInLastSector;
		m_numBytesLastSector = m_numBytesPerSector;
	}
}

void hkMemoryTrack::appendByMove( hkMemoryTrack* other )
{
	int otherNumSectors = other->m_sectors.getSize();
	for ( int i = 0; i < otherNumSectors-1; i++)
	{
		hkUint8* source = other->m_sectors[i];
		write ( source, other->m_numBytesPerSector );
		hkDeallocateChunk(source, other->m_numBytesPerSector, HK_MEMORY_CLASS_BASE );
	}
	if ( otherNumSectors )
	{
		hkUint8* source = other->m_sectors[otherNumSectors-1];
		write ( source, other->m_numBytesLastSector );
		hkDeallocateChunk(source, other->m_numBytesPerSector, HK_MEMORY_CLASS_BASE );
	}
	other->m_sectors.clearAndDeallocate();
	other->clear();
}


void hkMemoryTrack::read( void* data, int numBytes ) 
{
	HK_ASSERT2( 0xf023defd, m_numBytesRead + numBytes <= getSize(), "Cannot read data" );

	int currentSector = m_numBytesRead / m_numBytesPerSector - m_numSectorsUnloaded;
	int sectorOffset = m_numBytesRead - (m_numSectorsUnloaded * m_numBytesPerSector) - currentSector * m_numBytesPerSector;	

	while ( numBytes > 0 )
	{
		int bytesFreeInSector = (currentSector < m_sectors.getSize()-1 ) ? m_numBytesPerSector : m_numBytesLastSector;
		bytesFreeInSector -= sectorOffset;
		const hkUint8* source = m_sectors[currentSector] + sectorOffset;

		if ( numBytes <= bytesFreeInSector )
		{
			hkString::memCpy( data, source, numBytes );			
			m_numBytesRead += numBytes;
			numBytes = 0;

			break;
		}
		hkString::memCpy( data, source, bytesFreeInSector );
		numBytes -= bytesFreeInSector;
		m_numBytesRead += bytesFreeInSector;
		data = hkAddByteOffset( data, bytesFreeInSector );		
		currentSector++;		

		sectorOffset = 0;
	}
}

void hkMemoryTrack::unloadReadSectors()
{		
	int numBytesUnloaded = m_numSectorsUnloaded * m_numBytesPerSector;	
	int numSectorsToUnload = (m_numBytesRead - numBytesUnloaded) / m_numBytesPerSector;
	for( int i = 0; i < numSectorsToUnload; ++i )
	{
		hkDeallocateChunk(m_sectors[0], m_numBytesPerSector, HK_MEMORY_CLASS_BASE );
		m_sectors.removeAtAndCopy(0);
		m_numSectorsUnloaded++;
	}	
}

void hkMemoryTrackStreamWriter::clear()
{
	m_track->clear();
}

int hkMemoryTrackStreamWriter::write(const void* mem, int size)
{
	m_track->write( mem, size );
	return size;
}



void hkArrayStreamWriter::clear()
{
	m_arr->clear();
	m_offset = 0;
	nullTerminate();
}

int hkArrayStreamWriter::write(const void* mem, int size)
{
	HK_ASSERT2( 0x170ce358, m_offset <= m_arr->getSize(),
		"Array size has changed without a call to seek" );
	int spaceLeft = m_arr->getSize() - m_offset;
	if( size > spaceLeft )
	{
		int newSize = size + m_arr->getSize() - spaceLeft;
		m_arr->_reserve(m_allocator, 1 + newSize );
		m_arr->setSizeUnchecked( newSize );
		m_arr->begin()[ newSize ] = 0;
	}
	else if( m_arr->getCapacity() > m_arr->getSize() )
	{
		m_arr->begin()[ m_arr->getSize() ] = 0;
	}
	char* p = m_arr->begin() + m_offset;
	hkString::memCpy(p, mem, size);
	m_offset += size;

	return size;
}

hkMemoryTrackStreamReader::hkMemoryTrackStreamReader( const hkMemoryTrack* track, MemoryType t, bool unloadSectorsAfterRead)
{
	m_memType = t;
	m_track = track;	
	m_overflowOffset = -1;
	m_unloadSectorsAfterRead = unloadSectorsAfterRead;
}

hkMemoryTrackStreamReader::~hkMemoryTrackStreamReader()
{
	if( m_memType == MEMORY_TAKE )
	{
		delete m_track;
	}
}

int hkMemoryTrackStreamReader::read(void* buf, int nbytes)
{
	if( isOk() )
	{				
		int nRead = hkMath::min2( nbytes, m_track->getAvailable());

		// If the request to read is larger than what the track has
		// available note the track size and return 0 as no bytes can
		// be read.
		if( nbytes > m_track->getAvailable() )
		{
			m_overflowOffset = m_track->getNumBytesRead() + nbytes;
			return 0;
		}
		
		const_cast<hkMemoryTrack*>(m_track)->read(buf, nRead);				

		if( m_unloadSectorsAfterRead )
		{
			const_cast<hkMemoryTrack*>(m_track)->unloadReadSectors();
			
			// Reset the offset if we've read all the data.  This is done because offset can get very large
			// and overlow.
			if( m_track->hasReadAllData() )
			{				
				const_cast<hkMemoryTrack*>(m_track)->clear();
				m_overflowOffset = -1;				
			}
		}

		return nRead;
	}
	return 0;
}

int hkMemoryTrackStreamReader::skip( int nbytes)
{
	HK_ASSERT2(0x4a5c9ddb, false, "Not implemented.");
	return 0;
}

hkResult hkArrayStreamWriter::seek(int offset, SeekWhence whence)
{
	int absOffset = m_offset;
	switch( whence )
	{
	case STREAM_SET:
		absOffset = offset;
		break;
	case STREAM_CUR:
		absOffset = m_offset + offset;
		break;
	case STREAM_END:
		absOffset = m_arr->getSize() - offset;
		break;
	default:
		HK_ASSERT2(0x55f1b803, 0, "Bad 'whence' passed to seek()");
		break;
	}
	if( absOffset >= 0 )
	{
		if( absOffset > m_arr->getSize() )
		{
			m_arr->_setSize( m_allocator, absOffset+1, 0 ); // zero filled space, null terminated
			m_arr->setSizeUnchecked( absOffset );
		}
		m_offset = absOffset;
		return HK_SUCCESS;
	}
	return HK_FAILURE;
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
