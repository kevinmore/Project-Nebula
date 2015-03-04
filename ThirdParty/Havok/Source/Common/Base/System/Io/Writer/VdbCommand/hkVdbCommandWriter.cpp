/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Writer/VdbCommand/hkVdbCommandWriter.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>


hkVdbCommandWriter::hkVdbCommandWriter( hkStreamWriter* writer, hkUint8 command, hkUint32 size )
: m_buffer( size )
, m_writer( writer )
, m_writePos( 0 )
, m_command( command )
{	
	HK_ASSERT2( 0x231a52e7, size > 0, "The buffer size must be greater than 2." );
}

hkVdbCommandWriter::~hkVdbCommandWriter()
{
	flush();
}

void hkVdbCommandWriter::flush()
{
	if ( m_writePos )
	{
		writeCommand( true );
	}
}

hkBool hkVdbCommandWriter::isOk() const
{
	return m_writer->isOk();
}

// This doesn't ever send all the nbytes, since it doesn't know whether
// the last bytes are part of the final command. The buffer must be
// flushed to ensure that they are sent.
int hkVdbCommandWriter::write(const void* buf, int nbytes)
{
	int readPos = 0;
	while ( readPos != nbytes ) 
	{
		const int numBytesToWrite = hkMath::min2<int>( m_buffer.getSize() - m_writePos, nbytes - readPos );

		if ( numBytesToWrite )
		{
			for ( int i = 0; i < numBytesToWrite; ++i )
			{
				m_buffer[m_writePos + i] = ((char*) buf)[readPos + i];
			}

			readPos += numBytesToWrite;
			m_writePos += numBytesToWrite;
		}
		else
		{
			writeCommand( false );
		}
	}
	return readPos;
}

void hkVdbCommandWriter::writeCommand( hkBool final )
{
	{
		// Write command size.
		hkOArchive os( m_writer );
		os.write32u( 2 + m_writePos );
		// Write the header.
		os.write8u( m_command );
		os.write8u( final ? 1 : 0 );
	}
	m_writer->write( m_buffer.begin(), m_writePos );
	m_writePos = 0;
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
