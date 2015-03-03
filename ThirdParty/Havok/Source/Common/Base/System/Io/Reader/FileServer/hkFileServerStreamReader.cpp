/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/System/Io/Reader/FileServer/hkFileServerStreamReader.h>
#include <Common/Base/System/Io/Socket/hkSocket.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>

hkFileServerStreamReader::hkFileServerStreamReader( hkSocket* socket, hkUint32 fileId )
: m_socket(socket), m_id(fileId)
{
}

void hkFileServerStreamReader::close()
{
	// socket is not ours to close
	if (m_socket && m_socket->isOk())
	{
		//send a close to be nice
		hkOArchive out(&m_socket->getWriter());
		out.write32u( 2 * sizeof(int) );
		out.write32u( CLOSE );
		out.write32u( m_id );
	}

	m_socket = HK_NULL;
}

hkFileServerStreamReader::~hkFileServerStreamReader()
{
	close();
}

int hkFileServerStreamReader::read( void* buf, int nbytes)
{
	if (m_socket && m_socket->isOk())
	{
		int nread = 0;

		// request block (assumes it is buffered above this :) )
		hkOArchive out(&m_socket->getWriter());
		out.write32u( 3 * sizeof(int) );
		out.write32u( READ_CHUNK );
		out.write32u( m_id );
		out.write32u( nbytes );

		hkIArchive in(&m_socket->getReader());
		if (m_socket->isOk())
		{
			int packetSize = in.read32u();
			int cmd = in.read32u();
			while (m_socket->isOk())
			{
				if (cmd == SEND_CHUNK)
				{
					HK_ON_DEBUG(hkUint32 readId =) in.read32u(); 
					HK_ASSERT(0x032134, readId  == m_id);
					
					nread = in.read32u(); // amount sent
					int sread = 0;
					while ( m_socket->isOk() && (sread < nread))
					{
						int ret = in.readRaw(buf, nread); // read!
						if (ret > 0) 
							sread += ret;
						else 
							break;
					}

					if (sread != nread)
					{
						nread = 0; // connect failed without getting full data packet
					}
					break;
				}
				else if (cmd == EOF_OR_ERROR)
				{
					HK_ON_DEBUG(hkUint32 readId =) in.read32u(); 
					HK_ASSERT(0x032134, readId  == m_id);
					break; // done already
				}
				else 
				{
					HK_WARN_ALWAYS(0x4234f, "Got crazy stuff from hkFileServerStreamReader read socket..: " << cmd );
					if (packetSize < 0x100000) 
					{
						char* ignoreData = hkAllocate<char>( packetSize, HK_MEMORY_CLASS_BASE);
						in.readRaw(ignoreData, packetSize - sizeof(int));
						hkDeallocate<char>(ignoreData);
					}
					//loop
				}
			}
		}

		if(nread <= 0)
		{
			close();
		}
		return nread;
	}
	return 0; 
}

hkBool hkFileServerStreamReader::isOk() const
{
	return (m_socket != HK_NULL) && (m_socket->isOk());
}

hkResult hkFileServerStreamReader::seek(int offset, SeekWhence whence)
{
	if (m_socket && m_socket->isOk())
	{
		hkOArchive out(&m_socket->getWriter());
		out.write32u( 4 * sizeof(int) );
		out.write32u( SEEK );
		out.write32u( m_id );
		out.write32( offset );
		out.write32u( whence );
		// don't wait.. assume it does it
		return m_socket->isOk() ? HK_SUCCESS : HK_FAILURE;
	}

	return HK_FAILURE;
}

int hkFileServerStreamReader::tell() const
{
	if (m_socket && m_socket->isOk())
	{
		hkOArchive out(&m_socket->getWriter());
		out.write32u( 2 * sizeof(int) );
		out.write32u( TELL );
		out.write32u( m_id );
		
		hkIArchive in(&m_socket->getReader());
		if (m_socket->isOk())
		{
			int packetSize = in.read32u();
			int cmd = in.read32u();
			while (m_socket->isOk())
			{
				if (cmd == SEND_TELL)
				{
					HK_ON_DEBUG(hkUint32 tellId =) in.read32u(); 
					HK_ASSERT(0x032134, tellId == m_id);
					
					int tellVal = in.read32u(); //
					
					if (m_socket->isOk())
					{
						return tellVal;
					}
					break;
				}
				else if (cmd == EOF_OR_ERROR)
				{
					HK_ON_DEBUG(hkUint32 errorId =) in.read32u(); 
					HK_ASSERT(0x032134, errorId == m_id);
					break; // done already
				}
				else 
				{
					HK_WARN_ALWAYS(0x4234f, "Got crazy stuff from hkFileServerStreamReader tell socket..: " << cmd );
					if (packetSize < 0x100000) 
					{
						char* ignoreData = hkAllocate<char>( packetSize, HK_MEMORY_CLASS_BASE);
						in.readRaw(ignoreData, packetSize - sizeof(int));
						hkDeallocate<char>(ignoreData);
					}
					//loop
				}
			}
		}
	}

	return -1;
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
