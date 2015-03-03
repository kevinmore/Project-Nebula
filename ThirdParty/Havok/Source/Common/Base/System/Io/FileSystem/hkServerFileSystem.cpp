/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/FileSystem/hkServerFileSystem.h>
#include <Common/Base/System/Io/Socket/hkSocket.h>
#if defined(HK_PLATFORM_WIN32) && !defined(HK_PLATFORM_WINRT)
#include <Common/Base/System/Io/Platform/Bsd/hkBsdSocket.h>
#endif

#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>

#include <Common/Base/System/Io/Reader/FileServer/hkFileServerStreamReader.h>
#include <Common/Base/System/Io/Writer/FileServer/hkFileServerStreamWriter.h>
#include <Common/Base/System/Io/Reader/Buffered/hkBufferedStreamReader.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>

//
// Kind of backwards at the moment so that we can choose from the PC side
// what and when to serve to multiple clients from the same PC
// So in effect the hkServerFileSystem is really a client that 
// can only accept from the server, rather than knowing the server ip etc
// and just connecting direct. 
//

hkServerFileSystem::hkServerFileSystem(int port)
: m_mode(DEFAULT), m_connectionPort(port), m_listenSocket(HK_NULL), m_connectSocket(HK_NULL)
{
}

hkServerFileSystem::~hkServerFileSystem()
{
	shutdown();
}

void hkServerFileSystem::closeConnection()
{
	if (m_connectSocket)
	{
		m_connectSocket->close();
		m_connectSocket->removeReference();
		m_connectSocket = HK_NULL;
	}
}

void hkServerFileSystem::shutdown()
{
	closeConnection();

	if (m_listenSocket)
	{
		m_listenSocket->close();
		m_listenSocket->removeReference();
		m_listenSocket = HK_NULL;
	}
}

bool hkServerFileSystem::tryForConnection()
{
	if (!m_listenSocket)
	{
		m_listenSocket = hkSocket::create();
		m_listenSocket->listen(m_connectionPort);
		HK_REPORT("Virtual file system created and will poll on port " << m_connectionPort);
	}

	m_connectSocket = m_listenSocket->pollForNewClient();

#if defined(HK_PLATFORM_WIN32) && !defined(HK_PLATFORM_WINRT)
	if (m_connectSocket)
	{
		((hkBsdSocket*)m_connectSocket)->setBlocking(true);
	}
#endif

	return m_connectSocket != HK_NULL;
}

bool hkServerFileSystem::waitForConnection()
{
	while ( !tryForConnection() ) {  /* should have some sort of force quit.. */ }	
	return m_connectSocket != HK_NULL;
}

void hkServerFileSystem::setMode( Mode m )
{
	m_mode = m;
}

// {
//     int PACKET_SIZE; // num bytes following this int
//	   int CMD;
//     ..
//  }

// File Read open 
//  {
//     uint PACKET_SIZE; // num bytes following this int
//	   uint FILE_READ;
//	   uint filenamelen;
//	   char filename, (null term so can use as is on read)
//  }

hkRefNew<hkStreamReader> hkServerFileSystem::openReader( const char* name, OpenFlags mode )
{
	if (m_mode & VIRTUAL_READ)
	{
		if (m_connectSocket && !m_connectSocket->isOk())
		{
			closeConnection();
		}

		if (!m_connectSocket)
		{ 
			if (m_mode & WAIT_FOR_CONNECT)
			{
				waitForConnection();
			}
			else // single listen
			{
				tryForConnection();
			}
		}

		if (m_connectSocket && m_connectSocket->isOk()) 
		{
			int fileNameLen = hkString::strLen(name);
			hkOArchive out(&m_connectSocket->getWriter());
			out.write32u( fileNameLen + 1 + (3 * sizeof(int)) );
			out.write32u( FILE_READ );
			out.write32u( mode );
			out.write32u( fileNameLen );
			out.writeRaw( name, fileNameLen + 1 );

			// see if it has it
			hkIArchive in(&m_connectSocket->getReader());
			if (m_connectSocket->isOk())
			{
				int packetSize = in.read32u();
				int cmd = in.read32u();
				while (m_connectSocket->isOk())
				{
					if (cmd == ACK)
					{
						int id = in.read32u();
						HK_REPORT("Found " << name << " on server, id(" << id << "), loading..");
						hkFileServerStreamReader* fs = new hkFileServerStreamReader( m_connectSocket, id );
						hkStreamReader* b = new hkBufferedStreamReader(fs, 16*1024); // 4K is default size, so want something a little larger if we are going to put up with net latency in between reads
						fs->removeReference();
						return b;
					}
					else if (cmd == NOT_FOUND)
					{
						HK_REPORT("Cound not find " << name << " on server, looking locally..");
						break;
					}
					else
					{
						HK_WARN_ALWAYS(0xdafa3, "Got crazy stuff from file read socket..: " << cmd );
						if (packetSize < 0x100000) 
						{
							char* ignoreData = hkAllocate<char>( packetSize, HK_MEMORY_CLASS_BASE);
							in.readRaw(ignoreData, packetSize - sizeof(int));
							hkDeallocate<char>( ignoreData);
						
						}
						//loop
					}
				}
			}
		}
	}

	// fallback 
	return hkNativeFileSystem::openReader(name, mode);
}

hkRefNew<hkStreamWriter> hkServerFileSystem::openWriter( const char* name, OpenFlags mode )
{
	if (m_mode & VIRTUAL_WRITE)
	{
		if (m_connectSocket && !m_connectSocket->isOk())
		{
			closeConnection();
		}

		if (!m_connectSocket)
		{
			if (m_mode & WAIT_FOR_CONNECT)
			{
				waitForConnection();
			}
			else // single listen
			{
				tryForConnection();
			}
		}

		if (m_connectSocket && m_connectSocket->isOk()) 
		{
			int fileNameLen = hkString::strLen(name);
			hkOArchive out(&m_connectSocket->getWriter());
			out.write32u( fileNameLen + 1 + (3* sizeof(int)) );
			out.write32u( FILE_WRITE );
			out.write32u( mode );
			out.write32u( fileNameLen );
			out.writeRaw( name, fileNameLen + 1 );

			// see if it has it
			hkIArchive in(&m_connectSocket->getReader());
			if (m_connectSocket->isOk())
			{
				int packetSize = in.read32u();
				int cmd = in.read32u();
				while (m_connectSocket->isOk())
				{
					if (cmd == ACK)
					{
						int id = in.read32u();
						HK_REPORT("Created " << name << " on server, id (" << id << ") opening for write..");
						hkFileServerStreamWriter* fw = new hkFileServerStreamWriter( m_connectSocket, id );
						hkStreamWriter* b = new hkBufferedStreamWriter(fw);
						fw->removeReference();
						return b;
					}
					else if (cmd == NOT_FOUND)
					{
						HK_REPORT("Cound not open " << name << " for write on server, trying locally..");
						break;
					}
					else
					{
						HK_WARN_ALWAYS(0x32534,"Got crazy stuff from file write socket..: " << cmd );
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
	}
	
	// fallback 
	return hkNativeFileSystem::openWriter(name, mode);
}

hkFileSystem::Result hkServerFileSystem::stat( const char* name, hkFileSystem::Entry& entryOut )
{
// 	if (m_mode & VIRTUAL_READ)
// 	{
// 		if (m_connectSocket && !m_connectSocket->isOk())
// 		{
// 			closeConnection();
// 		}
// 
// 		if (!m_connectSocket)
// 		{
// 			if (m_mode & WAIT_FOR_CONNECT)
// 			{
// 				waitForConnection();
// 			}
// 			else // single listen
// 			{
// 				tryForConnection();
// 			}
// 		}
// 
// 		if (m_connectSocket && m_connectSocket->isOk()) 
// 		{
// 			int fileNameLen = hkString::strLen(name);
// 			hkOArchive out(&m_connectSocket->getWriter());
// 			out.write32u( fileNameLen + 1 + (2* sizeof(int)) );
// 			out.write32u( FILE_STAT );
// 			out.write32u( fileNameLen );
// 			out.writeRaw( name, fileNameLen + 1 );
// 
// 			// see if it has it
// 			hkIArchive in(&m_connectSocket->getReader());
// 			if (m_connectSocket->isOk())
// 			{
// 				int packetSize = in.read32u();
// 				int cmd = in.read32u();
// 				while (m_connectSocket->isOk())
// 				{
// 					if (cmd == SEND_STAT)
// 					{
// 						int id = in.read32u();
// 						HK_REPORT("Created " << name << " on server, id (" << id << ") opening for write..");
// 						hkFileServerStreamWriter* fw = new hkFileServerStreamWriter( m_connectSocket, id );
// 						hkStreamWriter* b = new hkBufferedStreamWriter(fw);
// 						fw->removeReference();
// 						return b;
// 					}
// 					else
// 					{
// 						HK_WARN_ALWAYS(0x32534,"Got crazy stuff from file write socket..: " << cmd );
// 						if (packetSize < 0x100000) 
// 						{
// 							char* ignoreData = hkAllocate<char>( packetSize, HK_MEMORY_CLASS_BASE);
// 							in.readRaw(ignoreData, packetSize - sizeof(int));
// 							hkDeallocate<char>(ignoreData);
// 						}
// 						//loop
// 					}
// 				}
// 			}
// 		}
// 	}
	return RESULT_NOT_IMPLEMENTED;
}


namespace
{
	struct ServerIterator : public hkFileSystem::Iterator::Impl
	{
		ServerIterator(const char* dirPath = HK_NULL) : m_idx(-1), m_listing(HK_NULL, dirPath) {}

		virtual bool advance(hkFileSystem::Entry& e)
		{
			if( m_idx+1 < m_listing.getEntries().getSize() )
			{
				m_idx += 1;
				e = m_listing.getEntries()[m_idx];
				return true;
			}

			return false;
		}

		int m_idx;
		hkFileSystem::DirectoryListing m_listing;
	}; 
}

hkRefNew<hkFileSystem::Iterator::Impl> hkServerFileSystem::createIterator( const char* basePath, const char* wildcard )
{
	if (m_connectSocket && !m_connectSocket->isOk())
	{
		closeConnection();
	}

	if (!m_connectSocket)
	{
		if (m_mode & WAIT_FOR_CONNECT)
		{
			waitForConnection();
		}
		else // single listen
		{
			tryForConnection();
		}
	}

	ServerIterator* iter = new ServerIterator(basePath);
	DirectoryListing& listingOut = iter->m_listing;
	listingOut.setFs(this);

	if (m_connectSocket && m_connectSocket->isOk()) 
	{
		int pathLen = hkString::strLen(basePath);
		hkOArchive out(&m_connectSocket->getWriter());
		out.write32u( pathLen + 1 + (2* sizeof(int)) );
		out.write32u( DIR_LIST );
		out.write32u( pathLen );
		out.writeRaw( basePath, pathLen + 1 );

		// see if it has it
		hkIArchive in(&m_connectSocket->getReader());
		if (m_connectSocket->isOk())
		{
			int packetSize = in.read32u();
			int cmd = in.read32u();
			while (m_connectSocket->isOk())
			{
				if (cmd == SEND_DIR_LIST)
				{
					int numEntries = in.read32u();
					HK_REPORT("Found path [" << basePath << "] on server, listing " << numEntries << " items");
					listingOut.clear();
					hkArray<char> nameBuffer;
					for (int ne=0; ne < numEntries; ++ne)
					{
						int nameLen = in.read32u();
						bool isDir = in.read8u() > 0;
						if (nameBuffer.getSize() <= nameLen  )
						{
							nameBuffer.setSize(nameLen + 1);
						}
						in.readRaw(nameBuffer.begin(), nameLen + 1); // incl null term
						if (m_connectSocket->isOk())
						{
							if (isDir)
							{
								listingOut.addDirectory( nameBuffer.begin() );
							}
							else
							{
								listingOut.addFile( nameBuffer.begin(), TimeStamp() /* time */ );
							}
						}
						else 
						{
							break;
						}
					}

					if (m_connectSocket->isOk())
					{
						return iter;
					}
					else
					{
						break; // try local..?
					}
				}
				else if (cmd == NOT_FOUND)
				{
					HK_REPORT("Cound not find dir [" << basePath << "] on server, looking locally..");
					break;
				}
				else
				{
					HK_WARN_ALWAYS(0xdafa3, "Got crazy stuff from listDirectory socket..: " << cmd );
					if (packetSize < 0x100000) 
					{
						char* ignoreData = hkAllocate<char>( packetSize, HK_MEMORY_CLASS_BASE);
						in.readRaw(ignoreData, packetSize - sizeof(int));
						hkDeallocate<char>( ignoreData);
					}
					//loop
				}
			}
		}
	}
	hkNativeFileSystem::listDirectory(basePath, listingOut);
	return iter;
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
