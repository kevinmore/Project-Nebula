/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkBaseTypes.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkNetworkedDeterminismUtil.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>

hkNetworkedDeterminismUtil* hkNetworkedDeterminismUtil::s_instance = HK_NULL;

hkNetworkedDeterminismUtil::hkNetworkedDeterminismUtil(const char* hostname, int port)
{
	m_serverAddress = hostname;
	m_serverPort = port;

	m_server = HK_NULL;
	m_client = HK_NULL;
}

hkNetworkedDeterminismUtil::~hkNetworkedDeterminismUtil()
{
	delete m_server; m_server = HK_NULL;
	delete m_client; m_server = HK_NULL;
}

void HK_CALL hkNetworkedDeterminismUtil::create(const char* host, int port)
{
	HK_ASSERT2(0xad903091, ! s_instance, "An instance already created.");

#if defined (HK_ENABLE_NETWORKED_DETERMINISM_UTIL)
	s_instance = new hkNetworkedDeterminismUtil(host, port);

	if ( ! s_instance->tryToCreateClient() )
	{
		s_instance->createServer();
	}
	hkCheckDeterminismUtil::createInstance();

#endif
}


void HK_CALL hkNetworkedDeterminismUtil::destroy()
{
#if defined (HK_ENABLE_NETWORKED_DETERMINISM_UTIL)
	HK_ASSERT2(0xad903091, s_instance, "An instance does not exist.");

	delete s_instance;
	s_instance = HK_NULL;

	hkCheckDeterminismUtil::destroyInstance();
#endif
}


hkNetworkedDeterminismUtil::Server::Server(int listeningPort, int maxNumClients)
{
	m_listeningSocket = hkSocket::create();
	m_maxNumClients = maxNumClients;

	HK_ASSERT2(0xad903092, m_listeningSocket, "Socket not created.");

	if (m_listeningSocket)
	{
		m_listeningSocket->listen(listeningPort);
		HK_REPORT("hkNetworkedDeterminismUtil::Server created and will poll for new client(s) on port " << listeningPort << " every frame");
	}
	else
	{
		HK_REPORT("hkNetworkedDeterminismUtil::Server could not be created, please check that you platform supports sockets with the hkBase library");
	}
}

hkNetworkedDeterminismUtil::Server::~Server()
{
	delete m_listeningSocket;

	for (int ci = 0; ci < m_clients.getSize(); ci++)
	{
		delete m_clients[ci];
	}
	m_clients.clearAndDeallocate();
}

hkNetworkedDeterminismUtil::Client::Client(hkSocket* socket)
{
	m_socket = socket;
}

hkNetworkedDeterminismUtil::Client::~Client()
{
	delete m_socket;
}


void hkNetworkedDeterminismUtil::Server::pollForNewClients()
{
	if (m_listeningSocket && m_clients.getSize() < m_maxNumClients)
	{
		hkSocket* newClient = m_listeningSocket->pollForNewClient();
		if (newClient)
		{
			m_clients.pushBack(newClient);
		}
	}
}

void hkNetworkedDeterminismUtil::Server::sendCommand(const Command& command)
{
	for (int i = 0; i < m_clients.getSize(); i++)
	{
		hkSocket* client = m_clients[i];

		client->getWriter().write(&command.m_type, sizeof(command.m_type));
		const int size = command.m_data.getSize();
		client->getWriter().write(&size, sizeof(size));
		client->getWriter().write(command.m_data.begin(), size);
		client->getWriter().flush();
	}
}

void hkNetworkedDeterminismUtil::Server::synchronizeWithClients()
{
	// Wait for 4 bytes from each client.
	//
	// Todo: timeouts.

	for (int i = 0; i < m_clients.getSize(); i++)
	{
		// use time out ...
		char buffer[4];
		if (HK_FAILURE == readFromSocket(m_clients[i], buffer, 4))
		{
			m_clients[i]->close();
			m_clients.removeAtAndCopy(i);
			i--;
		}
	}
}

void hkNetworkedDeterminismUtil::Client::sendSynchronizationBytes()
{
	char buffer[4];
	writeToSocket(m_socket, buffer, 4);

	// handle socket closing ..


}

void hkNetworkedDeterminismUtil::Client::processCommands(hkNetworkedDeterminismUtil::Command::Type expectedCommandType)
{
	hkArray<hkUint8> buffer;
	buffer.setSize(4);

	// Read command type
	/* hkResult socketResult =*/ readFromSocket(m_socket, buffer.begin(), 4);
	HK_ON_DEBUG( int type = *(int*)(buffer.begin()) );

	HK_ASSERT2(0xad903131, expectedCommandType == type, "Unexpected command type.");

	// Read data size
	HK_ON_DEBUG( hkResult socketResult = ) readFromSocket(m_socket, buffer.begin(), 4);
	int dataSize = *(int*)(buffer.begin());

	// Read actual data
	//
	buffer.setSize(dataSize);
	HK_ON_DEBUG( socketResult = ) readFromSocket(m_socket, buffer.begin(), dataSize);
	HK_ON_DEBUG( socketResult = socketResult ); // silence compiler warning
}



hkNetworkedDeterminismUtil::ControlCommand::ControlCommand(const void* buffer, int size)
{
	m_type = TYPE_CONTROL;
	m_data.setSize( size );
	hkString::memCpy(m_data.begin(), buffer, size);
}


hkNetworkedDeterminismUtil::DeterminismDataCommand::DeterminismDataCommand(const char* buffer, int bufferSize)
{
	m_type = TYPE_DETERMINISM_DATA;

	m_data.setSize(bufferSize);
	hkString::memCpy(m_data.begin(), buffer, bufferSize);
}


void hkNetworkedDeterminismUtil::startStepDemoImpl(ControlCommand& controlCommand)
{
	if (m_server)
	{
		// Cache control command
		m_controlCommand = controlCommand;

		hkCheckDeterminismUtil::getInstance().startWriteMode(HK_NULL);
	}
	if (m_client)
	{
		// Receive control command
		char buffer[2048];
		readFromSocket(m_client->m_socket, buffer, 8);
		HK_ASSERT2(0xad903252, ((int*)buffer)[0] == Command::TYPE_CONTROL, "Unexpected command.");
		int dataSize = ((int*)buffer)[1];
		controlCommand.m_data.setSize(dataSize);
		readFromSocket(m_client->m_socket, controlCommand.m_data.begin(), dataSize);

		// Receive determinism data command
		readFromSocket(m_client->m_socket, buffer, 8);
		HK_ASSERT2(0xad903252, ((int*)buffer)[0] == Command::TYPE_DETERMINISM_DATA, "Unexpected command.");
		dataSize = ((int*)buffer)[1];

		// Populate hkCheckDeterminismUtil in write mode.
		hkCheckDeterminismUtil::getInstance().startWriteMode(HK_NULL);
		while (dataSize > 0)
		{
			int thisSize = hkMath::min2( dataSize, 2048 );
			readFromSocket( m_client->m_socket, buffer, thisSize );
			g_checkDeterminismUtil->m_memoryTrack->write( buffer, thisSize );
			dataSize -= thisSize;

		}

		// Set member m_memoryTrack to null it so that it doesn't get deleted in finish().
		hkMemoryTrack* memoryTrack = hkCheckDeterminismUtil::getInstance().m_memoryTrack;
		hkCheckDeterminismUtil::getInstance().m_memoryTrack = HK_NULL; 
		hkCheckDeterminismUtil::getInstance().finish();

		// Start comparing mode.
		hkCheckDeterminismUtil::getInstance().m_memoryTrack = memoryTrack;
		hkCheckDeterminismUtil::getInstance().startCheckMode(HK_NULL);
	}

//	const bool isPrimaryThread = true;
//	hkCheckDeterminismUtil::getInstance().workerThreadStartFrame(isPrimaryThread);
}

void hkNetworkedDeterminismUtil::endStepDemoImpl()
{
//	hkCheckDeterminismUtil::workerThreadFinishFrame();

	if (m_server)
	{
		// check for new clients
		m_server->pollForNewClients();

		// send cached control to all clients
		m_server->sendCommand(m_controlCommand);

		// Send determinism data
		DeterminismDataCommand determinismDataCommand(0, 0);
		int dataSize = hkCheckDeterminismUtil::getInstance().m_memoryTrack->getSize();
		determinismDataCommand.m_data.setSize(dataSize);

		HK_ASSERT2(0xad903244, hkCheckDeterminismUtil::getInstance().m_memoryTrack != HK_NULL, "The hkNetworkedDeterminismUtil assumes the determinsm check util to use a hkMemoryTrack.");
		int offset = 0;

		hkArray<hkUint8*>& sectors = hkCheckDeterminismUtil::getInstance().m_memoryTrack->m_sectors;
		int sectorSize = hkCheckDeterminismUtil::getInstance().m_memoryTrack->m_numBytesPerSector;
		for (int si = 0; si < sectors.getSize()-1; si++)
		{
			hkString::memCpy(determinismDataCommand.m_data.begin()+offset, sectors[si], sectorSize);
			offset += sectorSize;
		}
		if (sectors.getSize())
		{
			hkString::memCpy(determinismDataCommand.m_data.begin()+offset, sectors.back(), hkCheckDeterminismUtil::getInstance().m_memoryTrack->m_numBytesLastSector);
			offset += hkCheckDeterminismUtil::getInstance().m_memoryTrack->m_numBytesLastSector;
		}
		HK_ASSERT2(0xad903246, offset == dataSize, "Data written doesn't match hkMemoryTrack's size.");

		hkCheckDeterminismUtil::getInstance().finish();

		m_server->sendCommand(determinismDataCommand);

	}

	if (m_client)
	{
		hkCheckDeterminismUtil::getInstance().finish();
	}
}

bool hkNetworkedDeterminismUtil::tryToCreateClient()
{
	HK_ASSERT2(0xad903093, ! m_client, "Client already created. To reconnect to a server destroy the existing client first.");

	hkSocket* socket = hkSocket::create();

	if (socket)
	{	
		if (HK_FAILURE == socket->connect(m_serverAddress.cString(), m_serverPort))
		{
			delete socket;
			return false;
		}
		
		// Create client with the socket.
		m_client = new Client(socket);
		return true;
	}
	
	return false;
}

void hkNetworkedDeterminismUtil::createServer()
{
	int maxNumClients = 1;
	m_server = new Server(m_serverPort, maxNumClients);
}

hkResult hkNetworkedDeterminismUtil::readFromSocket(hkSocket* socket, void* buffer, int size)
{
	const int fullSize = size;
	while(size)
	{
		if (!socket->isOk()) { break; }
		size -= socket->getReader().read(hkAddByteOffset(buffer, fullSize-size), size);
	}

	hkResult result = size == 0 ? HK_SUCCESS : HK_FAILURE;
	HK_ASSERT2(0xad930241, result == HK_SUCCESS, "Socket broke.");
	return result;
}

hkResult hkNetworkedDeterminismUtil::writeToSocket(hkSocket* socket, void* buffer, int size)
{
	int sizeWritten = socket->getWriter().write(buffer, size);
	HK_ASSERT2(0xad90324a, size == sizeWritten, "Not all data written to socket. Communication will now hang.");

	return size == sizeWritten ? HK_SUCCESS : HK_FAILURE;
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
