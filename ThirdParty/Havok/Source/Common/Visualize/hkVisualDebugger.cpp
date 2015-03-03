/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkVisualDebugger.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/System/Io/Socket/hkSocket.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Visualize/hkProcessRegisterUtil.h>
#include <Common/Visualize/hkServerProcessHandler.h>
#include <Common/Visualize/hkVersionReporter.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Process/hkDebugDisplayProcess.h>
#include <Common/Visualize/Process/hkRemoteObjectProcess.h>

hkVisualDebugger::hkVisualDebugger(const hkArray<hkProcessContext*>& contexts, const hkVtableClassRegistry* reg )
	:	m_server(HK_NULL),
		s_SuppressPollForNewClients(false),
		m_classReg( reg ),
		m_amTimingFrame(false),
		m_frameTimer("Frame Timer"),
		m_overrideFrameTimeIfZero(true)
{
	for (int c=0; c < contexts.getSize(); ++c)
	{
		addContext( contexts[c] );
	}

	HK_REPORT_SECTION_BEGIN(0x1293adef, "Visual Debugger");
	
	HK_REPORT("VDB Server instance has been created");

	hkProcessRegisterUtil::registerAllCommonProcesses();
	addDefaultProcess(hkDebugDisplayProcess::getName());
	addDefaultProcess("Shapes");
	addDefaultProcess("Mouse Picking");
	addDefaultProcess(hkRemoteObjectProcess::getName());

	if (m_classReg)
		m_classReg->addReference();

	HK_REPORT_SECTION_END();
}

hkVisualDebugger::~hkVisualDebugger()
{
	shutdown();
	// remove from contexts.
	for (int c=0; c < m_contexts.getSize(); ++c)
	{
		m_contexts[c]->setOwner(HK_NULL);
	}
	m_defaultProcesses.clear();
	m_requiredProcesses.clear();

	// release the class reg it was using.
	if (m_classReg)
		m_classReg->removeReference();

}

void hkVisualDebugger::addContext( hkProcessContext* context )
{
	HK_WARN_ON_DEBUG_IF( m_clients.getSize() > 0, 0x47cf27af, "Trying to add a context after a Visual Debugger client is connected. This probably won't work." );
	context->setOwner(this);
	m_contexts.pushBack(context);
}

void hkVisualDebugger::shutdown()
{
	// cleanup existing clients
	HK_REPORT_SECTION_BEGIN(0x1293adef, "Shutting down Visual Debugger..");
	for(int i = (m_clients.getSize() - 1); i >= 0; i--)
	{
		// Send a step command to make sure that this frame gets processed
		writeStep(i, 0);

		// delete it
		deleteClient(i);
		HK_REPORT("Client deleted.");
	}

	if(m_server)
	{
		delete m_server;
		m_server = HK_NULL;
		HK_REPORT("Server deleted.");
	}
	HK_REPORT_SECTION_END();
}

void hkVisualDebugger::serve(int listenPort)
{
	HK_REPORT_SECTION_BEGIN(0x1293ade8, "Serving");

	if(!m_server)
	{
		m_server = hkSocket::create();
		hkResult res = HK_FAILURE;
		if(m_server)
		{
			res = m_server->listen(listenPort);
			if (res == HK_SUCCESS)
			{
				HK_REPORT("Server created and will poll for new client(s) on port " << listenPort << " every frame");
			}
			else
			{
				m_server->removeReference();
				m_server = HK_NULL;
			}
		}
		
		if (res != HK_SUCCESS)
		{
			HK_REPORT("Server could not be created, please check that you platform supports sockets with the hkBase library");
		}
	}
	else
	{
		HK_REPORT("Server has already been created, only one server allowed per visual debugger instance");
	}

	HK_REPORT_SECTION_END();
}

void hkVisualDebugger::createRequiredAndDefaultProcessList( hkVisualDebuggerClient& vdbClient, hkStringBuf& viewerNames )
{
	viewerNames = "Turning on the following viewers: [";

	// Create required processes first
	for( int j = 0; j < m_requiredProcesses.getSize(); j++ )
	{
		int tag = vdbClient.m_processHandler->getProcessId( m_requiredProcesses[j] );
		if (tag >= 0 )
		{
			viewerNames.appendJoin(" ", m_requiredProcesses[j]);
			vdbClient.m_processHandler->createProcess(tag);
		}
	}

	// Create default processes next
	for(int i = 0; i < m_defaultProcesses.getSize(); i++)
	{
		int tag = vdbClient.m_processHandler->getProcessId( m_defaultProcesses[i] );
		
		// Only add processes that were not created already
		if (tag >= 0 && m_requiredProcesses.indexOf( m_defaultProcesses[i] ) == -1)
		{
			viewerNames.appendJoin(" ", m_defaultProcesses[i]);
			vdbClient.m_processHandler->createProcess(tag);
		}
	}

	viewerNames += "]";
}

void hkVisualDebugger::capture(const char* captureFilename)
{
	HK_REPORT_SECTION_BEGIN(0x1293ade7, "Capturing");

	hkRefPtr<hkStreamWriter> writer = hkFileSystem::getInstance().openWriter(captureFilename);
	if(writer && writer->isOk())
	{
		HK_REPORT( "Capturing simulation state to \'" << captureFilename << "\'" );

		createClient( HK_NULL, HK_NULL, writer );

		// At this point, the ids have been placed in the stream to be selected 
		// by default by the client, but as it is an inert file,
		// we will go ahead and confirm them all
		hkVisualDebuggerClient& vdbC = m_clients.back();
		hkStringBuf viewerNames;
		createRequiredAndDefaultProcessList( vdbC, viewerNames );
		
		HK_REPORT( viewerNames.cString() );
	}
	else
	{
		HK_REPORT( "Capture file \'" << captureFilename << "\' could not be opened for writing" );
	}

	HK_REPORT_SECTION_END();
}

void hkVisualDebugger::capture(hkMemoryTrack* outgoingMemory, hkMemoryTrack* incommingMemory)
{	
	HK_REPORT_SECTION_BEGIN(0xbebefec4, "Capturing to memory stream");
	
	hkStreamWriter* writer = new hkMemoryTrackStreamWriter(outgoingMemory, hkMemoryTrackStreamWriter::TRACK_BORROW);
	hkStreamReader* reader = new hkMemoryTrackStreamReader(incommingMemory, hkMemoryTrackStreamReader::MEMORY_INPLACE, true);
	if(writer && writer->isOk() && reader && reader->isOk())
	{
		HK_REPORT( "Capturing simulation state to memory stream." );

		createClient( HK_NULL, reader, writer );
		writer->removeReference();
		reader->removeReference();

		// At this point, the ids have been placed in the stream to be selected 
		// by default by the client, but as it is an inert file,
		// we will go ahead and confirm them all
		hkVisualDebuggerClient& vdbC = m_clients.back();
		hkStringBuf viewerNames;
		createRequiredAndDefaultProcessList( vdbC, viewerNames );

		HK_REPORT( viewerNames.cString() );

		// Flush the client's output stream so all data is sent after connection
		vdbC.m_processHandler->m_outStream->getStreamWriter()->flush();
	}
	else
	{
		HK_REPORT( "Could not create memory stream for writing." );
	}

	HK_REPORT_SECTION_END();
}

void hkVisualDebugger::endCapture()
{
	int nc = m_clients.getSize();
	for(int i = (nc - 1); i >= 0; --i) // backwards as they may be deleted.
	{
		hkVisualDebuggerClient& client = m_clients[i];
		if (!client.m_socket) // then must be a file based capture
		{
			deleteClient(i);
		}
	}
}

void hkVisualDebugger::deleteClient(int i)
{
	hkVisualDebuggerClient& client = m_clients[i];
	if(client.m_processHandler)
	{
		client.m_processHandler->removeReference();
	}
	if(client.m_socket)
	{
		client.m_socket->removeReference();
	}
	m_clients.removeAt(i);
}

void hkVisualDebugger::createClient( hkSocket* socket, hkStreamReader* reader, hkStreamWriter* writer )
{
	// send version information (the first thing in the stream back to Client / file)
	if (writer) 
	{ 
		hkVersionReporter::sendStructureLayout(writer);
		hkVersionReporter::sendVersionInformation(writer);
	}

	hkVisualDebuggerClient* newClient = m_clients.expandBy(1);
	newClient->m_socket = socket;
	newClient->m_processHandler = new hkServerProcessHandler(m_contexts, reader, writer);

	// Register available processes to the client
	newClient->m_processHandler->registerAllAvailableProcesss();

	// Create the ones we require
	for(int i = 0; i < m_requiredProcesses.getSize(); i++)
	{
		int tag = newClient->m_processHandler->getProcessId( m_requiredProcesses[i].cString() );
		if (tag >= 0)
		{
			newClient->m_processHandler->createProcess(tag);
			newClient->m_processHandler->selectProcess(tag);
		}
	}

	for(int i = 0; i < m_defaultProcesses.getSize(); i++)
	{
		int tag = newClient->m_processHandler->getProcessId( m_defaultProcesses[i].cString() );
		if (tag >= 0  && m_requiredProcesses.indexOf( m_defaultProcesses[i] ) == -1 )
		{
			newClient->m_processHandler->selectProcess(tag);
		}
	}

	// Step one frame.
	writeStep( m_clients.getSize() - 1, 0 );
}

void hkVisualDebugger::SupressPollForNewClient(bool val)
{
	s_SuppressPollForNewClients = val;
}

void hkVisualDebugger::pollForNewClients()
{
	if (s_SuppressPollForNewClients)
		return;

	if (m_amTimingFrame && m_frameTimer.isRunning())
		m_frameTimer.stop();	

	// see if there is a new client trying to connect
	if(m_server)
	{
		hkSocket* socket = m_server->pollForNewClient();
		if(socket)
		{
			HK_REPORT("A new network client has been received (host name not available at present)");
			hkStreamWriter* writer = &socket->getWriter();
			hkStreamReader* reader = &socket->getReader();
			createClient( socket, reader, writer );
		}
	}

	if (m_amTimingFrame)
		m_frameTimer.start();
}

void hkVisualDebugger::writeStep(int i, float t)
{
	hkVisualDebuggerClient& client = m_clients[i];
	if( client.m_processHandler->isOk() )
	{
		hkDisplaySerializeOStream* os = client.m_processHandler->m_outStream;
		os->write32u( sizeof(hkUint8) + sizeof(float) ); // cmd size
		os->write8u( hkVisualDebuggerProtocol::COMMAND_STEP );// cmd 
		os->writeFloat32( t ); // cmd data
		os->getStreamWriter()->flush(); // end of frame, so flush
	}
}

void hkVisualDebugger::step(hkReal frameTimeInMs)
{
	hkReal frameTime = 16.0f; // (this is an approximation for the first frame)
	if (m_amTimingFrame)
	{
		frameTime = m_frameTimer.getElapsedSeconds() * 1000.0f;
		m_frameTimer.stop();
	}
	m_amTimingFrame = true;
	if(hkMath::fabs(frameTimeInMs) > HK_REAL_EPSILON || !m_overrideFrameTimeIfZero)
	{
		frameTime = frameTimeInMs;
	}

	// Check connection status.
	pollForNewClients();

	// step all clients
	int nc = m_clients.getSize();
	for(int i = (nc - 1); i >= 0; --i) // backwards as they may be deleted.
	{
		hkVisualDebuggerClient& client = m_clients[i];
		if( client.m_processHandler->isOk() )
		{
			client.m_processHandler->step(frameTime);

			writeStep(i, hkFloat32(frameTime)); // a 0x00 command followed by the float32 for time
		}
		
		if( !client.m_processHandler->isOk() )
		{
			HK_REPORT_SECTION_BEGIN(0x76e3a642, "Client Dies");
			HK_REPORT("Client has died, cleaning up (host name not available at present)");
			HK_REPORT_SECTION_END();
			deleteClient(i); // not much use
		}
	}
	
	if (m_amTimingFrame)
	{
		m_frameTimer.reset();
		m_frameTimer.start();
	}
}

void hkVisualDebugger::clearDefaultAndRequiredProcesses()
{
	m_defaultProcesses.clear();
	m_requiredProcesses.clear();
}

void hkVisualDebugger::addDefaultProcess(const char* processName)
{
	m_defaultProcesses.expandOne() = processName;
}

void hkVisualDebugger::removeDefaultProcess(const char* processName)
{
	for(int i = 0; i < m_defaultProcesses.getSize(); i++)
	{
		if(hkString::strCmp(m_defaultProcesses[i].cString(), processName) == 0)
		{
			// Keep the processes in the same order in case they have dependencies 
			// (eg, hkRemoteObjectProcess must be created before hkbBehaviorServer).
			m_defaultProcesses.removeAtAndCopy(i);
			return;
		}
	}

	HK_REPORT_SECTION_BEGIN(0x76565454, "removeDefaultProcess");
	HK_REPORT("The default Process'" << processName << "', cannot not be removed from the default process list as it cannot be found!");
	HK_REPORT_SECTION_END();
}

void hkVisualDebugger::addRequiredProcess(const char* processName)
{
	m_requiredProcesses.expandOne() = processName;
}

void hkVisualDebugger::addTrackedObject(void* obj, const hkClass& klass, const char* group, hkUlong tagId )
{
	hkVisualDebuggerTrackedObject& to = m_trackedObjects.expandOne();
	to.m_ptr = obj;
	to.m_class = &klass;
	to.m_tag = tagId;
	for ( int c=0; c < m_trackCallbacks.getSize(); ++c)
	{
		(m_trackCallbacks[c])( obj, &klass, true, tagId, m_trackCallbackHandles[c] );
	}
}

void hkVisualDebugger::removeTrackedObject(void* obj)
{
	for( int ti=0; ti < m_trackedObjects.getSize(); ++ti)
	{
		if (m_trackedObjects[ti].m_ptr == obj)
		{
			m_trackedObjects.removeAt(ti);
			for ( int c=0; c < m_trackCallbacks.getSize(); ++c)
			{
				(m_trackCallbacks[c])( obj, HK_NULL, false, 0, m_trackCallbackHandles[c]);
			}
			return;
		}
	}
}

void hkVisualDebugger::addTrackedObjectCallback( hkVisualDebuggerTrackedObjectCallback callback, void* userHandle)
{
	m_trackCallbacks.pushBack( callback );
	m_trackCallbackHandles.pushBack( userHandle );
}

void hkVisualDebugger::removeTrackedObjectCallback( hkVisualDebuggerTrackedObjectCallback callback )
{
	int i = m_trackCallbacks.indexOf( callback );
	if (i >=0 )
	{
		m_trackCallbacks.removeAt(i);
		m_trackCallbackHandles.removeAt( i );
	}
}

void hkVisualDebugger::getCurrentProcesses( hkArray< hkProcess* >& process ) const
{
	int nc = m_clients.getSize();
	for(int i = 0; i < nc; ++i) 
	{
		const hkVisualDebuggerClient& client = m_clients[i];
		if (client.m_processHandler)
		{	
			const hkArray<hkProcess*>& proc = client.m_processHandler->getProcessList();
			process.insertAt( process.getSize(), proc.begin(), proc.getSize());
		}
	}
}

void hkVisualDebugger::removeContext( hkProcessContext* context )
{
	int index = m_contexts.indexOf(context);
	if (index >= 0)
	{
		m_contexts.removeAtAndCopy(index);
	}
}

hkProcess* hkVisualDebugger::getCurrentProcessByName( const char* name ) const
{	
	int id = hkProcessFactory::getInstance().getProcessId(name);	
	if( id != -1 )
	{
		hkArray<hkProcess*> processes;
		getCurrentProcesses(processes);

		for( int i = 0; i < processes.getSize(); ++i )
		{
			if( processes[i]->getProcessTag() == id )
			{
				return processes[i];
			}
		}
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
