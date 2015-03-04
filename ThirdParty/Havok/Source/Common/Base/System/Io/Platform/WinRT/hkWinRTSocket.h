/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_BASE_WINRT_SOCKET_H
#define HK_BASE_WINRT_SOCKET_H

#include <Common/Base/System/Io/Socket/hkSocket.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <vector>

class hkWinRTSocket : public hkSocket
{
	
	public:
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS );

		hkWinRTSocket( Windows::Networking::Sockets::StreamSocket^ s = nullptr );

		virtual ~hkWinRTSocket();

		virtual hkBool isOk() const;

		virtual void close();

		virtual int read( void* buf, int nbytes);

		virtual int write( const void* buf, int nbytes);

		// client

		virtual hkResult connect(const char* servername, int portNumber);

		virtual hkResult asyncSelect(void* notificationHandle, hkUint32 message, SOCKET_EVENTS events);

		virtual bool canRead();
		

		// server

		hkResult listen(int port);
		hkSocket* pollForNewClient();


		// Switch between blocking/non-blocking modes if desired
		hkResult setBlocking(hkBool blocking);
		

	protected:

		ref class SocketEventHandler
		{
			public:
				void OnNewConnection(Windows::Networking::Sockets::StreamSocketListener^ l,  Windows::Networking::Sockets::StreamSocketListenerConnectionReceivedEventArgs^ args );
			
			internal:
				SocketEventHandler( hkWinRTSocket* parent ) : m_parent(parent) { }
				hkWinRTSocket* m_parent;
		};

		friend ref class SocketEventHandler;


		hkResult createSocket();

		Windows::Networking::Sockets::StreamSocket^ m_socket;
		Windows::Networking::Sockets::StreamSocketListener^ m_listener;

		Windows::Storage::Streams::DataReader^ m_socketReader;
		Windows::Storage::Streams::DataWriter^ m_socketWriter;

		SocketEventHandler^ m_eventHandler;

		hkCriticalSection m_newSocketsLock;
		std::vector<Windows::Networking::Sockets::StreamSocket^> m_newSockets; 
		
};


#endif // HK_BASE_WINRT_SOCKET_H

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
