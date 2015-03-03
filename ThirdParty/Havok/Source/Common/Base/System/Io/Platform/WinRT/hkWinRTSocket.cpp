/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/WinRT/hkWinRTSocket.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Thread/Thread/WinRT/hkWinRTThreadUtils.h>
#include <Common/Base/Container/String/hkUtf8.h>

// sockets, so say <winsock2.h> are now Desktop App only.
// All network access from a Metro app has to go through the WinRT api 
using namespace Windows::Networking::Sockets;
using namespace Concurrency;
	
#include <Strsafe.h> // Metro app  allowed string funcs

#include <locale>
#include <codecvt>




hkWinRTSocket::hkWinRTSocket (StreamSocket^ s )
	: m_socket(s), m_listener(nullptr), m_eventHandler(nullptr), m_newSocketsLock(10000)
{
	if (s != nullptr)
	{
		m_socketReader = ref new DataReader(m_socket->InputStream);
		m_socketWriter = ref new DataWriter(m_socket->OutputStream);
	}
}

hkBool hkWinRTSocket::isOk() const
{
	if (m_socket != nullptr)
	{
		return true;
	}
	return false;
}

void hkWinRTSocket::close()
{
	m_eventHandler = nullptr;
	m_listener = nullptr;	
	m_socketReader = nullptr;
	m_socketWriter = nullptr;
	m_socket = nullptr;
}

hkResult hkWinRTSocket::createSocket()
{
	close();
	
	try { 
		m_socket = ref new StreamSocket();
		StreamSocketControl^ ctrl = m_socket->Control;
		ctrl->KeepAlive = false;
		//ctrl->NoDelay = 
		//ctrl->OutboundBufferSizeInBytes
		ctrl->QualityOfService = SocketQualityOfService::Normal; // or ::LowLatency

		m_socketReader = ref new DataReader(m_socket->InputStream);
		m_socketWriter = ref new DataWriter(m_socket->OutputStream);
		
		return HK_SUCCESS;
	}
	catch (...) // Have to catch all as code above this catches none
	{
		//::OutputDebugString(e->Message->Data());
	}
	return HK_FAILURE;
}

hkWinRTSocket::~hkWinRTSocket()
{
	close();
}


int hkWinRTSocket::read(void* buf, int nbytes)
{
	if(m_socket != nullptr)
	{
		try 
		{ 
			task<Havok::ByteArray> readTask = Havok::readDataAsync(m_socketReader, nbytes);
			Platform::Array<byte>^ readData;
			if ( Havok::taskWaitByteArray( readTask, readData ) && 
			    ( readData != nullptr) && (readData->Length > 0) )
			{
				// Todo, cut out extra copy here, as we are sync we could copy direct into buf[] within task
				hkString::memCpy(buf, readData->Data, readData->Length );
				return readData->Length;
			}
		}
		catch (...) // Have to catch all as code above this catches none
		{
			//::OutputDebugString(e->Message->Data());
		}
		close();
	}
	return 0;
}

int hkWinRTSocket::write( const void* buf, int nbytes)
{
	if(m_socket != nullptr)
	{
		try { 

			Platform::Array<byte>^ ibuf = ref new Platform::Array<byte>( (byte*) buf, nbytes );
			m_socketWriter->WriteBytes( ibuf );
			task<unsigned int> writeAction( m_socketWriter->StoreAsync() );
			unsigned int r; 
			if ( Havok::taskWaitUint( writeAction, r ) ) 
			{
				return (int)r;
			}
		}
		catch (...) // Have to catch all as code above this catches none
		{
			//::OutputDebugString(e->Message->Data());
		}
		close();
	}
	return 0;
}

static hkBool HK_CALL hkIsDigit(int c)
{
	return c >= '0' && c <= '9';
}

hkResult hkWinRTSocket::connect(const char* servername, int portNumber)
{
	if (m_socket == nullptr)
	{
		if (createSocket() != HK_SUCCESS )
		{
			return HK_FAILURE;
		}
	}

	try { 
		Platform::String^ hostNameStr = ref new Platform::String( hkUtf8::WideFromUtf8(servername).cString() );
		Windows::Networking::HostName^ hn = ref new Windows::Networking::HostName( hostNameStr );
		wchar_t wps[64]; StringCbPrintf(wps, 64, L"%d", portNumber); // wsprintf not allowed in Metro, StringCb* funcs are the replacement
		Platform::String^ portNameStr = ref new Platform::String( wps );
	
		task<void> connectAction( m_socket->ConnectAsync( hn, portNameStr, Windows::Networking::Sockets::SocketProtectionLevel::PlainSocket ) );
		if ( Havok::taskWaitVoid( connectAction ) )
		{
			OutputDebugString(wps);
			return HK_SUCCESS; // assume it will be ok, as would have jumped out with Exception if not
		}
	}
	catch (...) // Have to catch all as code above this catches none
	{
		//::OutputDebugString(e->Message->Data());
	}
	HK_WARN(0x46d25e96, "Cannot connect to server!");
	close();
	return HK_FAILURE;
}

hkResult hkWinRTSocket::asyncSelect(void* notificationHandle, hkUint32 message, SOCKET_EVENTS events)
{
	return HK_FAILURE;
}

hkResult hkWinRTSocket::listen(int port)
{
	if (m_listener != nullptr) // already listening..
	{
		HK_WARN(0x46d25e96, "Calling listen more than once on same socket");
		return HK_FAILURE;
	}

	try	{
		m_listener = ref new StreamSocketListener();
		m_eventHandler = ref new hkWinRTSocket::SocketEventHandler(this);
		m_listener->ConnectionReceived += ref new TypedEventHandler< StreamSocketListener^,  StreamSocketListenerConnectionReceivedEventArgs^ >(m_eventHandler, &hkWinRTSocket::SocketEventHandler::OnNewConnection );
		wchar_t wps[64]; StringCbPrintf(wps, 64, L"%d", port); // wsprintf not allowed in Metro, StringCb* funcs are the replacement
		Platform::String^ portNameStr = ref new Platform::String( wps );
		
		task<void> bindAction( m_listener->BindServiceNameAsync(portNameStr) );
		if ( Havok::taskWaitVoid( bindAction ) )
		{
			OutputDebugStringW(m_listener->Information->LocalPort->Data());
			return HK_SUCCESS; // assume it will be ok, as would have jumpped out with Exception if not
		}
	}
	catch (...) // Have to catch all as code above this catches none
	{
		//::OutputDebugStringW(e->Message->Data());
	}
	return HK_FAILURE;

}

void hkWinRTSocket::SocketEventHandler::OnNewConnection(Windows::Networking::Sockets::StreamSocketListener^ l,  Windows::Networking::Sockets::StreamSocketListenerConnectionReceivedEventArgs^ args )
{
	OutputDebugString(L"NEW CONNECT!");
	m_parent->m_newSocketsLock.enter();
	m_parent->m_newSockets.push_back( args->Socket );
	m_parent->m_newSocketsLock.leave();
}
			

bool hkWinRTSocket::canRead()
{
	if (m_socket != nullptr)
	{
		//return numHits > 0;
	}
	return false;
}

hkSocket* hkWinRTSocket::pollForNewClient()
{
	HK_ASSERT2( 0x73993156, m_listener != nullptr, "Calling pollForNewClient on socket without a listener");

	// poll the listener socket for new client sockets
	hkWinRTSocket* s = HK_NULL;
	m_newSocketsLock.enter();
	if ( m_newSockets.size() > 0 )
	{
		s = new hkWinRTSocket( m_newSockets[m_newSockets.size() - 1] );
		m_newSockets.pop_back();
	}
	m_newSocketsLock.leave();
	return s;
}

hkResult hkWinRTSocket::setBlocking(hkBool blocking)
{
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
