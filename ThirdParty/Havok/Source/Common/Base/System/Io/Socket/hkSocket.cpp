/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Socket/hkSocket.h>

#if defined(HK_PLATFORM_PSP)

// PSP(R) (PlayStation(R)Portable) not currently supported 
#	include <Common/Base/System/Io/Platform/Psp/hkPspSocket.h>
	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkPspSocket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = hkPspNetworkInit;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = hkPspNetworkQuit;
	
#elif defined(HK_PLATFORM_PSVITA)

#	include <Common/Base/System/Io/Platform/PsVita/hkPsVitaSocket.h>
	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkPsVitaSocket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = hkPsVitaNetworkInit;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = hkPsVitaNetworkQuit;
	
#elif defined(HK_PLATFORM_PS3_PPU)

// PlayStation(R)3 socket implementation
#	include <Common/Base/System/Io/Platform/Ps3/hkPs3Socket.h>
	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkPs3Socket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = hkPs3NetworkInit;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = hkPs3NetworkQuit;

#elif defined(HK_PLATFORM_WIIU)
	// Deprecated: Use HIO for the WiiU connection. TCP/IP is recommended
	#if defined(HK_USE_DEPRECATED_HIO_CONNECTION)
	#	include <Common/Base/System/Io/Platform/WiiU/hkWiiuSocket.h>

	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkWiiuSocket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = hkWiiuNetworkInit;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = hkWiiuNetworkQuit;

	#else // !defined(HK_DEPRECATED_HIO_CONNECTION)
	// Default behaviour is to use TCP/IP sockets on Wii U
	#include <Common/Base/System/Io/Platform/Bsd/hkBsdSocket.h>
	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkBsdSocket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = hkBsdNetworkInit;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = hkBsdNetworkQuit;
	#endif // !defined(HK_DEPRECATED_HIO_CONNECTION)

#elif defined(HK_PLATFORM_WINRT) 

#	include <Common/Base/System/Io/Platform/WinRT/hkWinRTSocket.h>
	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkWinRTSocket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = HK_NULL;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = HK_NULL;


#elif defined(HK_PLATFORM_GC) 

// GameCube not supported 
	static hkSocket* HK_CALL hkSocketCreate()
	{
		HK_WARN(0x1f65b352, "No socket implementation for this platform");
		return HK_NULL;
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = HK_NULL;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = HK_NULL;

#elif defined(HK_PLATFORM_NACL)

// Google Native Client, none for now
	static hkSocket* HK_CALL hkSocketCreate()
	{
		HK_WARN(0x1f65b352, "No socket implementation for this platform");
		return HK_NULL;
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = HK_NULL;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = HK_NULL;

#elif defined(HK_PLATFORM_PS4)

// PlayStation(R)4 socket implementation
#	include <Common/Base/System/Io/Platform/Ps4/hkPs4Socket.h>
static hkSocket* HK_CALL hkSocketCreate()
{
	return new hkPs4Socket();
}

void (HK_CALL *hkSocket::s_platformNetInit)() = hkPs4NetworkInit;
void (HK_CALL *hkSocket::s_platformNetQuit)() = hkPs4NetworkQuit;

#else

// General BSD socket support (LINUX, Windows, Mac, Xbox etc)
#	include <Common/Base/System/Io/Platform/Bsd/hkBsdSocket.h>
	static hkSocket* HK_CALL hkSocketCreate()
	{
		return new hkBsdSocket();
	}

	void (HK_CALL *hkSocket::s_platformNetInit)() = hkBsdNetworkInit;
	void (HK_CALL *hkSocket::s_platformNetQuit)() = hkBsdNetworkQuit;

#endif

hkBool hkSocket::s_platformNetInitialized = false;

hkSocket* (HK_CALL *hkSocket::create)() = hkSocketCreate;

hkSocket::hkSocket()
{
	m_reader.m_socket = this;
	m_writer.m_socket = this;

	if (s_platformNetInitialized == false && s_platformNetInit)
	{
		s_platformNetInit();
		s_platformNetInitialized = true;
	}
}

int hkSocket::ReaderAdapter::read( void* buf, int nbytes )
{
	char* cbuf = static_cast<char*>(buf);
	int size = 0;
	while( size < nbytes )
	{
		int r = m_socket->read(cbuf+size, nbytes-size);
		size += r;
		if( r == 0 )
		{
			return size;
		}
	}
	return nbytes;
}

hkBool hkSocket::ReaderAdapter::isOk() const
{
	return m_socket->isOk();
}

int hkSocket::WriterAdapter::write( const void* buf, int nbytes )
{
	const char* cbuf = static_cast<const char*>(buf);
	int size = 0;
	while( size < nbytes )
	{
		int w = m_socket->write(cbuf+size, nbytes-size);
		size += w;
		if( w == 0 )
		{
			return size;
		}
	}
	return nbytes;
	
}

hkBool hkSocket::WriterAdapter::isOk() const
{
	return m_socket->isOk();
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
