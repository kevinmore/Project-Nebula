/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HK_BASE_WIIU_SOCKET_H
#define HK_BASE_WIIU_SOCKET_H

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Socket/hkSocket.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>
#include <Common/Base/Container/Queue/hkQueue.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

// The HIO connection is not used by default. Define HK_USE_DEPRECATED_HIO_CONNECTION and rebuild if needed
#if defined(HK_USE_DEPRECATED_HIO_CONNECTION)
#include <cafe.h>
#include <cafe/hio.h>

class hkWiiuSocket : public hkSocket
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS );

		typedef int socket_t;

		hkWiiuSocket(int s=-1, hkWiiuSocket* parent = HK_NULL);

		virtual ~hkWiiuSocket();

		virtual hkBool isOk() const;

		virtual void close();

		virtual int read( void* buf, int nbytes);

		virtual int write( const void* buf, int nbytes);

		// client
		virtual hkResult connect(const char* servername, int portNumber);

		virtual bool canRead();
	
		// server
		virtual hkResult listen(int port);
		virtual hkSocket* pollForNewClient();

		// interna;
		void setStatus(HIOStatus s);

		void addChild(hkWiiuSocket* c);
		void removeChild(hkWiiuSocket* c);

	protected:
		
		HIOStatus m_status;
		OSEvent m_hostConnectionEvent;
		HIOHandle m_ioHandle;
		bool m_listener;
		int m_port;

		hkWiiuSocket* m_parent;
		hkArray<hkWiiuSocket*> m_children;

	public:

		hkUint32 m_pending;
		char m_pendingReadBuf[4];

		
};

/// Set up WiiU network. Set hkSocket::s_platformNetInit to HK_NULL to prevent this being called
void HK_CALL hkWiiuNetworkInit();

/// Set up WiiU network. Set hkSocket::s_platformNetQuit to HK_NULL to prevent this being called
void HK_CALL hkWiiuNetworkQuit();

#endif

#endif // HK_BASE_WIIU_SOCKET_H

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
