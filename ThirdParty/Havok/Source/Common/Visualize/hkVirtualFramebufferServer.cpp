/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>

#include <Common/Visualize/hkVirtualFramebufferServer.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/Socket/hkSocket.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>

hkVirtualFramebufferServer* hkVirtualFramebufferServer::g_instance = HK_NULL;

hkVirtualFramebufferServer::hkVirtualFramebufferServer()
: m_server(HK_NULL)
{
	g_instance = this;
}

hkVirtualFramebufferServer::~hkVirtualFramebufferServer()
{
	for (int c = m_clients.getSize()-1; c >= 0; --c)
	{
		deleteClient( c );
	}

	g_instance = HK_NULL;
	
	if (m_server)
	{
		m_server->removeReference();
	}
}

void hkVirtualFramebufferServer::deleteClient(int i)
{
	hkVirtualFramebufferServerClient* client = m_clients[i];
	if(client->m_outStream)
	{
		client->m_outStream->removeReference();
	}
	if(client->m_inStream)
	{
		client->m_inStream->removeReference();
	}
	if(client->m_socket)
	{
		client->m_socket->removeReference();
	}
	delete client;
	m_clients.removeAt(i);
}

void hkVirtualFramebufferServer::serve( int listenPort  )
{
	if(!m_server)
	{
		m_server = hkSocket::create();
		if(m_server)
		{
			m_server->listen(listenPort);
			HK_REPORT("hkVirtualFramebufferServer: created and will poll for new client(s) on port " << listenPort << " every frame");
		}
		else
		{
			HK_REPORT("hkVirtualFramebufferServer: could not be created, please check that you platform supports sockets with the hkBase library");
		}
	}
	else
	{
		HK_REPORT("hkVirtualFramebufferServer: has already been created, only one server allowed per visual debugger instance");
	}

}

int hkVirtualFramebufferServer::getNumConnectedClients() const
{
	return m_clients.getSize();
}

static void _sendConnectInfoToClient( hkVirtualFramebufferServerClient* c )
{
	hkStringBuf platformInfo;
#if defined(HK_PLATFORM_X64)
	platformInfo.printf("Win64");
#elif defined(HK_PLATFORM_WIN32)
	platformInfo.printf("Win32");
#elif defined(HK_PLATFORM_IOS)
	platformInfo.printf("iOS");
#elif defined(HK_PLATFORM_MAC386)
	platformInfo.printf("MacOSX");
#elif defined(HK_PLATFORM_RVL)
	platformInfo.printf("Wii");
#elif defined(HK_PLATFORM_PS3)
	platformInfo.printf("PS3");
#elif defined(HK_PLATFORM_XBOX360)
	platformInfo.printf("Xbox360");
#elif defined(HK_PLATFORM_LINUX)
	#ifdef HK_ARCH_X64
		platformInfo.printf("Linux32");
	#else	
		platformInfo.printf("Linux64");
	#endif
#else
	platformInfo.printf("UnknownPlatform");
#endif

	platformInfo += ", " HAVOK_SDK_VERSION_STRING;
	platformInfo += (HK_CONFIG_SIMD==HK_CONFIG_SIMD_ENABLED? ",SIMD" :"");
	platformInfo += (HK_CONFIG_THREAD==HK_CONFIG_MULTI_THREADED? ",MT" :"");
	int l = platformInfo.getLength();
	

	int packetSize = (3*4)+l;
	c->m_outStream->write32u( packetSize );
	c->m_outStream->write32u( hkVirtualFramebufferProtocol::SEND_INFO );
	c->m_outStream->write32u( hkVirtualFramebufferProtocol::CURRENT_VERSION ); // Ver 
	c->m_outStream->write32u( l ); // str len
	c->m_outStream->writeRaw( platformInfo.cString(), l ); // str
	
}

static inline void _downsample4(const hkUint8* HK_RESTRICT rawData, bool notLastX, bool notLastY, const hkVirtualFramebuffer* HK_RESTRICT buffer, hkUint8* HK_RESTRICT dest, bool yFlip )
{
	hkUint32 nextX  = notLastX? buffer->m_pixelStrideInBytes : 0;
	hkInt32  nextY ;
	if (yFlip) 
	{
		nextY = notLastY? -(hkInt32)buffer->m_rowPitchInBytes : 0;		
	}
	else
	{
		nextY = notLastY? buffer->m_rowPitchInBytes : 0;
	}
	hkInt32 nextXY = nextX + nextY;
	
	const hkUint8* cv = rawData;
	const hkUint8* xv = rawData + nextX;
	const hkUint8* yv = rawData + nextY;
	const hkUint8* xyv =rawData + nextXY;

	dest[0] = hkUint8( (((hkUint32)cv[0]) + ((hkUint32)xv[0]) + ((hkUint32)yv[0]) + ((hkUint32)xyv[0])) / 4 ); 
	dest[1] = hkUint8( (((hkUint32)cv[1]) + ((hkUint32)xv[1]) + ((hkUint32)yv[1]) + ((hkUint32)xyv[1])) / 4 ); 
	dest[2] = hkUint8( (((hkUint32)cv[2]) + ((hkUint32)xv[2]) + ((hkUint32)yv[2]) + ((hkUint32)xyv[2])) / 4 ); 
}

static inline void _888to565( const hkUint8* HK_RESTRICT rgb888, hkUint8* HK_RESTRICT rgb565)
{
	rgb565[0] = rgb888[0] & 0xF8; // top 5 bits
	rgb565[0] |= (rgb888[1] >> 5); // top 3 bits
	rgb565[1] = (rgb888[1] & 0x1C) << 3; // next 3 bits
	rgb565[1] |= (rgb888[2] >> 3); // top 5 bits	
}

static inline void _888to565_s( const hkUint8* HK_RESTRICT rgb888, hkUint8* HK_RESTRICT rgb565)
{
	rgb565[0] = rgb888[2] & 0xF8; // top 5 bits
	rgb565[0] |= (rgb888[1] >> 5); // top 3 bits
	rgb565[1] = (rgb888[1] & 0x1C) << 3; // next 3 bits
	rgb565[1] |= (rgb888[0] >> 3); // top 5 bits	
}

class hkVirtualFramebufferServerRawStream
{
public:

	hkVirtualFramebufferServerRawStream( hkDisplaySerializeOStream* s )
		: m_outStream(s)
	{
	}

	inline void end() { }

	inline void addPixelRGB888(const hkUint8* rgb /*3*/ )
	{
		m_outStream->writeRaw(rgb, 3);
	}

	inline void addPixelBGR888(const hkUint8* bgr /*3*/ )
	{
		m_outStream->write8u(bgr[2]);
		m_outStream->write8u(bgr[1]);
		m_outStream->write8u(bgr[0]);
	}

	inline void addPixelRGB565(const hkUint8* rgb /*2*/ )
	{
		m_outStream->writeRaw(rgb, 2);		
	}

	hkDisplaySerializeOStream* m_outStream;
};

class hkVirtualFramebufferServerRLEStream
{
public:

	hkVirtualFramebufferServerRLEStream( int byteEtimate, int numComp )
	{
		m_curFrameData.reserve(byteEtimate);
		m_curPixel[0] = m_curPixel[1] = m_curPixel[2] = 0xff;
		m_numCurPixel = 0;
		m_curPixelIndex = 0;
		m_numPixelComp = numComp;
	}

	inline void end()
	{
		if (m_numCurPixel > 0)
		{
			//send
			m_curFrameData.pushBack(m_numCurPixel);
			m_curFrameData.pushBack(m_curPixel[0]);
			m_curFrameData.pushBack(m_curPixel[1]);
			if (m_numPixelComp == 3)
			{
				m_curFrameData.pushBack(m_curPixel[2]);
			}
			m_numCurPixel = 0;
		}
	}

	inline void addPixelRGB888(const hkUint8* rgb /*3*/ )
	{
		if ( (m_numCurPixel == 0) || 
			(rgb[0] != m_curPixel[0]) ||
			(rgb[1] != m_curPixel[1]) ||
			(rgb[2] != m_curPixel[2]) )
		{
			if (m_numCurPixel > 0)
			{
				//send
				m_curFrameData.pushBack(m_numCurPixel);
				m_curFrameData.pushBack(m_curPixel[0]);
				m_curFrameData.pushBack(m_curPixel[1]);
				m_curFrameData.pushBack(m_curPixel[2]);
			}
			m_numCurPixel = 1;
			m_curPixel[0] = rgb[0];
			m_curPixel[1] = rgb[1];
			m_curPixel[2] = rgb[2];
		}
		else // have at least one of this pixel
		{
			++m_numCurPixel;
			if (m_numCurPixel == 0xff)
			{
				// send 
				m_curFrameData.pushBack(m_numCurPixel);
				m_curFrameData.pushBack(m_curPixel[0]);
				m_curFrameData.pushBack(m_curPixel[1]);
				m_curFrameData.pushBack(m_curPixel[2]);
				m_numCurPixel = 0;
			}
		}
	}

	inline void addPixelBGR888(const hkUint8* bgr /*3*/ )
	{
		if ( (m_numCurPixel == 0) || 
			(bgr[2] != m_curPixel[0]) ||
			(bgr[1] != m_curPixel[1]) ||
			(bgr[0] != m_curPixel[2]) )
		{
			if (m_numCurPixel > 0)
			{
				//send last
				m_curFrameData.pushBack(m_numCurPixel);
				m_curFrameData.pushBack(m_curPixel[0]);
				m_curFrameData.pushBack(m_curPixel[1]);
				m_curFrameData.pushBack(m_curPixel[2]);
			}
			m_numCurPixel = 1;
			m_curPixel[0] = bgr[2];
			m_curPixel[1] = bgr[1];
			m_curPixel[2] = bgr[0];
		}
		else // have at least one of this pixel
		{
			++m_numCurPixel;
			if (m_numCurPixel == 0xff)
			{
				// send 
				m_curFrameData.pushBack(m_numCurPixel);
				m_curFrameData.pushBack(m_curPixel[0]);
				m_curFrameData.pushBack(m_curPixel[1]);
				m_curFrameData.pushBack(m_curPixel[2]);
				m_numCurPixel = 0;
			}
		}
	}

	inline void addPixelRGB565(const hkUint8* rgb /*2*/ )
	{
		if ( (m_numCurPixel == 0) || 
			(rgb[0] != m_curPixel[0]) ||
			(rgb[1] != m_curPixel[1]) )
		{
			if (m_numCurPixel > 0)
			{
				//send
				m_curFrameData.pushBack(m_numCurPixel);
				m_curFrameData.pushBack(m_curPixel[0]);
				m_curFrameData.pushBack(m_curPixel[1]);
			}
			m_numCurPixel = 1;
			m_curPixel[0] = rgb[0];
			m_curPixel[1] = rgb[1];
		}
		else // have at least one of this pixel
		{
			++m_numCurPixel;
			if (m_numCurPixel == 0xff)
			{
				// send 
				m_curFrameData.pushBack(m_numCurPixel);
				m_curFrameData.pushBack(m_curPixel[0]);
				m_curFrameData.pushBack(m_curPixel[1]);
				m_numCurPixel = 0;
			}
		}
	}

	hkArray< hkUint8 > m_curFrameData;

	hkUint32 m_curPixelIndex;
	hkUint8 m_numCurPixel; // up to 256 each time
	hkUint8 m_curPixel[3];
	int m_numPixelComp;
};

struct hkVirtualFramebufferWriterSendOptions
{
	hkVirtualFramebufferProtocol::FramebufferFormat sendFormat;
	hkUint32 sendW;
	hkUint32 sendH;
	hkUint32 scale;
	hkInt32 startX;
	hkInt32 startY;
	hkInt32 endX;
	hkInt32 endY;
};

template <typename T>
class hkVirtualFramebufferWriter
{
public:

	hkVirtualFramebufferWriter(T* w) : m_writer(w) { }

	hkUint32 writeBuffer( const hkVirtualFramebuffer* buffer, const hkVirtualFramebufferWriterSendOptions& o )
	{
		hkUint32 pixelsSent = 0; // so we can account for rounding error for odd res / scales

		bool bufferHasSubRect = (buffer->m_startX > 0) || (buffer->m_endX > 0) ||
		   						(buffer->m_startY > 0) || (buffer->m_endY > 0);

		// check buffset sub rect bigger or the same as what we want to send from it
		HK_ASSERT(0x2e9ad680, !bufferHasSubRect || 
			( ( buffer->m_startX <= (hkUint32)o.startX) && 
			  ( buffer->m_startY <= (hkUint32)o.startY) &&
			  ( (buffer->m_endX < 0) || (buffer->m_endX >= (hkInt32)o.endX)) &&  
			  ( (buffer->m_endY < 0) || (buffer->m_endY >= (hkInt32)o.endY)) ) );

		const hkUint32 pixelStride = buffer->m_pixelStrideInBytes * o.scale;

		hkUint32 localStartX = o.startX;
		hkUint32 localStartY = o.startY;
		hkInt32 localEndX = o.endX;
		hkInt32 localEndY = o.endY;
		//hkUint32 localBufferWidth = buffer->m_fullBufferWidthInPixels;
		hkUint32 localBufferHeight= buffer->m_fullBufferHeightInPixels;
		if (bufferHasSubRect)
		{
			localStartX -= buffer->m_startX;
			localStartY -= buffer->m_startY;
			hkInt32 bufferEndX = buffer->m_endX >= 0? buffer->m_endX : (hkInt32)buffer->m_fullBufferWidthInPixels - 1;
			hkInt32 bufferEndY = buffer->m_endY >= 0? buffer->m_endY : (hkInt32)buffer->m_fullBufferHeightInPixels - 1;
			if (bufferEndX >= (hkInt32)buffer->m_fullBufferWidthInPixels) { bufferEndX = (hkInt32)buffer->m_fullBufferWidthInPixels - 1; }
			if (bufferEndY >= (hkInt32)buffer->m_fullBufferHeightInPixels) { bufferEndY = (hkInt32) buffer->m_fullBufferHeightInPixels - 1; }
			localEndX -= bufferEndX;
			localEndY -= bufferEndY;
		//	localBufferWidth = bufferEndX - buffer->m_startX + 1;
			localBufferHeight = bufferEndY - buffer->m_startY + 1;
		}

		//XX optimize this (take out all the if etc)
		const bool yFlip =  buffer->m_rowOrder == hkVirtualFramebuffer::DATA_TOP_LEFT ;

		const hkUint8* bufferStartRow =  buffer->m_data + (( yFlip ?  localBufferHeight - localStartY - 1 : localStartY ) * buffer->m_rowPitchInBytes);
		const hkInt32 rowsToSend = (hkInt32)o.sendH;
		for (hkInt32 y = 0; y <rowsToSend ; ++y)
		{
			const hkUint8* curDataPtr = bufferStartRow + ( ( yFlip ? -y : y ) * (hkInt32)( o.scale * buffer->m_rowPitchInBytes) ) + (localStartX * buffer->m_pixelStrideInBytes);
			const bool notLastY = y < (rowsToSend - 1);
			hkMath::prefetch128(curDataPtr + buffer->m_rowPitchInBytes);
			// RGB
			if (o.sendFormat == hkVirtualFramebufferProtocol::PIXEL_RGB888)
			{
				if ( buffer->m_format == hkVirtualFramebuffer::DATA_RGB)
				{
					for (hkUint32 x = 0; x < o.sendW; ++x)
					{
						const hkUint8* curPtr = curDataPtr + (x * pixelStride);
						if ( o.scale == 1 )
						{
							m_writer->addPixelRGB888(curPtr);
						}
						else // really only good for 4
						{
							hkUint8 downsample[3];
							_downsample4(curPtr, x<(o.sendW-1), notLastY, buffer, downsample, yFlip ); 
							m_writer->addPixelRGB888(downsample);
						}
					}
					pixelsSent += o.sendW;
				}
				else // Need BGR->RGB swap
				{
					for (hkUint32 x = 0; x < o.sendW; ++x)
					{
						const hkUint8* curPtr = curDataPtr + (x * pixelStride);
						if ( o.scale == 1 )
						{
							m_writer->addPixelBGR888(curPtr);
						}
						else // really only good for 4
						{
							hkUint8 downsample[3];
							_downsample4(curPtr, x<(o.sendW-1), notLastY, buffer, downsample, yFlip ); 
							m_writer->addPixelBGR888(downsample);
						}
					}
					pixelsSent += o.sendW;
				}
			}
			else // compressed format // XX add native support on iOS for direct RGB565 send etc
			{
				if ( buffer->m_format == hkVirtualFramebuffer::DATA_RGB)
				{
					for (hkUint32 x = 0; x < o.sendW; ++x)
					{
						const hkUint8* curPtr = curDataPtr + (x * pixelStride);
						hkUint8 data565[2];
						if ( o.scale == 1 )
						{
							_888to565( curPtr, data565 );
						}
						else 
						{
							hkUint8 downsample[3];
							_downsample4(curPtr, x<(o.sendW-1), notLastY, buffer, downsample, yFlip ); 
							_888to565( downsample, data565 );
						}
						m_writer->addPixelRGB565(data565);
					}
					pixelsSent += o.sendW;
				}
				else // Need BGR->RGB swap
				{
					for (hkUint32 x = 0; x < o.sendW; ++x)
					{
						const hkUint8* curPtr = curDataPtr + (x * pixelStride);
						hkUint8 data565[2];
						if ( o.scale == 1 )
						{
							_888to565_s( curPtr, data565 );
						}
						else 
						{
							hkUint8 downsample[3];
							_downsample4(curPtr, x<(o.sendW-1), notLastY, buffer, downsample, yFlip ); 
							_888to565_s( downsample, data565 );
						}
						m_writer->addPixelRGB565( data565 );
					}
					pixelsSent += o.sendW;
				}
			}
		}

		m_writer->end();

		return pixelsSent;
	}

	T* m_writer;
};

// not incl the packet size uint32
#define HEADER_PACKET_BYTES ((4 * 2) /* cmds */ + (2 * 10) /* 10x 16s */) 
static void _writeHeader(hkDisplaySerializeOStream* outStream, 
						 hkVirtualFramebufferProtocol::FramebufferCommands sendType,
						 hkUint32 packetSize,
						  const hkVirtualFramebuffer* buffer,
						 hkVirtualFramebufferWriterSendOptions& sendOptions)
{
	outStream->write32u( packetSize );
	outStream->write32u( hkVirtualFramebufferProtocol::SEND_FRAMEBUFFER );
	outStream->write32u( sendType ); 	
	
	outStream->write16u( (hkUint16)( buffer->m_fullBufferWidthInPixels)); // w full screen size
	outStream->write16u( (hkUint16)( buffer->m_fullBufferHeightInPixels)); // h full screen size

	outStream->write16u( (hkUint16)( sendOptions.scale )); // the scale wrt to full screen
	outStream->write16u( (hkUint16)( sendOptions.sendFormat));
	outStream->write16u( (hkUint16)( sendOptions.startX)); // start pixel X, in full buffer coords
	outStream->write16u( (hkUint16)( sendOptions.startY)); // start pixel Y, in full buffer coords
	outStream->write16u( (hkUint16)  sendOptions.sendW ); // num pixels sent, width (scaled coords)
	outStream->write16u( (hkUint16)  sendOptions.sendH ); // num pixels sent, height (scaled coords) 

	outStream->write16u( (hkUint16)  buffer->m_displayRotation );
	outStream->write16u( (hkUint16)  buffer->m_displayBufferId );

}

void hkVirtualFramebufferServer::sendFramebufferToClient( hkVirtualFramebufferServerClient& c, const hkVirtualFramebuffer* buffer )
{

	hkVirtualFramebufferWriterSendOptions sendOptions;

	sendOptions.scale = c.m_options.m_scale;
	sendOptions.sendFormat =  c.m_options.m_maxSendFormat;

	sendOptions.startX = (hkInt32)(c.m_options.m_areaOfInterest.m_startX * buffer->m_fullBufferWidthInPixels); 
	sendOptions.startY = (hkInt32)(c.m_options.m_areaOfInterest.m_startY * buffer->m_fullBufferHeightInPixels);
	sendOptions.endX   = (hkInt32)(c.m_options.m_areaOfInterest.m_endX * buffer->m_fullBufferWidthInPixels); 
	sendOptions.endY   = (hkInt32)(c.m_options.m_areaOfInterest.m_endY * buffer->m_fullBufferHeightInPixels);

	// -1 == all 
	if (sendOptions.endX < 0) sendOptions.endX = buffer->m_fullBufferWidthInPixels - 1;
	if (sendOptions.endY < 0) sendOptions.endY = buffer->m_fullBufferHeightInPixels - 1;

	const hkInt32 dz = 5; // closer than this to edges == edges
	
	// clamp to egdes / assumed values
	if (sendOptions.startX < dz) sendOptions.startX = 0;
	if ((sendOptions.endX < 0) || (sendOptions.endX > (hkInt32)( buffer->m_fullBufferWidthInPixels - dz ))) sendOptions.endX = (hkInt32)(buffer->m_fullBufferWidthInPixels - 1);
	if (sendOptions.endX < sendOptions.startX) sendOptions.endX = sendOptions.startX + 1;
	
	if (sendOptions.startY < dz) sendOptions.startY = 0;
	if ((sendOptions.endY < 0) || (sendOptions.endY > (hkInt32)( buffer->m_fullBufferHeightInPixels - dz ))) sendOptions.endY = (hkInt32)(buffer->m_fullBufferHeightInPixels - 1);
	if (sendOptions.endY < sendOptions.startY) sendOptions.endY = sendOptions.startY + 1;
	
	bool bufferHasSubRect = (buffer->m_startX > 0) || (buffer->m_endX > 0) ||
     						(buffer->m_startY > 0) || (buffer->m_endY > 0);

	// check buffset sub rect bigger or the same as what we want to send from it
	// can happen on edge cases as we store area of intrest as floats
	if (bufferHasSubRect)
	{
		if (buffer->m_startX > (hkUint32)sendOptions.startX)
			sendOptions.startX = buffer->m_startX;
		if (buffer->m_startY > (hkUint32)sendOptions.startY)
			sendOptions.startY = buffer->m_startY;
		if ((buffer->m_endX >= 0) && (buffer->m_endX < (hkInt32)sendOptions.endX))
			sendOptions.endX = buffer->m_endX;
		if ((buffer->m_endY >= 0) && (buffer->m_endY < (hkInt32)sendOptions.endY))
			sendOptions.endY = buffer->m_endY;
	}
	

	sendOptions.sendW = ((sendOptions.endX - sendOptions.startX ) / sendOptions.scale) + 1;
	sendOptions.sendH = ((sendOptions.endY - sendOptions.startY ) / sendOptions.scale) + 1;

	const hkUint32 bpp = sendOptions.sendFormat == hkVirtualFramebufferProtocol::PIXEL_RGB888? 3 : 2;
	const hkUint32 rawDataSize = sendOptions.sendW * sendOptions.sendH * bpp;

		// same options, same size == diff ok
	///XXX not good for multiple clients
	bool canSendDiff =  false; //XX not impl yet   (m_lastDiffSendOptions == c.m_options) && (m_framebufferDiffStore.getSize() == dataSize);
	//bool storeDiff = false;
		
	bool canSendRLE = c.m_options.m_allowRleSend;
	bool sendRaw = !canSendDiff && !canSendRLE;

	if (canSendDiff && (c.m_framebufferDiffStore.getSize() != (int)rawDataSize))
	{
		c.m_framebufferDiffStore.clearAndDeallocate();
		c.m_framebufferDiffStore.setSize(rawDataSize);
	}
	
	if (sendRaw)
	{
		const hkUint32 rawPacketSize = HEADER_PACKET_BYTES + rawDataSize; 
		_writeHeader( c.m_outStream, hkVirtualFramebufferProtocol::FRAMEBUFFER_RECT_RAW, rawPacketSize, buffer, sendOptions );
		
		hkVirtualFramebufferServerRawStream rawStream(c.m_outStream);
		hkVirtualFramebufferWriter< hkVirtualFramebufferServerRawStream > bw( &rawStream );
		HK_ON_DEBUG(int pixelsSent =)  bw.writeBuffer(buffer, sendOptions);
		HK_ASSERT(0x18783a8c, (pixelsSent*bpp) == rawDataSize);
	}
	else if (canSendRLE) // Diff or RLE (both variable size packets). Diff not impl yet
	{
		hkVirtualFramebufferServerRLEStream rleStream(rawDataSize / 3, bpp);
		hkVirtualFramebufferWriter< hkVirtualFramebufferServerRLEStream > bw( &rleStream );
		HK_ON_DEBUG(int pixelsSent =) bw.writeBuffer(buffer, sendOptions);
		HK_ASSERT(0x41d3545, (pixelsSent*bpp) == rawDataSize);
		int rleDataSize = rleStream.m_curFrameData.getSize();
		const hkUint32 rlePacketSize = HEADER_PACKET_BYTES + rleDataSize; 
		_writeHeader( c.m_outStream, hkVirtualFramebufferProtocol::FRAMEBUFFER_RECT_RLE, rlePacketSize, buffer, sendOptions );
		c.m_outStream->writeRaw( rleStream.m_curFrameData.begin(), rleDataSize );
	}
	
	c.m_outStream->getStreamWriter()->flush();
}

void hkVirtualFramebufferServer::recvInput( hkVirtualFramebufferServerClient& c )
{
	while (c.m_socket->canRead())
	{
		hkUint32 packetSize = c.m_inStream->read32u();
		hkUint32 cmd = 0;
		if (c.m_inStream->isOk())
		{
			cmd = c.m_inStream->read32u();
		}
		if (c.m_inStream->isOk() && cmd)
		{
			switch (cmd)
			{
				case hkVirtualFramebufferProtocol::SEND_KEY:
					{	
						hkVirtualKeyEvent evnt; 
						evnt.m_key   = (hkUint8) c.m_inStream->read8u();
						evnt.m_state = c.m_inStream->read8u() != 0;
						evnt.m_focusBufferId = c.m_inStream->read16u();
					
						if (c.m_inStream->isOk())
						{
							for (int ki=0; ki < m_keyboardHandlers.getSize(); ++ki)
							{
								m_keyboardHandlers[ki]->onVirtualKeyEventUpdate(evnt);
							}
						}
						break;
					}

				case hkVirtualFramebufferProtocol::SEND_MOUSE:
					{	
						hkVirtualMouse m; 
						m.m_buttons = (hkVirtualMouse::Button)c.m_inStream->read32u();
						m.m_screenX = c.m_inStream->readFloat32();
						m.m_screenY = c.m_inStream->readFloat32();
						m.m_wheelDelta = c.m_inStream->readFloat32();
						m.m_focusBufferId = c.m_inStream->read16u();
						if (c.m_inStream->isOk())
						{
							for (int ki=0; ki < m_mouseHandlers.getSize(); ++ki)
							{
								m_mouseHandlers[ki]->onVirtualMouseUpdate(m);
							}
						}
						break;
					}

				case hkVirtualFramebufferProtocol::SEND_GAMEPAD:
					{	
						hkVirtualGamepad g; 
						g.m_buttons = (hkVirtualGamepad::Button)c.m_inStream->read32u();
						g.m_sticks[0].x = c.m_inStream->readFloat32();
						g.m_sticks[0].y = c.m_inStream->readFloat32();
						g.m_sticks[1].x = c.m_inStream->readFloat32();
						g.m_sticks[1].y = c.m_inStream->readFloat32();
						g.m_triggers[0].z = c.m_inStream->readFloat32();
						g.m_triggers[1].z = c.m_inStream->readFloat32();
						g.m_gamePadNum = c.m_inStream->read32u();
						
						if (c.m_inStream->isOk())
						{
							for (int ki=0; ki < m_gamepadHandlers.getSize(); ++ki)
							{
								m_gamepadHandlers[ki]->onVirtualGamepadUpdate(g);
							}
						}
						break;
					}

				case hkVirtualFramebufferProtocol::SEND_OPTIONS:
					{
						hkVirtualFramebufferServerClient::RuntimeOptions inOptions;
						inOptions.m_scale = c.m_inStream->read32u();
						inOptions.m_areaOfInterest.m_framebufferId = c.m_inStream->read32();
						inOptions.m_areaOfInterest.m_startX = c.m_inStream->readFloat32();
						inOptions.m_areaOfInterest.m_startY = c.m_inStream->readFloat32();
						inOptions.m_areaOfInterest.m_endX = c.m_inStream->readFloat32();
						inOptions.m_areaOfInterest.m_endY = c.m_inStream->readFloat32();
						inOptions.m_areaOfInterest.normalize();
						inOptions.m_allowDiffSend = c.m_inStream->read8u() != 0;
						inOptions.m_allowRleSend = c.m_inStream->read8u() != 0;
						inOptions.m_maxSendFormat = (hkVirtualFramebufferProtocol::FramebufferFormat) c.m_inStream->read8u();
						
						if (c.m_inStream->isOk())
						{
							c.m_options = inOptions;
						}
						break;
					}

				case hkVirtualFramebufferProtocol::SEND_DROPFILES:
					{
						hkVirtualFileDrop d; 
						d.m_screenX = c.m_inStream->readFloat32();
						d.m_screenY = c.m_inStream->readFloat32();
						int numFiles = c.m_inStream->read32u();
						d.m_files.setSize(numFiles);
						for (int fi=0; (fi < numFiles) && c.m_inStream->isOk(); ++fi)
						{
							int strLenInclNull = c.m_inStream->read32u();
							int packetStrSize = HK_NEXT_MULTIPLE_OF(4, strLenInclNull);
							char* strAlloc = hkAllocate<char>(packetStrSize, HK_MEMORY_CLASS_STRING);
							c.m_inStream->readRaw( (void*)strAlloc, packetStrSize);
							d.m_files[fi].setPointerAligned( strAlloc );
						}

						if (c.m_inStream->isOk())
						{
							for (int ki=0; ki < m_filedropHandlers.getSize(); ++ki)
							{
								m_filedropHandlers[ki]->onVirtualFileDrop(d);
							}
						}
						break;
					}

			default:
				// skip data, don't know what it is
				{
					HK_WARN_ALWAYS(0xabbaf5b5,"UNKOWN COMMAND FROM CLIENT %d" << cmd << ", packet size " << packetSize << ", and connection is still valid?\n");
					
					int dataLeft = packetSize - 4;
					// No incoming packet is big in this protocol, so could be totally corrupt
					if ((dataLeft > 0) && (dataLeft < 1024))
					{
						hkLocalArray<hkUint8> d( dataLeft );
						c.m_inStream->readRaw(d.begin(), dataLeft );
					}
					else
					{
						int cindex = m_clients.indexOf(&c);
						deleteClient(cindex);
						return;
					}
				}
				break;
			}
		}
		else
		{
			int cindex = m_clients.indexOf(&c);
			HK_WARN_ALWAYS(0xabba41c7,"Remote view client disconnected : #" << cindex );
			deleteClient(cindex);
			return;
		}
	}
}

void hkVirtualFramebufferServer::step()
{
	// see if there is a new client trying to connect
	if(m_server)
	{
		hkSocket* socket = m_server->pollForNewClient();
		if(socket)
		{
			HK_REPORT("A new hkVirtualFramebufferServer network client has been received ");
			hkVirtualFramebufferServerClient* newClient = new hkVirtualFramebufferServerClient();
			
			newClient->m_socket = socket;
			newClient->m_outStream = new hkDisplaySerializeOStream( &socket->getWriter() ); 
			newClient->m_inStream = new hkDisplaySerializeIStream( &socket->getReader() );

			m_clients.pushBack(newClient);
			
			// Send connect info to new client, incls this sends version num
			_sendConnectInfoToClient( newClient );
		}
	}

	// See if all ok
	for (int dc=m_clients.getSize()-1; dc >= 0; --dc)
	{
		if (!m_clients[dc]->m_socket->isOk())
		{
			deleteClient(dc);
		}
	}

	// Backwards as could delete during recv
	for (int c=m_clients.getSize()-1; c>=0; --c)
	{
		// read any new input (last to connect gets prio as last to be checked here
		recvInput( *m_clients[c] );
	}

	// See if still all still ok
	for (int dc=m_clients.getSize()-1; dc >= 0; --dc)
	{
		if (!m_clients[dc]->m_socket->isOk())
		{
			deleteClient(dc);
		}
	}

}

void hkVirtualFramebufferServer::sendFrameBuffer( const hkVirtualFramebuffer* buffer )
{
	if (m_server && buffer)
	{
		for (int c=m_clients.getSize()-1; c >= 0; --c)
		{
			if (!m_clients[c]->m_socket->isOk())
			{
				deleteClient(c);
			}
			else
			{
				// write the latest buffer
				sendFramebufferToClient( *m_clients[c], buffer );
			}
		}
	}
}	

void hkVirtualFramebufferServer::sendString( const char* str )
{
	if (m_server && str)
	{
		for (int c=m_clients.getSize()-1; c >= 0; --c)
		{
			if (!m_clients[c]->m_socket->isOk())
			{
				deleteClient(c);
			}
			else
			{
				int strLen = hkString::strLen(str) + 1; // incl null to make life easy other end
				if (strLen > 1)
				{
					int alignedStrLen = HK_NEXT_MULTIPLE_OF(4, strLen);
					hkDisplaySerializeOStream* outStream = m_clients[c]->m_outStream;
					int packetSize = alignedStrLen + 4 + 4;
					outStream->write32u( packetSize );
					outStream->write32u( hkVirtualFramebufferProtocol::SEND_STRING );
					outStream->write32u( strLen ); 	
					outStream->writeRaw( str, strLen );
					if (strLen != alignedStrLen)
					{
						hkUint32 pad = 0;
						outStream->writeRaw( &pad, alignedStrLen - strLen );
					}
				
					outStream->getStreamWriter()->flush();
				}
			}
		}
	}
}

bool hkVirtualFramebufferServer::getFramebufferRectOfInterest( hkVirtualFramebufferRelativeRect& rect )
{
	rect.m_startX = 1000000.f;
	rect.m_startY = 1000000.f;
	rect.m_endX = -1000000.f;
	rect.m_endY = -1000000.f;
	rect.m_framebufferId = -1;

	bool clientWantsAll = false;
	if (m_server)
	{
		for (int c=m_clients.getSize()-1; c >= 0; --c)
		{
			rect += m_clients[c]->m_options.m_areaOfInterest;
			if (!clientWantsAll && (m_clients[c]->m_options.m_areaOfInterest.m_framebufferId >= 0))
			{
				rect.m_framebufferId = m_clients[c]->m_options.m_areaOfInterest.m_framebufferId;
			}
			else 
			{
				clientWantsAll = true;
				rect.m_framebufferId = -1;
			}
		}

		return (m_clients.getSize() > 0);
	}
	return false;
}


void hkVirtualFramebufferServer::registerGamepadCallback( hkVirtualGamepadHandler* h )
{
	if (m_gamepadHandlers.indexOf(h) < 0)
		m_gamepadHandlers.pushBack(h);
}

void hkVirtualFramebufferServer::unregisterGamepadCallback( hkVirtualGamepadHandler* h )
{
	int idx = m_gamepadHandlers.indexOf(h);
	if (idx >= 0)
		m_gamepadHandlers.removeAt(idx);
}

void hkVirtualFramebufferServer::registerKeyboardCallback( hkVirtualKeyEventHandler* h )
{
	if (m_keyboardHandlers.indexOf(h) < 0)
		m_keyboardHandlers.pushBack(h);
}

void hkVirtualFramebufferServer::unregisterKeyboardCallback( hkVirtualKeyEventHandler* h )
{
	int idx = m_keyboardHandlers.indexOf(h);
	if (idx >= 0)
		m_keyboardHandlers.removeAt(idx);
}

void hkVirtualFramebufferServer::registerMouseCallback( hkVirtualMouseHandler* h )
{
	if (m_mouseHandlers.indexOf(h) < 0)
		m_mouseHandlers.pushBack(h);
}

void hkVirtualFramebufferServer::unregisterMouseCallback( hkVirtualMouseHandler* h )
{
	int idx = m_mouseHandlers.indexOf(h);
	if (idx >= 0)
		m_mouseHandlers.removeAt(idx);
}

void hkVirtualFramebufferServer::registerFileDropCallback( hkVirtualFileDropHandler* h )
{
	if (m_filedropHandlers.indexOf(h) < 0)
		m_filedropHandlers.pushBack(h);
}

void hkVirtualFramebufferServer::unregisterFileDropCallback( hkVirtualFileDropHandler* h )
{
	int idx = m_filedropHandlers.indexOf(h);
	if (idx >= 0)
		m_filedropHandlers.removeAt(idx);
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
