/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/hkCommandRouter.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Visualize/hkProcess.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>

// This class is very similar to hkBufferedStreamReader, but uses a dynamic hkArray for buffering reads
class hkReplayStreamReader : public hkStreamReader
{
public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);

	hkReplayStreamReader(hkStreamReader* s)
		: m_stream(s), m_current(0) 
	{
		HK_ASSERT( 0x3a82bd80, m_stream != HK_NULL );
		m_stream->addReference();
	}

	/// Removes reference to the reader in the constructor if applicable.
	~hkReplayStreamReader() 
	{ 
		m_stream->removeReference(); 
	}

	virtual int read(void* buf, int nbytes)
	{
		const int bufSize = m_buf.getSize();
		// How many additional bytes must be read from our child stream to fulfill the read request
		int childStreamNbytes = m_current + nbytes - bufSize;

		// If this read request requires our internal buffer to be filled with new data, forward a read request of the needed size onto the child stream reader
		if (childStreamNbytes > 0)
		{
			// Expand our internal buffer to accomodate the expected child stream read size, and read from the child stream
			char* dataNew = m_buf.expandBy(childStreamNbytes);
			int childStreamNbytesRead = m_stream->read(dataNew, childStreamNbytes);

			// The number of bytes we can read from our existing buffer to service the read request
			const int internalBufNbytes = bufSize - m_current;
			const int nbytesRead = internalBufNbytes + childStreamNbytesRead;
			const int newBufSize = m_current + nbytesRead;

			// Refit our internal buffer to account for a read not returning the expected number of bytes
			m_buf.setSize(newBufSize);

			// Update the number of bytes being memcpy'd from our internal buffer
			nbytes = nbytesRead;
		}

		// Just copy the existing data from our internal buffer to the requested buffer
		hkString::memCpy(buf, m_buf.begin() + m_current, nbytes);
		m_current += nbytes;
		return nbytes;
	}

	virtual hkBool isOk() const 
	{ 
		return m_stream->isOk(); 
	}

	// Go back to the point in the stream this reader was in at the time it was bound to its child reader
	void rewind() 
	{ 
		m_current = 0; 
	}

private:
	hkStreamReader* m_stream; // child stream

	hkUint32 m_current; // current byte index in the array which will be read from
	hkArray<char> m_buf;
};

void hkCommandRouter::registerProcess(hkProcess* handler)
{
	hkUint8* commands = HK_NULL;
	int numCommands = 0;
	handler->getConsumableCommands(commands, numCommands);	

	for (int c = 0; c < numCommands; ++c)
	{
		hkUint8 cmd = commands[c];
		m_commandMap.insert( cmd, handler );
	}
}

void hkCommandRouter::unregisterProcess(hkProcess* handler)
{
	hkUint8* commands = HK_NULL;
	int numCommands = 0;
	handler->getConsumableCommands(commands, numCommands);	
	for (int c = 0; c < numCommands; ++c)
	{
		hkUint8 cmd = commands[c];
		m_commandMap.remove( cmd, handler );
	}
}

hkBool hkCommandRouter::consumeCommands(hkDisplaySerializeIStream* stream)
{
	hkUint8 command = stream->read8u();
	while (stream->isOk() && (command != hkVisualDebuggerProtocol::COMMAND_ACK))
	{
		bool commandsConsumed = false;
		if (m_commandMap.hasKey(command))
		{
			hkStreamReader* origReader = stream->getStreamReader();

			// Make the stream replayable so that commands can be rewinded and handled by all processes registered to them
			hkReplayStreamReader* replay = new hkReplayStreamReader(origReader);
			stream->setStreamReader(replay);

			// Iterate through all processes registered with the current command
			for (hkPointerMultiMap<hkUint8, hkProcess*>::Iterator it = m_commandMap.findKey(command); m_commandMap.isValid(it); it = m_commandMap.getNext(it, command))
			{
				hkProcess* handler = m_commandMap.getValue(it);
				if (handler)
				{
					commandsConsumed = true;
					HK_ASSERT2(0x7ce319b1, handler->m_inStream == stream, "VDB CommandRouter: Something gone astray with the inout streams.." );	
					handler->consumeCommand(command);

					// Prepare the stream for the next process which might read from it
					replay->rewind();
				}
			}

			// Restore the original stream reader
			stream->setStreamReader(origReader);
			replay->removeReference();
		}
		if (!commandsConsumed) 
		{
			// XXXXX Hack for older mouse picking in the client
			hkVector4 t;
			if (command == 0xB0) 
			{
				stream->readQuadVector4(t);
				stream->read64u();
			}
			else if (command == 0xB1)
			{
				stream->readQuadVector4(t);
			}
			hkStringBuf str;
			str.printf("VDB: Found a command (%x) with no handler. Could corrupt the stream.", int(command) );
			HK_WARN( 0xfdf334d, str.cString() );
		}

		command = stream->read8u();
	}
	
	return true;
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
