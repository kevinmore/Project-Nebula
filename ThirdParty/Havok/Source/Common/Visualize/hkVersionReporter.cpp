/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Visualize/hkVisualize.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Visualize/hkVersionReporter.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>
#include <Common/Serialize/Util/hkStructureLayout.h>

// Protocol Revision Information
// Must be updated for any change to the visual debugger protocol
// Extensions to the protocol do not necessarily make the the client
// and server incompatible.
//
// Client can support more packets then the server is going to send...
// Server can send packets and the client can ignore them.
// Visa-versa for both of the above.
//
// If an existing packet changes its format then the minimum version
// must be updated to match the new protocol.
// (Unless the packet has an internal versioning system.)
//
// Date      Date      Version   Information
// (HEAD)    (Release) ver(min)
// --------- --------- --------- -----------------------------------------
// 20030430  ?         200(200)  First versioned protocol version. (BC)
//
// 

int hkVersionReporter::m_protocolMinimumCompatible = 2300;
int hkVersionReporter::m_protocolVersion = 10100; 
// stats changed for 3.2, 
// new instance geom for 4.5, 
// 3d text support in 4.5.1, 
// collidable ids for filtering in 7.0, 
// added compiler info 7001
// Added inspect process tag to enable picking etc based on collidable id
// Added hash & cache support in 7500

// read as XYY -> vX.YY

#if defined(HK_PLATFORM_PS3_PPU)
#	if (HK_POINTER_SIZE==8)
#		define HK_PLATFORM_INFO_STRING "PS3_64"
#	else
#		define HK_PLATFORM_INFO_STRING "PS3_32"
#	endif
#elif defined(HK_PLATFORM_WIN32)
#	if (HK_POINTER_SIZE==8)
#		define HK_PLATFORM_INFO_STRING "PC_64"
#	else
#		define HK_PLATFORM_INFO_STRING "PC_32"
#	endif
#elif defined(HK_PLATFORM_PS4)
#	define HK_PLATFORM_INFO_STRING "PS4_64"	
#elif defined(HK_PLATFORM_XBOX)
#	define HK_PLATFORM_INFO_STRING "XBOX"
#elif defined(HK_PLATFORM_XBOX360)
#	define HK_PLATFORM_INFO_STRING "XBOX360"
#elif defined(HK_PLATFORM_WIIU)
// Pretend to be a Wii for backwards compatibility with prev VDB clients
#	define HK_PLATFORM_INFO_STRING "WII"
#elif defined(HK_PLATFORM_GC)
#	if defined(HK_PLATFORM_RVL)
#		define HK_PLATFORM_INFO_STRING "WII"
#	else
#		define HK_PLATFORM_INFO_STRING "GC"
#	endif
#elif defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_MACPPC)
#	define HK_PLATFORM_INFO_STRING "MAC"
#elif defined(HK_PLATFORM_IOS) 
#	define HK_PLATFORM_INFO_STRING "IOS"
#elif defined(HK_PLATFORM_LINUX)
#	if (HK_POINTER_SIZE==8)
#		define HK_PLATFORM_INFO_STRING "LINUX_64"
#	else
#		define HK_PLATFORM_INFO_STRING "LINUX_32"
#	endif
#elif defined(HK_PLATFORM_CTR)
#	define HK_PLATFORM_INFO_STRING "CTR"
#elif defined(HK_PLATFORM_PSVITA)
#	define HK_PLATFORM_INFO_STRING "PSVITA"
#elif defined(HK_PLATFORM_ANDROID)
#	define HK_PLATFORM_INFO_STRING "ANDROID"
#else
#	define HK_PLATFORM_INFO_STRING "UNKNOWN"
#endif

#if defined(HK_COMPILER_INTEL)
#	define HK_COMPILER_INFO_STRING "INTEL"
#	define HK_COMPILER_VERSION_INFO HK_COMPILER_INTEL_VERSION
#elif defined(HK_COMPILER_MSVC)
#	define HK_COMPILER_INFO_STRING "MSVC"
#	define HK_COMPILER_VERSION_INFO HK_COMPILER_MSVC_VERSION
#elif defined(HK_COMPILER_SNC) 
#	define HK_COMPILER_INFO_STRING "SNC"
#	define HK_COMPILER_VERSION_INFO HK_COMPILER_GCC_VERSION // this stores the snc compiler version also
#elif defined(HK_COMPILER_ARMCC)
#	define HK_COMPILER_INFO_STRING "ARMCC"
#	define HK_COMPILER_VERSION_INFO 0x0
#elif defined(HK_COMPILER_GCC)
#	define HK_COMPILER_INFO_STRING "GCC"
#	define HK_COMPILER_VERSION_INFO HK_COMPILER_GCC_VERSION
#elif defined(HK_COMPILER_MWERKS)
#	define HK_COMPILER_INFO_STRING "MWERKS"
#	define HK_COMPILER_VERSION_INFO __MWERKS__ // this is the version as a hex
#else
#	define HK_COMPILER_VERSION_UNKNOWN
#endif

hkResult hkVersionReporter::sendVersionInformation( hkStreamWriter* connection )
{
	// send the data chunk describing the version information
	hkArray<char> rawData;
	hkDisplaySerializeOStream commandStream(rawData);

	commandStream.write8u( hkVisualDebuggerProtocol::HK_VERSION_INFORMATION );

	// version and minimum compatible version
	commandStream.write32(m_protocolVersion);
	commandStream.write32(m_protocolMinimumCompatible);

	// send magic platform string
	const char* platformInfo = HK_PLATFORM_INFO_STRING;
	int length = hkString::strLen(platformInfo);
	if(length > 65535)
	{
		length = 65535;
	}
	commandStream.write16u((unsigned short)length);
	commandStream.writeRaw(platformInfo, length);

	// send compiler string
#ifndef HK_COMPILER_VERSION_UNKNOWN
	hkStringBuf compilerInfo;
#	if defined(HK_COMPILER_MWERKS)
		compilerInfo.printf("%s:%x", HK_COMPILER_INFO_STRING, HK_COMPILER_VERSION_INFO);
#	elif defined(HK_COMPILER_SNC) && defined(HK_PLATFORM_PSVITA)
		compilerInfo.printf("SNC:%x", __SN_FULL_VER__); 
#	else
		compilerInfo.printf("%s:%i", HK_COMPILER_INFO_STRING, HK_COMPILER_VERSION_INFO);
#	endif
#else
	hkStringBuf compilerInfo("UNKNOWN");
#endif
	length = compilerInfo.getLength();
	if(length > 65535)
	{
		length = 65535;
	}
	commandStream.write16u((unsigned short)length);
	commandStream.writeRaw(compilerInfo.cString(), length);

	// actually write the packet.
	hkDisplaySerializeOStream connectionStream(connection);
	connectionStream.write32( rawData.getSize() );
	connectionStream.writeRaw(rawData.begin(), rawData.getSize());

	return HK_SUCCESS;
}

hkResult HK_CALL hkVersionReporter::sendStructureLayout( hkStreamWriter* connection )
{
	hkDisplaySerializeOStream commandStream(connection);

	int packetSize = 1 + 4;
	commandStream.write32u(packetSize);
	commandStream.write8u(hkVisualDebuggerProtocol::HK_SERVER_LAYOUT);

	// the layout of this server  (4100 etc)
	commandStream.write8u( hkStructureLayout::HostLayoutRules.m_bytesInPointer );
	commandStream.write8u( hkStructureLayout::HostLayoutRules.m_littleEndian );
	commandStream.write8u( hkStructureLayout::HostLayoutRules.m_reusePaddingOptimization );
	commandStream.write8u( hkStructureLayout::HostLayoutRules.m_emptyBaseClassOptimization );

	return HK_SUCCESS;
}

hkResult HK_CALL hkVersionReporter::receiveVersionInformation( hkStreamReader* connection, int& protocolVersion, int& protocolMinimumCompatible, hkStringPtr& platformString, hkStringPtr& compilerString )
{
	hkDisplaySerializeIStream chunkStream(connection);
	return receiveVersionInformation( chunkStream, protocolVersion, protocolMinimumCompatible, platformString, compilerString );
}

hkResult HK_CALL hkVersionReporter::receiveVersionInformation( hkDisplaySerializeIStream& chunkStream, int& protocolVersion, int& protocolMinimumCompatible, hkStringPtr& platformString, hkStringPtr& compilerString )
{
	protocolVersion = chunkStream.read32();
	protocolMinimumCompatible = chunkStream.read32();

	// read magic platform string
	unsigned int length = chunkStream.read16u();

	hkInplaceArray<char, 4096> buffer;
	buffer.setSize(length + 1);

	chunkStream.readRaw(buffer.begin(), length);
	buffer[length] = '\0';

	platformString = buffer.begin();

	// read compiler string if available
	if (protocolVersion >= HK_VERSION_SUPPORTS_COMPILER_STRING )
	{
		length = chunkStream.read16u();

		buffer.setSize(length + 1);

		chunkStream.readRaw(buffer.begin(), length);
		buffer[length] = '\0';

		compilerString = buffer.begin();
	}
	else
	{
		compilerString = "UNAVAILABLE";
	}

	return HK_SUCCESS;
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
