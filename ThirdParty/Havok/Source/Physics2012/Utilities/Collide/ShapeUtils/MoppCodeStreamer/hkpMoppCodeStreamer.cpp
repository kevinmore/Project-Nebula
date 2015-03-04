/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Utilities/Collide/ShapeUtils/MoppCodeStreamer/hkpMoppCodeStreamer.h>


hkpMoppCode* HK_CALL hkpMoppCodeStreamer::readMoppCodeFromArchive(hkIArchive &inputArchive)
{
		// Read in the header info, see writeMoppCodeToArchive()
	HK_ALIGN_REAL(hkFloat32 fheader[4]);
	fheader[0] = inputArchive.readFloat32();
	fheader[1] = inputArchive.readFloat32();
	fheader[2] = inputArchive.readFloat32();
	fheader[3] = inputArchive.readFloat32();

	hkVector4 header; header.load<4>(&fheader[0]);

		// Read in the MOPP byte code size
	int byteCodeSize = inputArchive.read32();

		// Put in some very basic "safety" checks here to detect file corruption. It is "unlikely" that
		// the offset values will be very large in magnitude.
	HK_ASSERT2(0x6a8a18e5, (header(0) > -1e9f) && (header(0) < 1e9f), "Header of MoppCode Archive is corrupt!");
	HK_ASSERT2(0x51cc39fe, (header(1) > -1e9f) && (header(1) < 1e9f), "Header of MoppCode Archive is corrupt!");
	HK_ASSERT2(0x7a84f907, (header(2) > -1e9f) && (header(2) < 1e9f), "Header of MoppCode Archive is corrupt!");
		// Also, assume size of data less than ~100 meg! This may be overly conservative!
	HK_ASSERT2(0x5f442ae7, byteCodeSize < 10000000, "Input Archive data corrupt, byte code size is huge!");


	hkpMoppCode* code = new hkpMoppCode();
	code->m_data.setSize( byteCodeSize );
	code->m_info.m_offset = header;

		// Read in the byte code
	HK_ON_DEBUG(int numBytesRead =) inputArchive.readRaw( const_cast<hkUint8*>(&code->m_data[0]), byteCodeSize);
	HK_ASSERT2(0x1e5e6c92, numBytesRead == byteCodeSize, "Input Archive data corrupt, not enough bytes read!");

	hkpMoppCode::BuildType buildType;
	hkInt8 tempType;
	{
		// The format of the streamed MOPP code changed between Havok 5.5 and Havok 6.0 - previously the m_buildType wasn't stored.
		// If it's there, readRaw will return 1 and we're fine.
		// If it's not there, we're at the end of the file so readRaw will return 0, in which case we do our best to patch things up.

		int numBytesReadForType = inputArchive.readRaw(&tempType, 1);
		buildType = (hkpMoppCode::BuildType) tempType;

		if ( numBytesReadForType == 0) 
		{
			HK_WARN_ONCE(0x7ba85d07,	"Unable to read hkpMoppCode::BuildType in hkpMoppCodeStreamer. " \
										"This could happen if you are reading an old (Havok 5.5 or earlier) MOPP. " \
										"Re-exporting the MOPP will fix this warning. " \
										"Defaulting m_buildType to BUILT_WITHOUT_CHUNK_SUBDIVISION, which will prevent the MOPP from runnning on the SPU.");

			// We have to assume the worst - that the MOPP was not built with chunk subdivision. This will force the MOPP to be simulated on the PPU.
			buildType =  hkpMoppCode::BUILT_WITHOUT_CHUNK_SUBDIVISION;
		}

		HK_ASSERT2( 0x54977d10, (buildType == hkpMoppCode::BUILT_WITH_CHUNK_SUBDIVISION) || (buildType == hkpMoppCode::BUILT_WITHOUT_CHUNK_SUBDIVISION), "Invalid hkpMoppCode::BuildType." );
	}

	code->m_buildType =  buildType;
	
	return code;
}

void HK_CALL hkpMoppCodeStreamer::writeMoppCodeToArchive(const hkpMoppCode* code, hkOArchive &outputArchive)
{
		// Write out the header info
	HK_ALIGN_REAL(hkFloat32 foffset[4]);
	code->m_info.m_offset.store<4>(&foffset[0]);

	outputArchive.writeFloat32(foffset[0]);
	outputArchive.writeFloat32(foffset[1]);
	outputArchive.writeFloat32(foffset[2]);
	outputArchive.writeFloat32(foffset[3]);
			
		// Write out the MOPP byte code size
	outputArchive.write32(code->m_data.getSize());

		// Write out the byte code
	outputArchive.writeRaw(&code->m_data[0], code->m_data.getSize());

		// Write out the build type. We do this last for backwards compatibility
	outputArchive.write8((hkInt8) code->m_buildType);

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
