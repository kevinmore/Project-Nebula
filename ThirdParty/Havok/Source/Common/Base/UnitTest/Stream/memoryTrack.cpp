/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>

static int memoryTrack_main()
{		
	// Operations within a single section
	{
		hkMemoryTrack track(1024);

		hkMemoryTrackStreamReader reader(&track, hkMemoryTrackStreamReader::MEMORY_INPLACE, true);
		hkMemoryTrackStreamWriter writer(&track, hkMemoryTrackStreamWriter::TRACK_BORROW);

		hkIArchive inArchive(&reader);
		hkOArchive outArchive(&writer);

		// Write data
		for( int i = 0; i < 256; ++i )
		{
			outArchive.write32(i);
		}

		// Read some of it
		for( int i = 0; i < 64; ++i )
		{
			inArchive.read32();
		}

		// Read all of it and make sure it's the same as written to
		for( int i = 64; i < 256; ++i )
		{
			HK_TEST( i == inArchive.read32() );
		}

		// Check that the stream is still OK.  Haven't read over the end yet.
		HK_TEST( inArchive.isOk() );		
		
		// Read one more byte to make it not OK
		inArchive.read32();
		HK_TEST( !inArchive.isOk() );

		// Write some more data
		for( int i = 0; i < 128; ++i )
		{
			outArchive.write32(i);
		}

		// Read that data back in to make sure it's ok	
		for( int i = 0; i < 128; ++i )
		{
			HK_TEST( inArchive.read32() == i );
		}	

		// Check that its OK
		HK_TEST(inArchive.isOk());		
	}

	// Test reads and writes of different sizes.
	{
		const int trackSize = 1024;
		hkMemoryTrack track(trackSize);

		hkMemoryTrackStreamReader reader(&track, hkMemoryTrackStreamReader::MEMORY_INPLACE, true);
		hkMemoryTrackStreamWriter writer(&track, hkMemoryTrackStreamWriter::TRACK_BORROW);

		hkIArchive inArchive(&reader);
		hkOArchive outArchive(&writer);
		
		hkPseudoRandomGenerator rand(123);

		const int maxWriteSize = 15562;
		hkArray<hkInt32> randomWriteSizes;
		for( int i = 0; i < 1000; ++i )
		{			
			randomWriteSizes.pushBack( (int)rand.getRandRange(0, (hkReal)trackSize * 10) );			
		}
				
		hkArray<hkInt8> inData(maxWriteSize, 0);
		hkArray<hkInt8> outData(maxWriteSize, 0);
		for( int j = 0; j < maxWriteSize; ++j )
		{
			inData[j] = (hkChar)j;
		}		

		for( int i = 0; i < randomWriteSizes.getSize(); ++i )
		{
			outArchive.writeArray8(inData.begin(), randomWriteSizes[i]);
			
			// Read in different chunks
			int numRead = 0;
			while( numRead < randomWriteSizes[i] )
			{
				int readAmount = hkMath::min2( randomWriteSizes[i] - numRead, (int)rand.getRandRange( 1, trackSize * 2 ));				
				inArchive.readArray8(outData.begin() + numRead, readAmount);
				numRead += readAmount;
			}						
			
			for( int j = 0; j < randomWriteSizes[i]; ++j )
			{				
				HK_TEST( outData[j] == (hkChar)j);
			}
		}

		// Since we read all of the data the sectors should be clear
		HK_TEST( track.m_sectors.getSize() == 0 );
	}
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(memoryTrack_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__  );

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
