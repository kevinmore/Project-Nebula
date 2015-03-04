/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#if defined(HK_PLATFORM_SPU)
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>

// Uncomment the following line to enable debug traces
//define HK_DMA_MANAGER_TEST_DEBUG
#	if defined(HK_DMA_MANAGER_TEST_DEBUG)
#		include <spu_printf.h>
#	endif


static void testPutToMainMemorySmallAnySize()
{
	hkUint8* dataPpu = hkAllocateChunk<hkUint8>(16, HK_MEMORY_CLASS_BASE);
	hkUint8 dataWriten[16];
	hkUint8 dataRead[16];

	// Test all transfer sizes below 16 bytes
	for (int size = 1; size < 16; ++size)
	{
		// Test all different alignments with a 16 byte block
		for (int start = 0; start <= (16 - size); ++start)
		{
			// Clear PPU
			hkString::memClear16(dataRead, 1);
			hkSpuDmaManager::putToMainMemoryAndWaitForCompletion(dataPpu, dataRead, 16, hkSpuDmaManager::WRITE_NEW);

			// Initialize SPU data
			hkString::memSet4(dataWriten, 0xFFFFFFFF, 4);
			for (int i = 0; i < size; ++i)
			{
				dataWriten[start + i] = i + 1;
			}

			// Write to PPU
			hkSpuDmaManager::putToMainMemorySmallAnySize(dataPpu + start, dataWriten + start, size, hkSpuDmaManager::WRITE_NEW);
			hkSpuDmaManager::waitForAllDmaCompletion();

			// Read back from PPU
			hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(dataRead, dataPpu, 16, hkSpuDmaManager::READ_ONLY);

			// Setup expected data
			hkString::memClear16(dataWriten, 1);
			for (int i = 0; i < size; ++i)
			{
				dataWriten[start + i] = i + 1;
			}

			// Compare expected with what we actually got from PPU
			if (hkString::memCmp(dataWriten, dataRead, 16))
			{
			#if defined(HK_DMA_MANAGER_TEST_DEBUG)
				spu_printf("size %d start %d\n", size, start);
				spu_printf("dataWriten %02X %02X %02X %02X %02X %02X %02X %02X ", dataWriten[0], dataWriten[1],
					dataWriten[2], dataWriten[3], dataWriten[4], dataWriten[5], dataWriten[6], dataWriten[7]);
				spu_printf("%02X %02X %02X %02X %02X %02X %02X %02X\n", dataWriten[8], dataWriten[9],
					dataWriten[10], dataWriten[11], dataWriten[12], dataWriten[13], dataWriten[14], dataWriten[15]);
				spu_printf("dataRead   %02X %02X %02X %02X %02X %02X %02X %02X ", dataRead[0], dataRead[1],
					dataRead[2], dataRead[3], dataRead[4], dataRead[5], dataRead[6], dataRead[7]);
				spu_printf("%02X %02X %02X %02X %02X %02X %02X %02X\n", dataRead[8], dataRead[9],
					dataRead[10], dataRead[11], dataRead[12], dataRead[13], dataRead[14], dataRead[15]);
			#endif
				HK_TEST2(false, "Error in testPutToMainMemorySmallAnySize");
			}
		}
	}

	hkDeallocateChunk<hkUint8>(dataPpu, 16, HK_MEMORY_CLASS_BASE);
}
#endif

int dmaManagerTest_main()
{
#if defined(HK_PLATFORM_SPU)
	testPutToMainMemorySmallAnySize();
#endif
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(dmaManagerTest_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Spu/" );

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
