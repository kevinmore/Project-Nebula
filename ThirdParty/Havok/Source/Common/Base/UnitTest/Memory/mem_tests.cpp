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

static void clearBuffer( char* buffer, int nBytes, char fill)
{
	for (int i=0; i< nBytes; i++)
	{
		*buffer++ = fill;
	}
}

// Defeating the optimiser... (see bottom of memCpyTest)
int global_variable_that_is_zero = 0;
int* pointer_to_a_zero = &global_variable_that_is_zero;


static void memCpyTest()
{
	const int MAX_ALIGN = 32;
	const int MAX_SIZE  = 128;
	const int MAX_BUFFER_SIZE = MAX_SIZE + MAX_ALIGN;
	char* buffer1 = hkAllocateChunk<char>( MAX_BUFFER_SIZE, HK_MEMORY_CLASS_DEMO );
	char* buffer2 = hkAllocateChunk<char>( MAX_BUFFER_SIZE, HK_MEMORY_CLASS_DEMO );

	hkPseudoRandomGenerator rand(123);

	// Initialize buffers
	clearBuffer( buffer1, MAX_BUFFER_SIZE, 0x7F );
	clearBuffer( buffer2, MAX_BUFFER_SIZE, 0x7F );

	for (int alignSrc=0; alignSrc < MAX_ALIGN; alignSrc++)
	{
		char * src = buffer1 + alignSrc;
		
		for (int alignDst=0; alignDst < MAX_ALIGN; alignDst++)
		{
			char * dst = buffer2 + alignDst;

			for (int size = 0; size <= MAX_SIZE; size = size+1 +(size>>3) )
			{
				// Fill buffer 1
				char* srcCopy = src;
				for (int s=0; s < size; s++)
				{
					srcCopy[s] = (char)rand.getRandChar(255);
				}

				// memcpy
				hkString::memCpy(dst, src, size);

				// Test
				for (int t=0; t < size; t++)
				{
					HK_TEST( dst[t] == src[t] );
				}

				clearBuffer( dst, size, 0x7F );

				// hkMemUtil::memCpy
				hkMemUtil::memCpy(dst, src, size);

				// Test
				for (int t=0; t < size; t++)
				{
					HK_TEST( dst[t] == src[t] );
				}

				clearBuffer( dst, size, 0x7F );

				// hkMemUtil::memCpy
				hkMemUtil::memCpyBackwards(dst, src, size);

				// Test
				for (int t=0; t < size; t++)
				{
					HK_TEST( dst[t] == src[t] );
				}
			}
		}
	}

	// Test memMove
	clearBuffer( buffer2, MAX_BUFFER_SIZE, 0x7F );

	for (int alignSrc=1; alignSrc < MAX_ALIGN; alignSrc++)
	{
		char * src = buffer1 + alignSrc;
		char * srcCopy = buffer2 + alignSrc;

		for (int size = 0; size <= MAX_SIZE-alignSrc; size++)
		{
			// Fill buffer 1
			for (int s=0; s < size; s++)
			{
				src[s] = (char)rand.getRandChar(255);
			}

			// hkMemUtil::memCpy
			hkMemUtil::memCpy(srcCopy, src, size);

			// hkMemUtil::memMove
			char * dst = src+alignSrc;
			hkMemUtil::memMove(dst, src, size);

			// Test
			for (int t=0; t < size; t++)
			{
				HK_TEST( dst[t] == srcCopy[t] );
			}
		}
	}


	// mem{cpy,set} with a NULL source or destination is undefined behaviour,
	// even with a length of 0. Most implementations allow this, and since some
	// Havok code may rely on it, it's tested here.
	// (the HK_TESTs are fairly meaningless, the important thing is that this code
	// shouldn't crash).
	// To avoid the optimiser optimising away these "no-ops", we don't pass a
	// literal zero but read it from a global variable.
	const hkUint32 magic = 0x8932ff94;
	hkUint32 d = magic;
	hkString::memCpy(&d, HK_NULL, *pointer_to_a_zero);
	HK_TEST(d == magic);
	hkString::memCpy(HK_NULL, &d, *pointer_to_a_zero);
	HK_TEST(d == magic);
	hkString::memCpy(HK_NULL, HK_NULL, *pointer_to_a_zero);
	hkString::memSet(HK_NULL, 42, *pointer_to_a_zero);

	hkDeallocateChunk<char>( buffer1, MAX_BUFFER_SIZE, HK_MEMORY_CLASS_DEMO );
	hkDeallocateChunk<char>( buffer2, MAX_BUFFER_SIZE, HK_MEMORY_CLASS_DEMO );
}

int mem_tst_main()
{
	memCpyTest();
	memCpyTest(); // Must be run twice to reproduce Radix failure.
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(mem_tst_main, "Slow", "Common/Test/UnitTest/Base/", __FILE__);

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
