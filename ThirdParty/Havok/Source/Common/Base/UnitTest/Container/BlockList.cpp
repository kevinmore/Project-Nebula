/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/BlockList/hkBlockList.h>

#define NUM_ELEMENTS_TO_ADD 1000
// These need to add up to NUM_ELEMENTS_TO_ADD
#define FIRST_BATCH 128
#define SECOND_BATCH 0
#define THIRD_BATCH 384
#define FOURTH_BATCH 488

HK_COMPILE_TIME_ASSERT( FIRST_BATCH + SECOND_BATCH + THIRD_BATCH + FOURTH_BATCH == NUM_ELEMENTS_TO_ADD );

template<typename T, int BLOCK_SIZE>
void testNextBatch( typename hkBlockList<T, BLOCK_SIZE>::BatchConsumer& consumer, int numElements, int& currentCount )
{
	consumer.setNumElements( numElements );
	int lastCount = currentCount;
	T* t;
	int batchSize;
	while ( ( batchSize = consumer.accessBatch( t ) ) > 0 )
	{
		for ( int j = 0; j < batchSize; ++j )
		{
			HK_TEST( t[j].m_val == currentCount + j );
		}
		currentCount += batchSize;
	}

	HK_TEST( currentCount - lastCount == numElements );
}

template<typename T, int BLOCK_SIZE>
void block_list_test_batches()
{
	hkBlockList<T, BLOCK_SIZE> blockList;

	{
		typename hkBlockList<T, BLOCK_SIZE>::BatchWriter writer;
		writer.setToStartOfList( &blockList );
		
		T source[NUM_ELEMENTS_TO_ADD];
		for ( int i = 0; i < NUM_ELEMENTS_TO_ADD; ++i )
		{
			source[i].m_val = i;
		}

		writer.writeBatch( &source[0], FIRST_BATCH );
		writer.writeBatch( &source[FIRST_BATCH], SECOND_BATCH );
		writer.writeBatch( &source[FIRST_BATCH + SECOND_BATCH], THIRD_BATCH );
		writer.writeBatch( &source[FIRST_BATCH + SECOND_BATCH + THIRD_BATCH], FOURTH_BATCH );

		writer.finalize();
	}

#if defined(HK_DEBUG)
	HK_TEST( blockList.getTotalNumElems() == NUM_ELEMENTS_TO_ADD );
#endif

	{
		typename hkBlockList<T, BLOCK_SIZE>::BatchConsumer consumer;
		consumer.setToStartOfList( &blockList );
		int count = 0;
		testNextBatch<T, BLOCK_SIZE>( consumer, FIRST_BATCH, count );
		testNextBatch<T, BLOCK_SIZE>( consumer, SECOND_BATCH, count );
		testNextBatch<T, BLOCK_SIZE>( consumer, THIRD_BATCH, count );
		testNextBatch<T, BLOCK_SIZE>( consumer, FOURTH_BATCH, count );

		consumer.finalize();
	}

#if defined(HK_DEBUG)
	blockList.checkEmpty();
#endif
}

struct UnalignedTestStructure
{
	int m_val;
};

struct AlignedTestStructure
{
	HK_ALIGN16( int m_val );
};

int fixed_size_block_list_main()
{
	block_list_test_batches<UnalignedTestStructure, 512>();
	block_list_test_batches<UnalignedTestStructure, 128>();

	block_list_test_batches<AlignedTestStructure, 512>();
	block_list_test_batches<AlignedTestStructure, 128>();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(fixed_size_block_list_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
