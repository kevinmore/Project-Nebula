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
#include <Common/Base/Algorithm/Sort/hkSort.h>

int sort_main()
{
	const int NTEST = 1000;
	const int ASIZE = 1000;
	const int MAX = 100000;

	for(int i=0; i<NTEST; ++i)
	{
		int array[ASIZE];
		const int n = int(hkUnitTest::randRange( 0, hkReal(ASIZE) ));
		int j;

		for(j=0; j<n; ++j)
		{
			array[j] = (int)hkUnitTest::randRange(1, (float)MAX);
		}

		if (i&1)
		{
			hkAlgorithm::quickSort(array, n);
		} 
		else
		{
			hkAlgorithm::heapSort(array, n);
		}
		
		for(j=0; j<n-1; ++j)
		{
			if(array[j] > array[j+1])
			{
				break;
			}
		}

		if( n > 0)
		{
			HK_TEST2(j == n-1, " sort failed on iteration " << i << " element "<< j << (i&1?"(quicksort)":"(heapsort)") );
		}

	}

	// RADIX sort
	{
		for(int i=0; i<NTEST; ++i)
		{
			hkRadixSort::SortData32 array[ASIZE];
			hkRadixSort::SortData32 buffer[ASIZE];
			const int n = (~0x3) & int(hkUnitTest::randRange( 0, hkReal(ASIZE) ));
			int j;

			for(j=0; j<n; ++j)
			{
				array[j].m_key = (int)hkUnitTest::randRange(1, hkReal(0x10000000));
			}

			hkRadixSort::sort32(array, n, buffer);

			for(j=0; j<n-1; ++j)
			{
				if(array[j].m_key > array[j+1].m_key)
				{
					break;
				}
			}

			if( n > 0)
			{
				HK_TEST2(j == n-1, " radix sort failed on iteration " << i << " element "<< j << (i&1?"(quicksort)":"(heapsort)") );
			}
		}
	}

	// List sort
	{
		hkPseudoRandomGenerator prng(0);
		for( int asize = 2; asize < 100; ++asize )
		{
			for( int iter = 0; iter < 10; ++iter )
			{
				// create random linked list
				hkAlgorithm::ListElement* head;
				hkArray<hkAlgorithm::ListElement>::Temp arr(asize);
				{
					hkArray<int>::Temp ind(asize);
					for( int i = 0; i < asize; ++i )
					{
						ind[i] = i;
					}
					prng.shuffle( ind.begin(), asize);

					for( int i = 0; i < asize-1; ++i )
					{
						arr[ ind[i] ].next = &arr[ ind[i+1] ];
					}
					arr[ ind[asize-1] ].next = HK_NULL;
					head = &arr[ ind[0] ];
				}

				// sort
				head = hkAlgorithm::sortList( head );

				// test it was sorted
				int count = 0;
				while( head != HK_NULL )
				{
					count += 1;
					HK_TEST( head->next == HK_NULL || head < head->next );
					head = head->next;
				}
				HK_TEST( count == asize );
			}
		}
	}
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(sort_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
