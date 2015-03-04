/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Container/BitField/hkBitField.h>

#if 0
struct hk1AxisSweep_Iterator_collideSelf
{
	hk1AxisSweep_Iterator_collideSelf(hk1AxisSweep::AabbInt* pa, int numA);

	hkBool32 isValid() const;
	void next();
	void getKeyPair( hkKeyPair& pair );

private:
	const hk1AxisSweep::AabbInt* m_aabbs;
	int m_numAabbs;

	int m_current;
	int m_potential;
};





struct hk1AxisSweep_Iterator_collideAB
{
public:
	hk1AxisSweep_Iterator_collideAB(hk1AxisSweep::AabbInt* pa, int numA, hk1AxisSweep::AabbInt* pb, int numB );
	hkBool32 isValid() const;
	void next();
	void getKeyPair( hkKeyPair& pair );

private:
	void _updatePtrs();

	hkBool32 aIsBigger;
	const hk1AxisSweep::AabbInt* currentPtr;
	const hk1AxisSweep::AabbInt* potentialPtr;

	const hk1AxisSweep::AabbInt* m_pa;
	const hk1AxisSweep::AabbInt* m_pb;

	int m_numA;
	int m_numB;

};


#endif



static void setupAabbs( hkArray<hkAabb>& aabbs, hkArray<hk1AxisSweep::AabbInt>& aabbInt, hkPseudoRandomGenerator& rand, int numAabbs, bool validEndMarkers)
{
	aabbs.setSize(numAabbs);
	aabbInt.setSize(numAabbs + 4);

	for (int i=0; i<numAabbs; i++)
	{
		hkVector4 v0, v1;
		rand.getRandomVector11( v0 );
		rand.getRandomVector11( v1 );
		aabbs[i].m_min.setMin(v0, v1);
		aabbs[i].m_max.setMax(v0, v1);

		aabbInt[i].set(aabbs[i], i);
	}

	for (int i=0; i<4; i++)
	{
		aabbInt[numAabbs+i].setEndMarker();
		aabbInt[numAabbs+i].m_max[0] = hkUint32(-1);

		if(validEndMarkers)
		{
			aabbInt[numAabbs+i].m_min[1] = 0;
			aabbInt[numAabbs+i].m_min[2] = 0;
			aabbInt[numAabbs+i].m_max[1] = hkUint32(-1);
			aabbInt[numAabbs+i].m_max[2] = hkUint32(-1);
		}
		else
		{
			aabbInt[numAabbs+i].m_min[1] = 2;
			aabbInt[numAabbs+i].m_min[2] = 2;
			aabbInt[numAabbs+i].m_max[1] = 1;
			aabbInt[numAabbs+i].m_max[2] = 1;

		}
	}
}

static void checkBitfields(const hkArray<hkAabb>& aabbsA, const hkArray<hkAabb>& aabbsB, const hkArray<hkBitField>& overlaps, hkBool checkSymmetric)
{
	int numA  = aabbsA.getSize();
	int numB  = aabbsB.getSize();
	for (int i=0; i<numA; i++)
	{
		for (int j=0; j<numB; j++)
		{
			if (!checkSymmetric && (i==j))
				continue;

			bool floatOverlap = aabbsA[i].overlaps(aabbsB[j]);
			bool sweepOverlap = overlaps[i].get(j) != 0;
			HK_TEST(sweepOverlap == floatOverlap);
		}
	}
}

static void resetBitfields( hkArray<hkBitField>& overlaps, int numA, int numB )
{
	overlaps.setSize(numA);
	for (int i=0; i<numA; i++)
	{
		overlaps[i].resize(0, numB);
		overlaps[i].assignAll( 0 );
	}
}


static void oneAxisSweep__AvsA_test(hkPseudoRandomGenerator& rand,  int numAabbs, bool validEndmarkers)
{
	//int numAabbs = 100;

	hkArray<hkAabb> aabbs; 
	hkArray<hk1AxisSweep::AabbInt> aabbInt; 

	// Setup and sort
	setupAabbs(aabbs, aabbInt, rand, numAabbs, validEndmarkers);
	hk1AxisSweep::sortAabbs(aabbInt.begin(), numAabbs);

	hkArray<hkKeyPair> pairs0, pairs1;

	// Keep track of which overlaps we find
	hkArray<hkBitField> overlaps;
	resetBitfields(overlaps, numAabbs, numAabbs);

	// Get the sweep results directly and store them in an array
	if(numAabbs)
	{
		int current = 0;
		do
		{
			int potential = current + 1;

			HK_ASSERT(0x5f5b97c0, potential < numAabbs+4 && current < numAabbs + 4 );
			while( aabbInt[potential].m_min[0] <= aabbInt[current].m_max[0] )
			{
				HK_ASSERT(0x5f5b97c0, potential < numAabbs+4 && current < numAabbs + 4 );
				if ( !hk1AxisSweep::AabbInt::yzDisjoint( aabbInt[potential], aabbInt[current] ) )
				{
					int i = aabbInt[current  ].getKey();
					int j = aabbInt[potential].getKey();
					overlaps[i].set(j);
					overlaps[j].set(i);

					hkKeyPair& pair = pairs0.expandOne();
					pair.m_keyA = i;
					pair.m_keyB = j;

				}
				potential++;
			}
			
			current++;
		} while ( current < numAabbs -1 );
	}


	// Make sure the brute force overlaps agree with the sweep results
	checkBitfields(aabbs, aabbs, overlaps, false);

	// Clear the bitfields so we can reuse them
	resetBitfields(overlaps, numAabbs, numAabbs);

	// Use an iterator to find the overlapping pairs
	{
		hk1AxisSweep::IteratorAA iter(aabbInt.begin(), numAabbs);
		for (; iter.isValid(); iter.next() )
		{
			hkKeyPair pair;
			iter.getKeyPair(pair);

			overlaps[pair.m_keyA].set(pair.m_keyB);
			overlaps[pair.m_keyB].set(pair.m_keyA);

			pairs1.pushBack(pair);
		}
	
	}

	// Check the bitfields again
	checkBitfields(aabbs, aabbs, overlaps, false);

	// Make sure the two methods give the same results.
	HK_TEST(pairs0.getSize() == pairs1.getSize());
	for (int i=0; i<hkMath::min2(pairs0.getSize(), pairs1.getSize()); i++)
	{
		HK_TEST( pairs0[i].m_keyA == pairs1[i].m_keyA );
		HK_TEST( pairs0[i].m_keyB == pairs1[i].m_keyB );
	}



}

static void oneAxisSweep__AvsB_test( hkPseudoRandomGenerator& rand, int numA, int numB, bool validEndmarkers )
{
	hkArray<hkAabb> aabbsA, aabbsB; 
	hkArray<hk1AxisSweep::AabbInt> a, b;
	
	// Set up and sort the AABBs
	setupAabbs(aabbsA, a, rand, numA, validEndmarkers);
	setupAabbs(aabbsB, b, rand, numB, validEndmarkers);

	hk1AxisSweep::sortAabbs(a.begin(), numA);
	hk1AxisSweep::sortAabbs(b.begin(), numB);

	// Keep track of which overlaps we find
	hkArray<hkBitField> overlaps;
	resetBitfields(overlaps, numA, numB);

	// Get the overlapping pairs
	hkArray<hkKeyPair> pairs0; pairs0.reserve(10);
	{
		// Need to initialize as collide does not
		hkPadSpu<int> numPairsSkipped = 0;
 		int res = hk1AxisSweep::collide(a.begin(), a.getSize() - 4, b.begin(), b.getSize() - 4, pairs0.begin(), pairs0.getCapacity(), numPairsSkipped);
 
 		if (numPairsSkipped > 0)
 		{
 			pairs0.clear();
 			pairs0.reserve(pairs0.getCapacity() + numPairsSkipped);
 			numPairsSkipped = 0;
 			res = hk1AxisSweep::collide(a.begin(), a.getSize() - 4, b.begin(), b.getSize() - 4, pairs0.begin(), pairs0.getCapacity(), numPairsSkipped);
 
 			HK_ASSERT(0x34243, numPairsSkipped == 0);
 		}
 		pairs0.setSizeUnchecked(res);


		for (int pairIdx = 0; pairIdx<pairs0.getSize(); pairIdx++)
		{
			int i = pairs0[pairIdx].m_keyA;
			int j = pairs0[pairIdx].m_keyB;
			overlaps[i].set(j);
		}
	}

	// Make sure the brute force overlaps agree with the sweep results
	checkBitfields(aabbsA, aabbsB, overlaps, true);

	// Clear the bitfields so we can reuse them
	resetBitfields(overlaps, numA, numB);

	// Get the results with an iterator
	hkArray<hkKeyPair> pairs1;
	{
		hk1AxisSweep::IteratorAB iter(a.begin(), numA, b.begin(), numB);
		for (; iter.isValid(); iter.next() )
		{
 			hkKeyPair pair;
 			iter.getKeyPair(pair);
 
 			overlaps[pair.m_keyA].set(pair.m_keyB);
 
 			pairs1.pushBack(pair);
		}
	}

	checkBitfields(aabbsA, aabbsB, overlaps, true);

	// Make sure the two methods agree
	HK_TEST(pairs0.getSize() == pairs1.getSize());
	for (int i=0; i<pairs0.getSize(); i++)
	{
		HK_TEST( pairs0[i].m_keyA == pairs1[i].m_keyA );
		HK_TEST( pairs0[i].m_keyB == pairs1[i].m_keyB );
	}

}

int oneAxisSweep_main()
{
	hkPseudoRandomGenerator rand(12345);

	for (int j=0; j<2; j++)
	{
		bool validEndMarkers = (j==0);
		oneAxisSweep__AvsA_test(rand, 0, validEndMarkers);
		oneAxisSweep__AvsA_test(rand, 1, validEndMarkers);

		oneAxisSweep__AvsB_test(rand, 0, 100, validEndMarkers);
		oneAxisSweep__AvsB_test(rand, 100, 0, validEndMarkers);

		oneAxisSweep__AvsB_test(rand, 1, 100, validEndMarkers);
		oneAxisSweep__AvsB_test(rand, 100, 1, validEndMarkers);

		for (int i=10; i<=100; i+=10)
		{
			oneAxisSweep__AvsA_test(rand, i, validEndMarkers);
			oneAxisSweep__AvsB_test(rand, i, 2*i + 3, validEndMarkers);
		}
	}
	
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(oneAxisSweep_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
