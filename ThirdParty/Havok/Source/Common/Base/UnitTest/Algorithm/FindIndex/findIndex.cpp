/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/FindIndex/hkFindIndex.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

struct DistanceToPointTest
{
	DistanceToPointTest( hkVector4Parameter p ) : m_p(p) { }
	hkReal operator() (hkVector4Parameter p) const { return m_p.distanceToSquared(p).getReal(); }
	hkVector4 m_p;
};

// --------------------- USAGE OF VOLATILE IN UNIT TESTS ---------------------//
// With optimization enabled the compiler might perform comparisons between 
// previously computed floating point values loaded from memory and values just
// computed inside the FPU stack. Values inside the stack will have different
// precision from the ones just loaded, we must force storage of results in memory
// to make sure that the comparisons do make sense.
// One way to force the compiler to do this is storing the values that will be
// compared in volatile variables.
// ---------------------------------------------------------------------------//

int findIndex_main()
{
	const int NTEST = 500;
	int ASIZE[] = {1, 10, 100};
	//const int ASIZE = 1000;

	hkPseudoRandomGenerator rand(6);
	hkSimdReal maxSr; maxSr.setFromFloat( 1000.0f );
	
	for (int n = 0; n < (int) HK_COUNT_OF(ASIZE); n++)
	{
		hkArray<hkVector4> vectors(ASIZE[n]);
		for (int i=0; i<ASIZE[n]; i++)
		{
			rand.getRandomVector11( vectors[i] );
			vectors[i].mul( maxSr );
		}

		// Find closest and furthest point
		{
			for (int i=0; i<NTEST; i++)
			{
				hkVector4 point;
				rand.getRandomVector11( point );
				point.mul( maxSr );

				int minIndex = hkAlgorithm::findMinimumIndex( vectors.begin(), vectors.getSize(), DistanceToPointTest( point ) );
				int maxIndex = hkAlgorithm::findMaximumIndex( vectors.begin(), vectors.getSize(), DistanceToPointTest( point ) );

				HK_TEST( minIndex >= 0 && minIndex < ASIZE[n]);
				HK_TEST( maxIndex >= 0 && maxIndex < ASIZE[n]);

				const volatile hkReal minDistance = vectors[minIndex].distanceTo(point).getReal();
				const volatile hkReal maxDistance = vectors[maxIndex].distanceTo(point).getReal();

				for (int j = 0; j<ASIZE[n]; j++)
				{
					const volatile hkReal dist = vectors[j].distanceTo(point).getReal();
					HK_TEST(dist >= minDistance);
					HK_TEST(dist <= maxDistance);
				}
			}

			for (int i=0; i<NTEST; i++)
			{
				hkVector4 point;
				rand.getRandomVector11( point );
				point.mul( maxSr );

				int minIndex2, maxIndex2;
				hkReal minDistance2Tmp, maxDistance2Tmp;
				
				minIndex2 = hkAlgorithm::findMinimumIndexAndValue( vectors.begin(), vectors.getSize(), minDistance2Tmp, DistanceToPointTest(point) );
				maxIndex2 = hkAlgorithm::findMaximumIndexAndValue( vectors.begin(), vectors.getSize(), maxDistance2Tmp, DistanceToPointTest(point) );
				const volatile hkReal minDistance2 = minDistance2Tmp;
				const volatile hkReal maxDistance2 = maxDistance2Tmp;

				HK_TEST( minIndex2 >= 0 && minIndex2 < ASIZE[n]);
				HK_TEST( maxIndex2 >= 0 && maxIndex2 < ASIZE[n]);

				for (int j = 0; j<ASIZE[n]; j++)
				{
					const volatile hkReal dist2 = vectors[j].distanceToSquared(point).getReal();
					HK_TEST(dist2 >= minDistance2);
					HK_TEST(dist2 <= maxDistance2);
				}
			}
		}
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(findIndex_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
