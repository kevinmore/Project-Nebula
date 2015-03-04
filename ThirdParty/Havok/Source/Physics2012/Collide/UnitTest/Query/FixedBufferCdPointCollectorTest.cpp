/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpRootCdPoint.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpFixedBufferCdPointCollector.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#define MAX_TEST_CAPACITY	25
#define NUM_RANDOM_TESTS	25
#define SEED				1234

void testHitArray( hkLocalArray<hkReal>& hitArray )
{
	//
	//	Setup CdPoint
	//
	hkTransform* t = HK_NULL;
	hkpShape* shape = HK_NULL;
	hkpCdBody bodyA(shape, t);
	hkpCdBody bodyB(shape, t);
	hkpCdPoint cdPoint(bodyA,bodyB);
	hkVector4 pos; pos.set(0.0f,0.0f,0.0f);
	hkVector4 normal; normal.set(1.0f,0.0f,0.0f);
	cdPoint.setContact(pos, normal, 0.0f );

	//
	// Test with various collector capacities
	//
	for(int collectorCapacity = 1; collectorCapacity < MAX_TEST_CAPACITY ; ++collectorCapacity)
	{		
		//
		// Allocate memory for the collector
		//
		hkpRootCdPoint*	collectorBuffer;
		hkpFixedBufferCdPointCollector* collector;
		{
			collectorBuffer = hkAllocateStack<hkpRootCdPoint>( collectorCapacity, "" );
			collector = new hkpFixedBufferCdPointCollector(collectorBuffer, collectorCapacity);
		}

		const int arraySize = hitArray.getSize();

		//
		// Add the hits to the collector
		//
		for(int j = 0; j < arraySize ; ++j )
		{
			cdPoint.setContactDistance(hitArray[j]);
			hkpFixedBufferCdPointCollector::addCdPointImplementation(cdPoint, collector);
		}

		//
		// Sort the arrays
		//
		hkSort(hitArray.begin(), arraySize);
		hkSort(collector->m_pointsArrayBase.val(), collector->m_numPoints);

		//
		// Do the tests
		//
		{	
			HK_TEST2( collector->m_numPoints <= arraySize, "Collected more points than given!");
			if( arraySize <= collectorCapacity )
			{
				HK_TEST2( collector->m_numPoints == arraySize, "Collected wrong number of points!");
			}

			if( arraySize >= collectorCapacity )
			{
				HK_TEST2( collector->m_numPoints == collectorCapacity, "Buffer should be filled.");
			}

			for(int i = 0 ; i < collector->m_numPoints ; ++i)
			{
				HK_TEST2( (collector->m_pointsArrayBase + i)->m_contact.getDistance() == hitArray[i], "Wrong points collected!");
			}

			const hkReal earlyOutDistance = collector->getEarlyOutDistance();
			const hkReal furthestHit = (collector->m_pointsArrayBase + collector->m_numPoints - 1)->m_contact.getDistance();

			// The early out distance is only set AFTER the capacity has been exceeded
			if( collectorCapacity < arraySize )
			{
				HK_TEST2( furthestHit == earlyOutDistance, "Early out distance is wrong");
			}	
		}
		

		delete collector;
		hkDeallocateStack(collectorBuffer, collectorCapacity);

	}
	
}

int fixedBufferCdPointCollector_test()
{	
	hkDisableError disable0xaf531e14( 0xaf531e14 );

	//
	// Test trivial cases
	//
	{
		hkLocalArray<hkReal> test1(1);test1.setSizeUnchecked(1);	test1[0]  = 5.0f;										testHitArray( test1 );	
		hkLocalArray<hkReal> test2(2);test2.setSizeUnchecked(2);	test2[0]  = 5.0f;	test2[1]  = 3.0f;					testHitArray( test2 );	
		hkLocalArray<hkReal> test3(2);test3.setSizeUnchecked(2);	test3[0]  = 3.0f;	test3[1]  = 5.0f;					testHitArray( test3 );	
		hkLocalArray<hkReal> test4(3);test4.setSizeUnchecked(3);	test4[0]  = 3.0f;	test4[1]  = 5.0f; test4[2]  = 4.0f;	testHitArray( test4 );	
		hkLocalArray<hkReal> test5(3);test5.setSizeUnchecked(3);	test5[0]  = 5.0f;	test5[1]  = 3.0f; test5[2]  = 4.0f;	testHitArray( test5 );	
		hkLocalArray<hkReal> test6(3);test6.setSizeUnchecked(3);	test6[0]  = 3.0f;	test6[1]  = 5.0f; test6[2]  = 6.0f;	testHitArray( test6 );	
		hkLocalArray<hkReal> test7(3);test7.setSizeUnchecked(3);	test7[0]  = 5.0f;	test7[1]  = 5.0f; test7[2]  = 6.0f;	testHitArray( test7 );	
		hkLocalArray<hkReal> test8(3);test8.setSizeUnchecked(3);	test8[0]  = 5.0f;	test8[1]  = 5.0f; test8[2]  = 2.0f;	testHitArray( test8 );	
		hkLocalArray<hkReal> test9(3);test9.setSizeUnchecked(3);	test9[0]  = 2.0f;	test9[1]  = 5.0f; test9[2]  = 5.0f;	testHitArray( test9 );	
		hkLocalArray<hkReal> test10(3);test10.setSizeUnchecked(3);	test10[0] = 5.0f;	test10[1] = 5.0f; test10[2] = 7.0f;	testHitArray( test10 );	
		hkLocalArray<hkReal> test11(3);test11.setSizeUnchecked(3);	test11[0] = 5.0f;	test11[1] = 5.0f; test11[2] = 2.0f;	testHitArray( test11 );	
		hkLocalArray<hkReal> test12(3);test12.setSizeUnchecked(3);	test12[0] = 5.0f;	test12[1] = 5.0f; test12[2] = 5.0f;	testHitArray( test12 );	
	}
	
	
	//
	// Test random arrays
	//
	{
		hkLocalArray< hkReal > randomTest(NUM_RANDOM_TESTS);
		randomTest.setSizeUnchecked(NUM_RANDOM_TESTS);
		hkPseudoRandomGenerator rand(SEED);
		for(int i = 0 ; i < randomTest.getSize(); ++i)
		{
			//
			// Fill array with random numbers
			//
			for(int j = 0; j < NUM_RANDOM_TESTS ; ++j)
			{
				randomTest[j] = rand.getRandReal11();
			}

			//
			// Do the test
			//
			testHitArray( randomTest );
		}
	}
	
	

	return 0;
}


//
// test registration
//
#if defined( HK_COMPILER_MWERKS )
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( fixedBufferCdPointCollector_test , "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__     );

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
