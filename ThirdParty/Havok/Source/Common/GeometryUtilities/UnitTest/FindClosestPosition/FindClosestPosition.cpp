/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/GeometryUtilities/hkGeometryUtilities.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindClosestPositionUtil/hkFindClosestPositionUtil.h>

static void checkClosestPoint( const hkFindClosestPositionUtil& positionUtil, hkVector4Parameter v)
{
	// Find the closest point
	int index = positionUtil.findClosest(v);
	int index2 = positionUtil.findClosestLinearly(v);

	if (index != index2)
	{
		// One found a point and the other didn't
		HK_TEST( !(index < 0 || index2 < 0) );
		if (index < 0 || index2 < 0)
		{
			return;
		}

		hkVector4 diff;
		diff.setSub(positionUtil.getPoint(index), v);
		hkReal dist = diff.lengthSquared<3>().getReal();

		diff.setSub(positionUtil.getPoint(index2), v);
		hkReal dist2 = diff.lengthSquared<3>().getReal();

		HK_TEST( dist == dist2 );
	}
}

static void checkPointIsNotInUtil( const hkFindClosestPositionUtil& positionUtil, hkVector4Parameter v)
{
	// Find the closest point
	int index = positionUtil.findClosest(v);
	int index2 = positionUtil.findClosestLinearly(v);

	if (index > 0)
	{
		hkVector4 diff;
		diff.setSub(positionUtil.getPoint(index), v);
		hkReal dist = diff.lengthSquared<3>().getReal();
		HK_TEST(dist > 0.0f) ;
	}

	if (index2 > 0)
	{
		hkVector4 diff;
		diff.setSub(positionUtil.getPoint(index2), v);
		hkReal dist2 = diff.lengthSquared<3>().getReal();
		HK_TEST(dist2 > 0.0f) ;
	}
}

static void findClosestPosition_addPoints()
{
	// taken from hkFindClosestPositionUtil::selfCheck();
	hkFindClosestPositionUtil positionUtil;

	hkArray<hkVector4> positions;
	const int numPoints = 128;
	hkPseudoRandomGenerator rand(100);

	hkSimdReal scale; scale.setFromFloat(50.f);
	for (int i = 0; i < numPoints; i++)
	{
		hkVector4 v;
		rand.getRandomVector11(v);

		v.mul(scale);
		positions.pushBack(v);
	}

	hkAabb aabb;
	aabb.m_max.setAll(scale);
	aabb.m_min.setAll(-scale);

	hkReal thresholds[] = { 1e-5f, 1e-2f, 1.0f, 15.0f, 100.0f };
	for (unsigned int i = 0; i < HK_COUNT_OF(thresholds); i++)
	{
		const hkReal threshold = thresholds[i];

		positionUtil.start(aabb, threshold);

		// Add all the positions
		positionUtil.addPoints(positions.begin(), positions.getSize());

		for (int j = 0; j < positions.getSize() + numPoints; j++)
		{
			hkVector4 v;

			if (j < positions.getSize())
			{
				v = positions[j];
			}
			else
			{
				rand.getRandomVector11(v);
				v.mul(scale);
			}

			checkClosestPoint(positionUtil, v);
		}
		positionUtil.end();
	}

}

static void findClosestPosition_removePoint()
{
	hkFindClosestPositionUtil positionUtil;

	hkArray<hkVector4> positions;
	const int numPoints = 128;
	hkPseudoRandomGenerator rand(100);

	hkSimdReal scale; scale.setFromFloat(50.f);
	for (int i = 0; i < numPoints; i++)
	{
		hkVector4 v;
		rand.getRandomVector11(v);

		v.mul(scale);
		positions.pushBack(v);
	}

	hkAabb aabb;
	aabb.m_max.setAll(scale);
	aabb.m_min.setAll(-scale);

	hkReal thresholds[] = { 1e-5f, 1e-2f, 1.0f, 15.0f, 100.0f };
	for (unsigned int i = 0; i < HK_COUNT_OF(thresholds); i++)
	{
		const hkReal threshold = thresholds[i];

		positionUtil.start(aabb, threshold);

		// Add all the positions
		positionUtil.addPoints(positions.begin(), positions.getSize());

		// Now remove half of them (the even ones)
		for (int j=0; j<positions.getSize(); j+=2)
		{
			positionUtil.removePoint( positions[j] );
		}

		HK_TEST( positionUtil.getPoints().getSize() == numPoints / 2 );

		for (int j = 0; j < 2 * positions.getSize(); j++)
		{
			hkVector4 v;
			hkBool isRemoved = false;
			
			if (j < positions.getSize())
			{
				isRemoved = ((j%2) == 0);
				v = positions[j];
			}
			else
			{
				rand.getRandomVector11(v);
				v.mul(scale);
			}

			if(isRemoved)
			{
				checkPointIsNotInUtil(positionUtil, v);
			}
			else
			{
				checkClosestPoint(positionUtil, v);
			}
		}
		positionUtil.end();
	}
}

int findClosestPosition_main()
{
	{
		findClosestPosition_addPoints();
		findClosestPosition_removePoint();
	}
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(findClosestPosition_main, "Fast", "Common/Test/UnitTest/GeometryUtilities/", __FILE__     );

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
