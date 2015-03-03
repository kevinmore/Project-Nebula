/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>


static void testGetAabb()
{
	const int numTests = 1000;
	const hkReal halfLength = 10;
	const int numSegments = 128;
	
	hkPseudoRandomGenerator rnd(1);
	hkpCylinderShape::setNumberOfVirtualSideSegments(numSegments);
	hkVector4 min; min.set(-halfLength, -halfLength, -halfLength);
	hkVector4 max; max.set(halfLength, halfLength, halfLength);
	hkVector4 totalDifferenceMin; totalDifferenceMin.setZero();
	hkVector4 totalDifferenceMax; totalDifferenceMax.setZero();
	hkVector4 totalHalfExtents; totalHalfExtents.setZero();

	for (int i = 0; i < numTests; ++i)
	{
		// Create cylinder with random vertices and radius
		hkVector4 vertexA;
		hkVector4 vertexB;
		rnd.getRandomVectorRange(min, max, vertexA);
		rnd.getRandomVectorRange(min, max, vertexB);
		hkReal radius = halfLength * rnd.getRandReal01();
		hkpCylinderShape cylinder(vertexA, vertexB, radius);

		// Obtain aabb
		hkAabb aabb;
		cylinder.getAabb(hkTransform::getIdentity(), 0, aabb);
		hkVector4 halfExtents;
		aabb.getHalfExtents(halfExtents);
		totalHalfExtents.add(halfExtents);

		// Test if the aabb contains the supporting vertices in all axis directions
		hkAabb aabbFromSupport; aabbFromSupport.setEmpty();
		for (int j = 0; j < 3; ++j)
		{
			hkcdVertex supportingVertex;
			hkVector4 direction; direction.setZero();
			direction(j) = 1;
			cylinder.getSupportingVertex(direction, supportingVertex);
			HK_TEST(aabb.containsPoint(supportingVertex));
			aabbFromSupport.includePoint(supportingVertex);
			direction.setNeg<3>(direction);
			cylinder.getSupportingVertex(direction, supportingVertex);
			HK_TEST(aabb.containsPoint(supportingVertex));
			aabbFromSupport.includePoint(supportingVertex);
		}		

		// Keep track of the difference
		hkVector4 difference; difference.setSub(aabb.m_min, aabbFromSupport.m_min);
		difference.setAbs(difference);
		totalDifferenceMin.add(difference);
		difference.setSub(aabb.m_max, aabbFromSupport.m_max);
		difference.setAbs(difference);
		totalDifferenceMax.add(difference);
	}
	
	hkVector4 numTestsVec; numTestsVec.setAll(hkReal(numTests));
	totalDifferenceMin.div(numTestsVec);
	totalDifferenceMax.div(numTestsVec);
	hkStringBuf text;
	text.printf("Average(aabb.m_max - supportingVertex) : (%f, %f, %f)", totalDifferenceMax(0), totalDifferenceMax(1), totalDifferenceMax(2));
	HK_REPORT(text);
	text.printf("Average(aabb.m_min - supportingVertex) : (%f, %f, %f)", totalDifferenceMin(0), totalDifferenceMin(1), totalDifferenceMin(2));
	HK_REPORT(text);
	totalHalfExtents.div(numTestsVec);
	text.printf("Average aabb volume : %f", 8 * totalHalfExtents(0) * totalHalfExtents(1) * totalHalfExtents(2));
	HK_REPORT(text);
	
}

int CylinderShapeUnitTest()
{
	testGetAabb();

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( CylinderShapeUnitTest, "Fast", "Physics2012/Test/UnitTest/Collide/", __FILE__ );

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
