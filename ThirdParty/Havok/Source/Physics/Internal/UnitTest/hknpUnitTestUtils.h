/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_UNIT_TEST_UTILS_H
#define HKNP_UNIT_TEST_UTILS_H

#include <Common/Base/hkBase.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>


namespace hknpUnitTestUtils
{
	/// verify that the aabb of shape1 is contained in aabb of shape2
	 void compareAABBs(const hknpShape* shape1, const hknpShape* shape2, hkReal epsilon);

	 /// Check that two equivalent shapes returns equivalent support points
	 void compareSupportVertices( const hknpShape& shape1, const hknpShape& shape2, hkVector4Parameter support, hkReal epsilon );

	/// Check that the support points a and b are equally extremal in the direction of support
	 hkBool compareSupportPoints( hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter support, hkReal epsilon);

	/// Setup a transform by explicitly specified coordinates
	 void setupTransform( const hkReal axisx, const hkReal axisy, const hkReal axisz,  const hkReal radians, const hkReal tx, const hkReal ty, const hkReal tz, hkTransform& transform );

	/// Compare two hit points from two collectors
	 void compareTwoHitPoints( hknpClosestHitCollector& collector1, hknpClosestHitCollector& collector2 );

	/// Setup a ray with some default settings
	 void setupRay( hknpRayCastQuery &ray, hkVector4Parameter from, hkVector4Parameter to, const hkTransform& transform );

	 /// Cast a ray against a shape and report to the collector
	 void castRayAgainstShape( hkVector4Parameter from, hkVector4Parameter to, const hknpShape* shape, hknpCollisionQueryCollector* collector);

	/// Cast the same ray against two shapes with two collectors
	 void castRayAgainstShapes( hkVector4Parameter from, hkVector4Parameter to, const hknpShape* shape1, hknpCollisionQueryCollector* collector1, const hknpShape* shape2, hknpCollisionQueryCollector* collector2);

	/// Cast the same ray against two shapes with two collectors and compare hit points
	 void castRayAgainstShapesAndCompare( hkVector4Parameter from, hkVector4Parameter to, const hknpShape* shape1, const hknpShape* shape2 );

	/// Create an explicit dodecahedron convex shape under the given transform and scale
	 hknpConvexShape* createDodecahedron( const hkTransform& transform, hkVector4Parameter scale, hkReal radius = 0.0f);

	/// Create an explicit icosahedron convex shape under the given transform and scale
	 hknpConvexShape* createIcosahedron( const hkTransform& transform, hkVector4Parameter scale, hkReal radius = 0.0f);
}


#endif // HKNP_UNIT_TEST_UTILS_H

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
