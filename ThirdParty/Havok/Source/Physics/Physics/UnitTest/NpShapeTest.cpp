/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Common/Base/Types/Geometry/Geometry/hkGeometryUtil.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>


void testCalcSize()
{
	// Convex cube, no faces
	{
		hkVector4 halfExtents; halfExtents.set(0.5f, 0.5f, 0.5f);
		hknpConvexShape::BuildConfig config;
		config.m_buildFaces = false;
		hknpShape* shape = hknpConvexShape::createFromHalfExtents(halfExtents, 0, config);
		int calculatedSize = shape->calcSize();
		int expectedSize = sizeof(hknpConvexShape) + 8 * sizeof(hkcdVertex);
		expectedSize = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, expectedSize);
		HK_TEST(calculatedSize == expectedSize);
		shape->removeReference();
	}

	// Polytope cube
	{
		hkVector4 halfExtents; halfExtents.set(0.5f, 0.5f, 0.5f);
		hknpConvexShape::BuildConfig config;
		config.m_buildFaces = true;
		hknpShape* shape = hknpConvexShape::createFromHalfExtents(halfExtents, 0, config);
		int calculatedSize = shape->calcSize();
		int expectedSize = hknpConvexPolytopeShape::calcConvexPolytopeShapeSize(8,6,24,sizeof(hknpConvexPolytopeShape));
		HK_TEST(calculatedSize == expectedSize);
		shape->removeReference();
	}

	// Sphere
	{
		hknpShape* shape = hknpSphereShape::createSphereShape(hkVector4::getZero(), 1);
		int calculatedSize = shape->calcSize();
		int expectedSize = sizeof(hknpSphereShape) + 4 * sizeof(hkcdVertex);
		expectedSize = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, expectedSize);
		HK_TEST(calculatedSize == expectedSize);
		shape->removeReference();
	}

	// Capsule
	{
		hkVector4 b; b.set(1, 1, 1);
		hknpShape* shape = hknpCapsuleShape::createCapsuleShape(hkVector4::getZero(), b, 1);
		int calculatedSize = shape->calcSize();
		int expectedSize = hknpConvexPolytopeShape::calcConvexPolytopeShapeSize(8,6,24,sizeof(hknpCapsuleShape));
		HK_TEST(calculatedSize == expectedSize);
		shape->removeReference();
	}

	// Triangle
	{
		hknpShape* shape = hknpTriangleShape::createEmptyTriangleShape();
		int calculatedSize = shape->calcSize();
		int expectedSize = sizeof(hknpTriangleShape) + 4 * sizeof(hkcdVertex) +
						   HK_NEXT_MULTIPLE_OF(4, 1) * sizeof(hkVector4) + 4 * sizeof(hknpConvexPolytopeShape::Face) +
						   1 * 4 * sizeof(hknpConvexPolytopeShape::VertexIndex);
		expectedSize = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, expectedSize);
		HK_TEST(calculatedSize == expectedSize);
		shape->removeReference();
	}

	// CMS
	{
		hkGeometry geometry;
		hkGeometryUtil::createSphere(hkVector4::getZero(), 1, 12, &geometry);
		hknpDefaultCompressedMeshShapeCinfo meshInfo(&geometry);
		hknpShape* shape = new hknpCompressedMeshShape(meshInfo);
		int calculatedSize = shape->calcSize();
		int expectedSize = sizeof(hknpCompressedMeshShape);
		HK_TEST(calculatedSize == expectedSize);
		shape->removeReference();
	}
}

int NpShape_main()
{
	// Test disabled since shape size is not fully predictable (see HNP-342)
	testCalcSize();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(NpShape_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__);

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
