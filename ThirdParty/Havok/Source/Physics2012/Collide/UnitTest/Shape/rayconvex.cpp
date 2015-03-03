/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastOutput.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

#define NTEST	10000

//#define MAX_BOX	900.f
//#define MED_BOX	850.f
//#define MIN_BOX 800.f

#define MAX_BOX	500.f
#define MED_BOX	200.f
#define MIN_BOX 100.f


// Testing the ray(segment) cast on convex objects against
// rayTriangle, for random convex objects (convex hull of a random point cloud)

// Current Test Issues:
// Precondition: make sure the 'from' starts 'out' of the primitive, else the result is undefined !
// What is the tested range ?
// Convex hull (qhull) generation makes the testing relatively 'slow'
// memleaks

int rayconvex_main()
{

	int numvert=4;

	for (int n = 0; n < NTEST; n++ )
	{
		hkVector4 minbox;
		minbox.set(
			hkUnitTest::randRange(MIN_BOX,MED_BOX),
			hkUnitTest::randRange(MIN_BOX,MED_BOX),
			hkUnitTest::randRange(MIN_BOX,MED_BOX));

		hkVector4 maxbox;
		maxbox.set(
			hkUnitTest::randRange(MED_BOX,MAX_BOX),
			hkUnitTest::randRange(MED_BOX,MAX_BOX),
			hkUnitTest::randRange(MED_BOX,MAX_BOX));
#define FIXED_FROM_TO
#ifdef FIXED_FROM_TO
		hkpShapeRayCastInput rc_input;
		rc_input.m_from.set(1000.f,1000.f,1000.f);
		rc_input.m_to.set(1.f,2.f,3.f);
#else
		hkVector4 from(
			hkUnitTest::randRange(-MAX_BOX,MAX_BOX),
			hkUnitTest::randRange(-MAX_BOX,MAX_BOX),
			hkUnitTest::randRange(-MAX_BOX,MAX_BOX));
		hkVector4 to(
			hkUnitTest::randRange(-MAX_BOX,MAX_BOX),
			hkUnitTest::randRange(-MAX_BOX,MAX_BOX),
			hkUnitTest::randRange(-MAX_BOX,MAX_BOX));

#endif	//FIXED_FROM




		int i;
		hkpShapeRayCastOutput rayResults;

		hkArray<hkVector4> verts(numvert);
		for(i = 0; i < numvert; ++i)
		{
			for(int j = 0; j < 3; ++j)
			{
				verts[i](j) = hkUnitTest::randRange( minbox(j), maxbox(j) );
			}
		}

		hkpConvexVerticesShape* shape = new hkpConvexVerticesShape(verts);
		hkGeometry				geom;
		hkGeometryUtility::createConvexGeometry(verts,geom);
		
		int hit  = shape->castRay( rc_input, rayResults );
		int hitalternative = 0;

		hkpShapeRayCastOutput triangleResults;

		for (i=0;i<geom.m_triangles.getSize();i++)
		{
			hkGeometry::Triangle ind = geom.m_triangles[i];
			hkpTriangleShape triangle(
				geom.m_vertices[ind.m_a],
				geom.m_vertices[ind.m_b],
				geom.m_vertices[ind.m_c]);

			int trianglehit = triangle.castRay( rc_input, triangleResults );
			if (trianglehit)
			{
				//only replace resulting point if closest
				hitalternative=1;
			}
		}

		hkBool testguard = (hit == hitalternative);
		HK_TEST2(testguard ,"boolean test iteration " << n);
		if (testguard && hit)
		{
			HK_TEST2(  hkMath::fabs(rayResults.m_hitFraction - triangleResults.m_hitFraction) < 1e-4f, "mindist test iteration " << n);
		}

	}
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

//HK_TEST_REGISTER(rayconvex_main, "UNKNOWN", "Physics2012/Test/UnitTest/Collide/", __FILE__  );

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
