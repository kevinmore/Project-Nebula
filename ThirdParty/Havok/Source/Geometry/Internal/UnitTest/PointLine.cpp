/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/hkcdInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>

#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointLine.h>

#define MX_LEN 4
#define ITERATIONS 100000

static void pointLineWithProjection()
{
	hkVector4 point;
	hkVector4 lineStart;
	hkVector4 lineEnd;
	hkVector4 projection;

	point.set(-5.278641f, -4.428662f, 6.390675f);
	lineStart.set(-2.318453f, 2.436151f, -3.125482f);
	lineEnd.set(0.155562f, 1.635951f, 5.934510f);
	
	hkSimdReal distSqr = hkcdPointSegmentDistanceSquared(point, lineStart, lineEnd, &projection);
	hkSimdReal distSqrNoProj = hkcdPointSegmentDistanceSquared(point, lineStart, lineEnd); 
	
	HK_TEST( hkMath::equal(distSqr.getReal(), 66.294434f) );
	// Test that we get the same distance with or without projection
	HK_TEST( hkMath::equal(distSqr.getReal(), distSqrNoProj.getReal() ) );

	hkVector4 result; result.set(0.031407f, 1.676108f, 5.479846f, 0.949816f);
	hkSimdReal eps; eps.setFromFloat(1e-5f);
	HK_TEST( projection.allEqual<4>(result, eps) );
}

// Test zero-length segment
static void pointLineDegenerate()
{
	hkVector4 point;
	hkVector4 lineStart;
	hkVector4 lineEnd;
	hkVector4 projection;

	point.set(2.0f, 0.0f, 0.0f);
	lineStart.set(-1.0f, 0.0f, 0.0f);
	lineEnd.set(-1.0f, 0.0f, 0.0f);

	hkSimdReal distSqr = hkcdPointSegmentDistanceSquared(point, lineStart, lineEnd, &projection);

	HK_TEST( hkMath::equal(distSqr.getReal(), 9.0f) );

	hkVector4 result = lineStart;
	hkSimdReal eps; eps.setFromFloat(1e-5f);
	HK_TEST( projection.allEqual<3>(result, eps) );
}

static void pointLineWithProjectionBundle()
{
	hkVector4 point;
	hkVector4 lineStart;
	hkVector4 lineEnd;

	point.set(-5.278641f, -4.428662f, 6.390675f);
	lineStart.set(-2.318453f, 2.436151f, -3.125482f);
	lineEnd.set(0.155562f, 1.635951f, 5.934510f);

	hkFourTransposedPoints points;
	points.setAll(point);

	hkFourTransposedPoints projection;
	hkVector4 distSqr = hkcdPointSegmentDistanceSquared(points, lineStart, lineEnd, &projection);

	HK_TEST( hkMath::equal(distSqr(0), 66.294434f) );

	
	//hkVector4 result; result.set(0.031407f, 1.676108f, 5.479846f, 0.949816f);
	//hkSimdReal eps(1e-5f);
	//HK_TEST( projection.allEqual<4>(result, eps) );
}

static void pointSegmentWithoutProjectionMx()
{
	hkReal p[] = {
		1.0f, 2.0f, 3.0f, 0.0f, // repeat MX_LEN times
		1.0f, 2.0f, 3.0f, 0.0f,
		1.0f, 2.0f, 3.0f, 0.0f,
		1.0f, 2.0f, 3.0f, 0.0f
	};
	hkReal ls[] = {
		0.0f, 0.0f, -10.0f, 0.0f, // repeat MX_LEN times
		0.0f, 0.0f, -10.0f, 0.0f,
		0.0f, 0.0f, -10.0f, 0.0f,
		0.0f, 0.0f, -10.0f, 0.0f,
	};
	hkReal le[] = {
		0.0f, 0.0f, 10.0f, 0.0f, // repeat MX_LEN times
		0.0f, 0.0f, 10.0f, 0.0f,
		0.0f, 0.0f, 10.0f, 0.0f,
		0.0f, 0.0f, 10.0f, 0.0f,
	};
	hkMxVector<MX_LEN> point; point.loadNotAligned(p);
	hkMxVector<MX_LEN> lineStart; lineStart.loadNotAligned(ls);
	hkMxVector<MX_LEN> lineEnd; lineEnd.loadNotAligned(le);
	
	hkStopwatch sw;
	sw.start();
	hkReal dist[MX_LEN];
	hkString::memSet(dist, 0, sizeof(hkReal)*MX_LEN);
	for (int i = 0; i < ITERATIONS; ++i )
	{
		hkcdPointSegmentDistanceSquared<MX_LEN>(point, lineStart, lineEnd).storeNotAligned(dist);
	}
	sw.stop();
	//HK_REPORT("MxVector code: " << sw.getElapsedTicks());
	
	HK_TEST(dist[0] == 5.0f);
	HK_TEST(dist[1] == 5.0f);
	// HK_TEST(dist[2] == 5.0f);
	// HK_TEST(dist[3] == 5.0f);
}

void pointCapsuleNormal()
{
	hkVector4 p; p.set(1, 2, 3);
	hkVector4 ls; ls.set(0, 0, -10);
	hkVector4 le; le.set(0, 0, 10);
	hkVector4 proj;
	hkVector4 normal;

	hkSimdReal c; c.setFromFloat(1.2f);
	hkSimdReal dist = hkcdPointCapsuleClosestPoint(p, ls, le, c, &proj, &normal);

// 	HK_REPORT("distance = " << dist);
// 	HK_REPORT("projection = " << proj.getComponent<0>());
	HK_TEST(c != dist);
}

int PointLine_main()
{
	pointLineWithProjection();

	pointLineDegenerate();

	pointLineWithProjectionBundle();
	
	pointSegmentWithoutProjectionMx();

	pointCapsuleNormal();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(PointLine_main, "Fast", "Geometry/Test/UnitTest/Internal/", __FILE__ );

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
