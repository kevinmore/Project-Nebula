/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Internal/hkcdInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Geometry/Internal/Algorithms/ClosestPoint/hkcdClosestPointLineLine.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistanceSegmentSegment.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastCapsule.h>
#include <Geometry/Internal/Algorithms/ClosestPoint/hkcdClosestPointCapsuleCapsule.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Geometry/Internal/Types/hkcdRay.h>




static void capsuleCapsuleDistance()
{
	hkVector4 ls1;
	hkVector4 le1;
	hkVector4 ls2;
	hkVector4 le2;

	ls1.set(4.640474f, -9.645779f, 4.473800f);
	le1.set(9.545380f, 7.564442f, 8.718603f);
	ls2.set(3.907253f, 5.442764f, 2.809450f);
	le2.set(1.063313f, -5.920457f, -3.643977f);

	hkVector4 point; point.setZero();
	hkVector4 normal; normal.setZero();
	hkSimdReal radiusA = hkSimdReal::getConstant<HK_QUADREAL_1>();
	hkSimdReal radiusB = hkSimdReal::getConstant<HK_QUADREAL_2>();
	hkSimdReal tolerance; tolerance.setFromFloat(0.1f);

	hkcdClosestPointCapsuleCapsule(ls1, le1, radiusA, ls2, le2, radiusB, tolerance, &point, &normal);

	HK_TEST( point.getComponent<0>().getReal() != -1.0f );
}

static void lineLineDistance()
{
	{
		hkVector4 ls1;
		hkVector4 le1;
		hkVector4 ls2;
		hkVector4 le2;


		ls1.set(3.126088f, -2.528251f, 8.175564f);
		le1.set(8.346388f, -5.436057f, -8.012310f);
		ls2.set(2.267486f, -8.824311f, -1.423940f);
		le2.set(6.315895f, -9.785579f, 3.703618f);

		hkVector4 p1;
		hkVector4 p2;
		hkSimdReal distSqr;

		hkcdClosestPointSegmentSegment(ls1, le1, ls2, le2, p1, p2, distSqr);

		HK_TEST( distSqr.getReal() > 0.0f );
	}

	// Skew lines, overlap in 2D
	{
		hkVector4 A1; A1.set(-1, 0, 1);
		hkVector4 A2; A2.set( 1, 0, 1);

		hkVector4 B1; B1.set(0, -1, -1);
		hkVector4 B2; B2.set(0,  1, -1);

		hkVector4 dA; dA.setSub(A2, A1);
		hkVector4 dB; dB.setSub(B2, B1);

		hkVector4 closestPointAout, closestAminusClosestBout;
		hkSimdReal distanceSquaredOut;
		
		hkcdClosestPointSegmentSegment(A1, dA, B1, dB, closestPointAout, closestAminusClosestBout, distanceSquaredOut);

		hkSimdReal t, u;
		hkcdClosestPointSegmentSegment(A1, dA, B1, dB, t, u);

		hkVector4 closestPointA; closestPointA.set(0,0,1);
		hkVector4 closestAminusClosestB; closestAminusClosestB.set(0, 0, 2);

		hkSimdReal eps; eps.setFromFloat(1e-5f);
		
		HK_TEST( closestPointA.allEqual<3>(closestPointAout, eps) );
		HK_TEST( closestAminusClosestB.allEqual<3>(closestAminusClosestBout, eps) );
		HK_TEST( t.approxEqual(hkSimdReal_Half, eps) );
		HK_TEST( u.approxEqual(hkSimdReal_Half, eps) );
		HK_TEST( distanceSquaredOut.approxEqual( hkSimdReal_4, eps ) );

		hkSimdReal distanceSquaredAlternate = hkcdDistanceSegmentSegment( A1, dA, B1, dB );
		HK_TEST( distanceSquaredOut.approxEqual( distanceSquaredAlternate, eps ) );

	}
}

static void testLineLineDegenerate( hkVector4Parameter point, hkVector4Parameter lineStart, hkVector4Parameter lineEnd )
{
	hkVector4 dPoint = hkVector4::getConstant<HK_QUADREAL_0>();
	hkVector4 dLine; dLine.setSub(lineEnd, lineStart);
	hkSimdReal eps; eps.setFromFloat(1e-5f);

	hkVector4 projection;
	hkSimdReal distSqrPointLine = hkcdPointSegmentDistanceSquared(point, lineStart, lineEnd, &projection);

	// Test degenerate first segment
	{
		hkVector4 closestPointAout, closestAminusClosestBout;
		hkSimdReal distSqrLineLine, t, u;

		hkcdClosestPointSegmentSegment( point, dPoint, lineStart, dLine, t, u);
		hkcdClosestPointSegmentSegment( point, dPoint, lineStart, dLine, closestPointAout, closestAminusClosestBout, distSqrLineLine);

		// Make sure no NANs
		HK_TEST( t.isOk() && u.isOk() && distSqrLineLine.isOk() && closestPointAout.isOk<3>() && closestAminusClosestBout.isOk<3>() );

		HK_TEST( t >= hkSimdReal_0 && t <= hkSimdReal_1 );
		HK_TEST( u >= hkSimdReal_0 && u <= hkSimdReal_1 );
		HK_TEST( distSqrLineLine.approxEqual(distSqrPointLine, eps) );

		// Check u value (t doesn't matter);
		hkVector4 interp; interp.setInterpolate(lineStart, lineEnd, u);
		HK_TEST( interp.allEqual<3>(projection, eps) );

		// Closest point on the degenerate segment is just the point
		HK_TEST( closestPointAout.allEqual<3>(point, eps) );

		// A - (closestPointA - closestPointB) = closestPointB
		hkVector4 projectionLineLine; projectionLineLine.setSub(point, closestAminusClosestBout);
		HK_TEST( projectionLineLine.allEqual<3>(projection, eps) );
	}

	// Test degenerate second segment
	{
		hkVector4 closestPointAout, closestAminusClosestBout;
		hkSimdReal distSqrLineLine, t, u;

		hkcdClosestPointSegmentSegment( lineStart, dLine, point, dPoint, t, u);
		hkcdClosestPointSegmentSegment( lineStart, dLine, point, dPoint, closestPointAout, closestAminusClosestBout, distSqrLineLine);

		// Make sure no NANs
		HK_TEST( t.isOk() && u.isOk() && distSqrLineLine.isOk() && closestPointAout.isOk<3>() && closestAminusClosestBout.isOk<3>() );

		HK_TEST( t >= hkSimdReal_0 && t <= hkSimdReal_1 );
		HK_TEST( u >= hkSimdReal_0 && u <= hkSimdReal_1 );
		HK_TEST( distSqrLineLine.approxEqual(distSqrPointLine, eps) );

		// Check t value (u doesn't matter);
		hkVector4 interp; interp.setInterpolate(lineStart, lineEnd, t);
		HK_TEST( interp.allEqual<3>(projection, eps) );

		// Projection onto the segment is the same as closest point on A
		HK_TEST( closestPointAout.allEqual<3>(projection, eps) );

		// (projection - point) == (closestPointA - closestPointB)
		hkVector4 projMinusPoint; projMinusPoint.setSub(projection, point);
		HK_TEST( projMinusPoint.allEqual<3>(closestAminusClosestBout, eps) );
	}

}

static void lineLineDegenerate()
{
	{
		// These are taked from the point-segment tests
		hkVector4 point, lineStart, lineEnd;

		point.set(-5.278641f, -4.428662f, 6.390675f);
		lineStart.set(-2.318453f, 2.436151f, -3.125482f);
		lineEnd.set(0.155562f, 1.635951f, 5.934510f);
		testLineLineDegenerate(point, lineStart, lineEnd);
	}

	{
		// Test when the closest point is an endpoint
		hkVector4 point; point.setAll(2.0f);
		hkVector4 lineStart; lineStart.setAll(1.0f);
		hkVector4 lineEnd; lineEnd.setAll(-1.0f);
		testLineLineDegenerate(point, lineStart, lineEnd);

		point.setAll(-2.0f);
		testLineLineDegenerate(point, lineStart, lineEnd);
	}

	// Pick some random points
	hkPseudoRandomGenerator rand(1234);
	for (int i=0; i<10; i++)
	{

		hkVector4 point, lineStart, lineEnd;
		rand.getRandomVector11(point);
		rand.getRandomVector11(lineStart);
		rand.getRandomVector11(lineEnd);

		testLineLineDegenerate(point, lineStart, lineEnd);
	}
	
	
}

int LineLine_main()
{
	lineLineDistance();
	lineLineDegenerate();

	capsuleCapsuleDistance();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(LineLine_main, "Fast", "Geometry/Test/UnitTest/Internal/", __FILE__ );

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
