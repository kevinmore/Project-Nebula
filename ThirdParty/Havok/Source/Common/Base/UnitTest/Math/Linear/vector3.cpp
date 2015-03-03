/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkPackedVector3.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Common/Base/UnitTest/Math/Linear/mathtestutils.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>


template <int N>
static void checkEqual23Bit(hkVector4Parameter x, hkVector4Parameter y)
{
	for (int i = 0; i < N; i++)
	{
		HK_TEST( hkIsEqual23BitAccurate(x(i), y(i)));
	}
}

template <int N>
static void checkEqual12Bit(hkVector4Parameter x, hkVector4Parameter y)
{
	for (int i = 0; i < N; i++)
	{
		HK_TEST( hkIsEqual12BitAccurate(x(i), y(i)));
	}
}


static void vector_assign_basic()
{
	// set
	{
		hkVector4 x; x.set(5,2,1,3);
		HK_TEST(x(0)==5);
		HK_TEST(x(1)==2);
		HK_TEST(x(2)==1);
		HK_TEST(x(3)==3);
	}

	{
		hkVector4 x;
		x.set(hkSimdReal::getConstant<HK_QUADREAL_5>(), hkSimdReal::getConstant<HK_QUADREAL_2>(), hkSimdReal::getConstant<HK_QUADREAL_1>(), hkSimdReal::getConstant<HK_QUADREAL_3>());
		HK_TEST(x(0)==5);
		HK_TEST(x(1)==2);
		HK_TEST(x(2)==1);
		HK_TEST(x(3)==3);
	}

	// setAll
	{
		hkVector4 x;
		x.setAll(93);
		HK_TEST(x(0)==93);
		HK_TEST(x(1)==93);
		HK_TEST(x(2)==93);
		HK_TEST(x(3)==93);
	}

	{
		hkVector4 x;
		x.setAll(94);
		HK_TEST(x(0)==94);
		HK_TEST(x(1)==94);
		HK_TEST(x(2)==94);
	}

	{
		hkVector4 x;
		hkSimdReal s; s.setFromFloat(94.0f);
		x.setAll(s);
		HK_TEST(x(0)==94);
		HK_TEST(x(1)==94);
		HK_TEST(x(2)==94);
		HK_TEST(x(3)==94);
	}

	{
		hkVector4 x;
		x.setZero();
		HK_TEST(x(0)==0);
		HK_TEST(x(1)==0);
		HK_TEST(x(2)==0);
		HK_TEST(x(3)==0);
	}

	{
		hkVector4 x;
		x.setZero();
		HK_TEST(x(0)==0);
		HK_TEST(x(1)==0);
		HK_TEST(x(2)==0);
		HK_TEST(x(3)==0);
	}

	// operator =
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;

		HK_TEST(y(0)==5);
		HK_TEST(y(1)==2);
		HK_TEST(y(2)==1);
		HK_TEST(y(3)==3);
	}


	// Test component sets
	{
		hkVector4 v;
		hkVector4 val;
		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		hkSimdReal five; five.setFromFloat(5.0f);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		v.setComponent<0>(five);	val(0) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		v.setComponent<1>(five);	val(1) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		v.setComponent<2>(five);	val(2) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		v.setComponent<3>(five);	val(3) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));
	}
	{
		hkVector4 v;
		hkVector4 val;
		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		hkSimdReal five; five.setFromFloat(5.0f);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		v.setComponent(0, five);	val(0) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		v.setComponent(1, five);	val(1) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		v.setComponent(2, five);	val(2) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		val = v;
		v.setComponent(3, five);	val(3) = 5.0f;
		HK_TEST(v.allEqual<4>(val,eps));

		HK_TEST_ASSERT(0x6d0c31d7, v.setComponent(-1, five));
		HK_TEST_ASSERT(0x6d0c31d7, v.setComponent(4, five));
		HK_TEST_ASSERT(0x6d0c31d7, v.setComponent(5, five));
	}

	hkPseudoRandomGenerator random(10);
	for (int i =0; i < 100; i++)
	{
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 v1; random.getRandomVector11( v1 );
		hkVector4 v2; random.getRandomVector11( v2 );
		hkSimdReal eps; eps.setFromFloat(1e-3f);

		// setXYZW
		{
			hkVector4 ref; ref.set( v0(0), v0(1),v0(2), v1(3));
			hkVector4 re2; re2.setXYZ_W( v0, v1 );
			HK_TEST( ref.allEqual<4>(re2,eps));
		}

		// setXYZW simdreal
		{
			hkVector4 ref; ref.set( v0(0), v0(1),v0(2), v1(1));
			hkVector4 re2; re2.setXYZ_W( v0, v1.getComponent<1>() );
			HK_TEST( ref.allEqual<4>(re2,eps));
		}
	}


	// Testing functionalities of setXYZW(),setW(),setXYZ0(),setXYZ()
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		hkVector4 y;
		hkVector4 z;
		const int NUM_TIMES = 100;

		for(int i = 0; i < NUM_TIMES; i++)
		{
			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
			y.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());

			z.setXYZ_W(x,y);
			HK_TEST( hkMath::equal(z(0),x(0)) );
			HK_TEST( hkMath::equal(z(1),x(1)) );
			HK_TEST( hkMath::equal(z(2),x(2)) );
			HK_TEST( hkMath::equal(z(3),y(3)) );

			z.setW(x);
			HK_TEST( z(3) == x(3) );

			z.setXYZ_0(x);
			HK_TEST( hkMath::equal(z(0),x(0)) );
			HK_TEST( hkMath::equal(z(1),x(1)) );
			HK_TEST( hkMath::equal(z(2),x(2)) );
			HK_TEST( hkMath::equal(z(3),0.0f) );

			z.setAll(1.0f);
			z.setXYZ(y);
			HK_TEST( hkMath::equal(z(0),y(0)) );
			HK_TEST( hkMath::equal(z(1),y(1)) );
			HK_TEST( hkMath::equal(z(2),y(2)) );
			HK_TEST( hkMath::equal(z(3), 1.0f) );
		}

		// Testing functionality of setXYZ()
		{
			hkVector4 vec;
			for(hkReal i = 0; i < NUM_TIMES; i++)
			{
				vec.setXYZ(i);
				HK_TEST(vec(0) == i && vec(1) == i && vec(2) == i );
			}

			for(hkReal i = 0; i < NUM_TIMES; i++)
			{
				hkSimdReal ii; ii.setFromFloat(i);
				vec.setXYZ(ii);
				HK_TEST(vec(0) == i && vec(1) == i && vec(2) == i );
			}
		}

	}
}

#if defined(HK_PLATFORM_WINRT) && defined(HK_ARCH_ARM)
// The optimizer goes a bit nuts (~300MiB mem and takes a long long time to complete, on the rows of comparision tests in this file
// TODO: Repro in smaller file and report
#pragma optimize("", off)
#endif

static void vector_vector4comparisons()
{
	// Verify vector4 comparison class
	{
		// get mask
		{
			HK_TEST(hkVector4Comparison::getMaskForComponent(0) == hkVector4ComparisonMask::MASK_X);
			HK_TEST(hkVector4Comparison::getMaskForComponent(1) == hkVector4ComparisonMask::MASK_Y);
			HK_TEST(hkVector4Comparison::getMaskForComponent(2) == hkVector4ComparisonMask::MASK_Z);
			HK_TEST(hkVector4Comparison::getMaskForComponent(3) == hkVector4ComparisonMask::MASK_W);
		}

		// set and prep for other tests.
		hkLocalArray<hkVector4Comparison> masks(16);
		masks.setSize(0);
		hkVector4Comparison MASK_NONE; MASK_NONE.set(hkVector4ComparisonMask::MASK_NONE); HK_TEST(MASK_NONE.getMask() == hkVector4ComparisonMask::MASK_NONE); masks.pushBack(MASK_NONE);
		hkVector4Comparison MASK_W; MASK_W.set(hkVector4ComparisonMask::MASK_W); HK_TEST(MASK_W.getMask() == hkVector4ComparisonMask::MASK_W); masks.pushBack(MASK_W);
		hkVector4Comparison MASK_Z; MASK_Z.set(hkVector4ComparisonMask::MASK_Z); HK_TEST(MASK_Z.getMask() == hkVector4ComparisonMask::MASK_Z); masks.pushBack(MASK_Z);
		hkVector4Comparison MASK_ZW; MASK_ZW.set(hkVector4ComparisonMask::MASK_ZW); HK_TEST(MASK_ZW.getMask() == hkVector4ComparisonMask::MASK_ZW); masks.pushBack(MASK_ZW);
		hkVector4Comparison MASK_Y; MASK_Y.set(hkVector4ComparisonMask::MASK_Y); HK_TEST(MASK_Y.getMask() == hkVector4ComparisonMask::MASK_Y); masks.pushBack(MASK_Y);
		hkVector4Comparison MASK_YW; MASK_YW.set(hkVector4ComparisonMask::MASK_YW); HK_TEST(MASK_YW.getMask() == hkVector4ComparisonMask::MASK_YW); masks.pushBack(MASK_YW);
		hkVector4Comparison MASK_YZ; MASK_YZ.set(hkVector4ComparisonMask::MASK_YZ); HK_TEST(MASK_YZ.getMask() == hkVector4ComparisonMask::MASK_YZ); masks.pushBack(MASK_YZ);
		hkVector4Comparison MASK_YZW; MASK_YZW.set(hkVector4ComparisonMask::MASK_YZW); HK_TEST(MASK_YZW.getMask() == hkVector4ComparisonMask::MASK_YZW); masks.pushBack(MASK_YZW);
		hkVector4Comparison MASK_X; MASK_X.set(hkVector4ComparisonMask::MASK_X); HK_TEST(MASK_X.getMask() == hkVector4ComparisonMask::MASK_X); masks.pushBack(MASK_X);
		hkVector4Comparison MASK_XW; MASK_XW.set(hkVector4ComparisonMask::MASK_XW); HK_TEST(MASK_XW.getMask() == hkVector4ComparisonMask::MASK_XW); masks.pushBack(MASK_XW);
		hkVector4Comparison MASK_XZ; MASK_XZ.set(hkVector4ComparisonMask::MASK_XZ); HK_TEST(MASK_XZ.getMask() == hkVector4ComparisonMask::MASK_XZ); masks.pushBack(MASK_XZ);
		hkVector4Comparison MASK_XZW; MASK_XZW.set(hkVector4ComparisonMask::MASK_XZW); HK_TEST(MASK_XZW.getMask() == hkVector4ComparisonMask::MASK_XZW); masks.pushBack(MASK_XZW);
		hkVector4Comparison MASK_XY; MASK_XY.set(hkVector4ComparisonMask::MASK_XY); HK_TEST(MASK_XY.getMask() == hkVector4ComparisonMask::MASK_XY); masks.pushBack(MASK_XY);
		hkVector4Comparison MASK_XYW; MASK_XYW.set(hkVector4ComparisonMask::MASK_XYW); HK_TEST(MASK_XYW.getMask() == hkVector4ComparisonMask::MASK_XYW); masks.pushBack(MASK_XYW);
		hkVector4Comparison MASK_XYZ; MASK_XYZ.set(hkVector4ComparisonMask::MASK_XYZ); HK_TEST(MASK_XYZ.getMask() == hkVector4ComparisonMask::MASK_XYZ); masks.pushBack(MASK_XYZ);
		hkVector4Comparison MASK_XYZW; MASK_XYZW.set(hkVector4ComparisonMask::MASK_XYZW); HK_TEST(MASK_XYZW.getMask() == hkVector4ComparisonMask::MASK_XYZW); masks.pushBack(MASK_XYZW);

		// Set (templated)
		{
			hkVector4Comparison test;
			test.set<hkVector4ComparisonMask::MASK_NONE>(); HK_TEST(test.getMask() == MASK_NONE.getMask());
			test.set<hkVector4ComparisonMask::MASK_W>(); HK_TEST(test.getMask() == MASK_W.getMask());
			test.set<hkVector4ComparisonMask::MASK_Z>(); HK_TEST(test.getMask() == MASK_Z.getMask());
			test.set<hkVector4ComparisonMask::MASK_ZW>(); HK_TEST(test.getMask() == MASK_ZW.getMask());
			test.set<hkVector4ComparisonMask::MASK_Y>(); HK_TEST(test.getMask() == MASK_Y.getMask());
			test.set<hkVector4ComparisonMask::MASK_YW>(); HK_TEST(test.getMask() == MASK_YW.getMask());
			test.set<hkVector4ComparisonMask::MASK_YZ>(); HK_TEST(test.getMask() == MASK_YZ.getMask());
			test.set<hkVector4ComparisonMask::MASK_YZW>(); HK_TEST(test.getMask() == MASK_YZW.getMask());
			test.set<hkVector4ComparisonMask::MASK_X>(); HK_TEST(test.getMask() == MASK_X.getMask());
			test.set<hkVector4ComparisonMask::MASK_XW>(); HK_TEST(test.getMask() == MASK_XW.getMask());
			test.set<hkVector4ComparisonMask::MASK_XZ>(); HK_TEST(test.getMask() == MASK_XZ.getMask());
			test.set<hkVector4ComparisonMask::MASK_XZW>(); HK_TEST(test.getMask() == MASK_XZW.getMask());
			test.set<hkVector4ComparisonMask::MASK_XY>(); HK_TEST(test.getMask() == MASK_XY.getMask());
			test.set<hkVector4ComparisonMask::MASK_XYW>(); HK_TEST(test.getMask() == MASK_XYW.getMask());
			test.set<hkVector4ComparisonMask::MASK_XYZ>(); HK_TEST(test.getMask() == MASK_XYZ.getMask());
			test.set<hkVector4ComparisonMask::MASK_XYZW>(); HK_TEST(test.getMask() == MASK_XYZW.getMask());
		}

		// setAnd
		{
			hkVector4Comparison a;

			// single components
			a.setAnd(MASK_X, MASK_W);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAnd(MASK_Y, MASK_W);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAnd(MASK_X, MASK_Y);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAnd(MASK_X, MASK_Z);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);


			a.setAnd(MASK_X, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);
			a.setAnd(MASK_Y, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Y);
			a.setAnd(MASK_Z, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Z);
			a.setAnd(MASK_W, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_W);


			// 2 components
			a.setAnd(MASK_XW, MASK_YZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAnd(MASK_XY, MASK_ZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAnd(MASK_XZ, MASK_YW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);

			a.setAnd(MASK_XY, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);
			a.setAnd(MASK_YZ, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Y);
			a.setAnd(MASK_ZW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_ZW);
			a.setAnd(MASK_XW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_W);

			a.setAnd(MASK_XY, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);
			a.setAnd(MASK_YZ, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZ);
			a.setAnd(MASK_ZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_ZW);
			a.setAnd(MASK_XW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XW);

			// 3 components
			a.setAnd(MASK_XYW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYW);
			a.setAnd(MASK_XYZ, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZ);
			a.setAnd(MASK_YZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZW);
			a.setAnd(MASK_XZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZW);

			a.setAnd(MASK_XYW, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYW);
			a.setAnd(MASK_XYW, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);
			a.setAnd(MASK_XYW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YW);
			a.setAnd(MASK_XYW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XW);

			a.setAnd(MASK_XYZ, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZ);
			a.setAnd(MASK_XYZ, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);
			a.setAnd(MASK_XYZ, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZ);

			a.setAnd(MASK_XZW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZW);
			a.setAnd(MASK_XZW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_ZW);
			a.setAnd(MASK_YZW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZW);

			// 4 components
			a.setAnd(MASK_XYZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setAnd(MASK_XYZW, MASK_NONE);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAnd(MASK_NONE, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
		}


		// setAndNot
		{
			hkVector4Comparison a;

			// single components
			a.setAndNot(MASK_X, MASK_W);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);
			a.setAndNot(MASK_Y, MASK_W);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Y);
			a.setAndNot(MASK_X, MASK_Y);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);
			a.setAndNot(MASK_X, MASK_Z);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);


			a.setAndNot(MASK_X, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_Y, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_Z, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_W, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);


			// 2 components
			a.setAndNot(MASK_XW, MASK_YZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XW);
			a.setAndNot(MASK_XY, MASK_ZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);
			a.setAndNot(MASK_XZ, MASK_YW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZ);

			a.setAndNot(MASK_XY, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_YZ, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Z);
			a.setAndNot(MASK_ZW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);

			a.setAndNot(MASK_XY, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_YZ, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_ZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);

			// 3 components
			a.setAndNot(MASK_XYW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XYZ, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_YZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);

			a.setAndNot(MASK_XYW, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XYW, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_W);
			a.setAndNot(MASK_XYW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);
			a.setAndNot(MASK_XYW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Y);

			a.setAndNot(MASK_XYZ, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XYZ, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Z);
			a.setAndNot(MASK_XYZ, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);

			a.setAndNot(MASK_XZW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XZW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);
			a.setAndNot(MASK_YZW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);

			// 4 components
			a.setAndNot(MASK_XYZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setAndNot(MASK_XYZW, MASK_NONE);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setAndNot(MASK_NONE, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
		}


		// setOr
		{
			hkVector4Comparison a;

			// single components
			a.setOr(MASK_X, MASK_W);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XW);
			a.setOr(MASK_Y, MASK_W);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YW);
			a.setOr(MASK_X, MASK_Y);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);
			a.setOr(MASK_X, MASK_Z);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZ);


			a.setOr(MASK_X, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_Y, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_Z, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_W, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);


			// 2 components
			a.setOr(MASK_XW, MASK_YZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XY, MASK_ZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XZ, MASK_YW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);

			a.setOr(MASK_XY, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZ);
			a.setOr(MASK_YZ, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_ZW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZW);
			a.setOr(MASK_XW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);

			a.setOr(MASK_XY, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_YZ, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_ZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);

			// 3 components
			a.setOr(MASK_XYW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XYZ, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_YZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);

			a.setOr(MASK_XYW, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYW);
			a.setOr(MASK_XYW, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XYW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XYW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);

			a.setOr(MASK_XYZ, MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZ);
			a.setOr(MASK_XYZ, MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XYZ, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);

			a.setOr(MASK_XZW, MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZW);
			a.setOr(MASK_XZW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_YZW, MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZW);

			// 4 components
			a.setOr(MASK_XYZW, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_XYZW, MASK_NONE);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
			a.setOr(MASK_NONE, MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
		}


		// setNot
		{
			hkVector4Comparison a;

			// single components
			a.setNot(MASK_X);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZW);
			a.setNot(MASK_Y);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZW);
			a.setNot(MASK_X);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZW);
			a.setNot(MASK_X);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZW);

			// 2 components
			a.setNot(MASK_XW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YZ);
			a.setNot(MASK_XY);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_ZW);
			a.setNot(MASK_XZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_YW);
			a.setNot(MASK_YZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XW);
			a.setNot(MASK_YW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XZ);
			a.setNot(MASK_ZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XY);

			// 3 components
			a.setNot(MASK_XYW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Z);
			a.setNot(MASK_XYZ);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_W);
			a.setNot(MASK_XZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_Y);
			a.setNot(MASK_YZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_X);

			// 4 components
			a.setNot(MASK_XYZW);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_NONE);
			a.setNot(MASK_NONE);
			HK_TEST(a.getMask() == hkVector4ComparisonMask::MASK_XYZW);
		}

		// setSelect
		{
			int size = masks.getSize();

			// Iterate through all the select masks
			for (int i = 0; i < size; i++)
			{
				const hkVector4Comparison& comp = masks[i];
				for (int j = 0; j < size; j++)
				{
					const hkVector4Comparison& trueValue = masks[j];
					for (int k = 0; k < size; k++)
					{
						const hkVector4Comparison& falseValue = masks[k];
						hkVector4Comparison result;
						result.setSelect(comp, trueValue, falseValue);

						{
							const hkVector4ComparisonMask::Mask m = hkVector4ComparisonMask::MASK_X;
							HK_TEST(bool(result.allAreSet(m)) == bool(comp.allAreSet(m) ? trueValue.allAreSet(m) : falseValue.allAreSet(m)) );
						}
						{
							const hkVector4ComparisonMask::Mask m = hkVector4ComparisonMask::MASK_Y;
							HK_TEST(bool(result.allAreSet(m)) == bool(comp.allAreSet(m) ? trueValue.allAreSet(m) : falseValue.allAreSet(m)) );
						}
						{
							const hkVector4ComparisonMask::Mask m = hkVector4ComparisonMask::MASK_Z;
							HK_TEST(bool(result.allAreSet(m)) == bool(comp.allAreSet(m) ? trueValue.allAreSet(m) : falseValue.allAreSet(m)) );
						}
						{
							const hkVector4ComparisonMask::Mask m = hkVector4ComparisonMask::MASK_W;
							HK_TEST(bool(result.allAreSet(m)) == bool(comp.allAreSet(m) ? trueValue.allAreSet(m) : falseValue.allAreSet(m)) );
						}
					}
				}
			}

		}

		// allAreSet
		{
			HK_TEST(MASK_NONE.allAreSet()	== hkFalse32);
			HK_TEST(MASK_W.allAreSet()		== hkFalse32);
			HK_TEST(MASK_Z.allAreSet()		== hkFalse32);
			HK_TEST(MASK_ZW.allAreSet()		== hkFalse32);
			HK_TEST(MASK_Y.allAreSet()		== hkFalse32);
			HK_TEST(MASK_YW.allAreSet()		== hkFalse32);
			HK_TEST(MASK_YZ.allAreSet()		== hkFalse32);
			HK_TEST(MASK_YZW.allAreSet()	== hkFalse32);
			HK_TEST(MASK_X.allAreSet()		== hkFalse32);
			HK_TEST(MASK_XW.allAreSet()		== hkFalse32);
			HK_TEST(MASK_XZ.allAreSet()		== hkFalse32);
			HK_TEST(MASK_XZW.allAreSet()	== hkFalse32);
			HK_TEST(MASK_XY.allAreSet()		== hkFalse32);
			HK_TEST(MASK_XYW.allAreSet()	== hkFalse32);
			HK_TEST(MASK_XYZ.allAreSet()	== hkFalse32);
			HK_TEST(MASK_XYZW.allAreSet());
		}

		// anyIsSet
		{
			HK_TEST(MASK_NONE.anyIsSet()	== hkFalse32);
			HK_TEST(MASK_W.anyIsSet());
			HK_TEST(MASK_Z.anyIsSet());
			HK_TEST(MASK_ZW.anyIsSet());
			HK_TEST(MASK_Y.anyIsSet());
			HK_TEST(MASK_YW.anyIsSet());
			HK_TEST(MASK_YZ.anyIsSet());
			HK_TEST(MASK_YZW.anyIsSet());
			HK_TEST(MASK_X.anyIsSet());
			HK_TEST(MASK_XW.anyIsSet());
			HK_TEST(MASK_XZ.anyIsSet());
			HK_TEST(MASK_XZW.anyIsSet());
			HK_TEST(MASK_XY.anyIsSet());
			HK_TEST(MASK_XYW.anyIsSet());
			HK_TEST(MASK_XYZ.anyIsSet());
			HK_TEST(MASK_XYZW.anyIsSet());
		}
	}
}

void vector_vector4comparisons2()
{
	{
		// set and prep for other tests.
		hkVector4Comparison MASK_NONE; MASK_NONE.set(hkVector4ComparisonMask::MASK_NONE);
		hkVector4Comparison MASK_W; MASK_W.set(hkVector4ComparisonMask::MASK_W);
		hkVector4Comparison MASK_Z; MASK_Z.set(hkVector4ComparisonMask::MASK_Z);
		hkVector4Comparison MASK_ZW; MASK_ZW.set(hkVector4ComparisonMask::MASK_ZW);
		hkVector4Comparison MASK_Y; MASK_Y.set(hkVector4ComparisonMask::MASK_Y);
		hkVector4Comparison MASK_YW; MASK_YW.set(hkVector4ComparisonMask::MASK_YW);
		hkVector4Comparison MASK_YZ; MASK_YZ.set(hkVector4ComparisonMask::MASK_YZ);
		hkVector4Comparison MASK_YZW; MASK_YZW.set(hkVector4ComparisonMask::MASK_YZW);
		hkVector4Comparison MASK_X; MASK_X.set(hkVector4ComparisonMask::MASK_X);
		hkVector4Comparison MASK_XW; MASK_XW.set(hkVector4ComparisonMask::MASK_XW);
		hkVector4Comparison MASK_XZ; MASK_XZ.set(hkVector4ComparisonMask::MASK_XZ);
		hkVector4Comparison MASK_XZW; MASK_XZW.set(hkVector4ComparisonMask::MASK_XZW);
		hkVector4Comparison MASK_XY; MASK_XY.set(hkVector4ComparisonMask::MASK_XY);
		hkVector4Comparison MASK_XYW; MASK_XYW.set(hkVector4ComparisonMask::MASK_XYW);
		hkVector4Comparison MASK_XYZ; MASK_XYZ.set(hkVector4ComparisonMask::MASK_XYZ);
		hkVector4Comparison MASK_XYZW; MASK_XYZW.set(hkVector4ComparisonMask::MASK_XYZW);

		// allAreSet (Mask)
		{
			// MASK_NONE
			HK_TEST(MASK_NONE.allAreSet(MASK_NONE.getMask()) 			);
			HK_TEST(MASK_W.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_Z.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_ZW.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_Y.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_YW.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_YZ.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_X.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XW.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XZ.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XY.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_NONE.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_NONE.getMask())			);

			// MASK_W
			HK_TEST(!MASK_NONE.allAreSet(MASK_W.getMask())			);
			HK_TEST(MASK_W.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_ZW.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_YW.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_XW.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_W.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_W.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_W.getMask())			);

			// MASK_Z
			HK_TEST(!MASK_NONE.allAreSet(MASK_Z.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_Z.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_ZW.allAreSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_YZ.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XZ.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_Z.getMask())			);

			// MASK_ZW
			HK_TEST(!MASK_NONE.allAreSet(MASK_ZW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_ZW.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_ZW.getMask())			);

			// MASK_Y
			HK_TEST(!MASK_NONE.allAreSet(MASK_Y.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_Y.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_YW.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_YZ.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XY.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_Y.getMask())			);

			// MASK_YW
			HK_TEST(!MASK_NONE.allAreSet(MASK_YW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_YW.getMask())				);
			HK_TEST(MASK_YW.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_YW.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_YW.getMask())			);

			// MASK_YZ
			HK_TEST(!MASK_NONE.allAreSet(MASK_YZ.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_YZ.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_YZ.getMask())			);

			// MASK_YZW
			HK_TEST(!MASK_NONE.allAreSet(MASK_YZW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_YZW.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_YZW.getMask())			);

			// MASK_X
			HK_TEST(!MASK_NONE.allAreSet(MASK_X.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_X.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_X.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_X.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_X.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_X.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_X.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_X.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XW.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XZ.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XY.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_X.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_X.getMask())			);

			// MASK_XW
			HK_TEST(!MASK_NONE.allAreSet(MASK_XW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XW.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XW.getMask())			);

			// MASK_XZ
			HK_TEST(!MASK_NONE.allAreSet(MASK_XZ.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XZ.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XZ.getMask())			);

			// MASK_XZW
			HK_TEST(!MASK_NONE.allAreSet(MASK_XZW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XZW.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XZW.getMask())			);

			// MASK_XY
			HK_TEST(!MASK_NONE.allAreSet(MASK_XY.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XY.allAreSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XY.getMask())			);

			// MASK_XYW
			HK_TEST(!MASK_NONE.allAreSet(MASK_XYW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XYW.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XYW.getMask())			);

			// MASK_XYZ
			HK_TEST(!MASK_NONE.allAreSet(MASK_XYZ.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XYZ.allAreSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XYZ.getMask())			);

			// MASK_XYZW
			HK_TEST(!MASK_NONE.allAreSet(MASK_XYZW.getMask())			);
			HK_TEST(!MASK_W.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_Z.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_ZW.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_Y.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_YW.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_YZ.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_YZW.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_X.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_XW.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_XZ.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_XZW.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_XY.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_XYW.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(!MASK_XYZ.allAreSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XYZW.allAreSet(MASK_XYZW.getMask())			);
		}
	}
}

void vector_vector4comparisons3()
{
	{
		// set and prep for other tests.
		hkVector4Comparison MASK_NONE; MASK_NONE.set(hkVector4ComparisonMask::MASK_NONE);
		hkVector4Comparison MASK_W; MASK_W.set(hkVector4ComparisonMask::MASK_W);
		hkVector4Comparison MASK_Z; MASK_Z.set(hkVector4ComparisonMask::MASK_Z);
		hkVector4Comparison MASK_ZW; MASK_ZW.set(hkVector4ComparisonMask::MASK_ZW);
		hkVector4Comparison MASK_Y; MASK_Y.set(hkVector4ComparisonMask::MASK_Y);
		hkVector4Comparison MASK_YW; MASK_YW.set(hkVector4ComparisonMask::MASK_YW);
		hkVector4Comparison MASK_YZ; MASK_YZ.set(hkVector4ComparisonMask::MASK_YZ);
		hkVector4Comparison MASK_YZW; MASK_YZW.set(hkVector4ComparisonMask::MASK_YZW);
		hkVector4Comparison MASK_X; MASK_X.set(hkVector4ComparisonMask::MASK_X);
		hkVector4Comparison MASK_XW; MASK_XW.set(hkVector4ComparisonMask::MASK_XW);
		hkVector4Comparison MASK_XZ; MASK_XZ.set(hkVector4ComparisonMask::MASK_XZ);
		hkVector4Comparison MASK_XZW; MASK_XZW.set(hkVector4ComparisonMask::MASK_XZW);
		hkVector4Comparison MASK_XY; MASK_XY.set(hkVector4ComparisonMask::MASK_XY);
		hkVector4Comparison MASK_XYW; MASK_XYW.set(hkVector4ComparisonMask::MASK_XYW);
		hkVector4Comparison MASK_XYZ; MASK_XYZ.set(hkVector4ComparisonMask::MASK_XYZ);
		hkVector4Comparison MASK_XYZW; MASK_XYZW.set(hkVector4ComparisonMask::MASK_XYZW);

		// anyIsSet (Mask)
		{
			// MASK_NONE
			HK_TEST(!MASK_NONE.anyIsSet(MASK_NONE.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_ZW.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_YW.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_YZ.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_YZW.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XW.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XZ.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XZW.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XY.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XYW.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XYZ.anyIsSet(MASK_NONE.getMask())				);
			HK_TEST(!MASK_XYZW.anyIsSet(MASK_NONE.getMask())			);

			// MASK_W
			HK_TEST(!MASK_NONE.anyIsSet(MASK_W.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_YZ.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_XZ.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_XY.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_W.getMask())				);
			HK_TEST(!MASK_XYZ.anyIsSet(MASK_W.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_W.getMask())			);

			// MASK_Z
			HK_TEST(!MASK_NONE.anyIsSet(MASK_Z.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_YW.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_XW.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_XY.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(!MASK_XYW.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_Z.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_Z.getMask())			);

			// MASK_ZW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_ZW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(!MASK_XY.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_ZW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_ZW.getMask())			);

			// MASK_Y
			HK_TEST(!MASK_NONE.anyIsSet(MASK_Y.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_ZW.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_XW.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_XZ.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(!MASK_XZW.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_Y.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_Y.getMask())			);

			// MASK_YW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_YW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(!MASK_XZ.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_YW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_YW.getMask())			);

			// MASK_YZ
			HK_TEST(!MASK_NONE.anyIsSet(MASK_YZ.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(!MASK_XW.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_YZ.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_YZ.getMask())			);

			// MASK_YZW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_YZW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(!MASK_X.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_YZW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_YZW.getMask())			);

			// MASK_X
			HK_TEST(!MASK_NONE.anyIsSet(MASK_X.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_X.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_X.getMask())				);
			HK_TEST(!MASK_ZW.anyIsSet(MASK_X.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_X.getMask())				);
			HK_TEST(!MASK_YW.anyIsSet(MASK_X.getMask())				);
			HK_TEST(!MASK_YZ.anyIsSet(MASK_X.getMask())				);
			HK_TEST(!MASK_YZW.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_X.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_X.getMask())			);

			// MASK_XW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(!MASK_YZ.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XW.getMask())			);

			// MASK_XZ
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XZ.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(!MASK_YW.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XZ.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XZ.getMask())			);

			// MASK_XZW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XZW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(!MASK_Y.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XZW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XZW.getMask())			);

			// MASK_XY
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XY.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(!MASK_ZW.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XY.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XY.getMask())			);

			// MASK_XYW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XYW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(!MASK_Z.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XYW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XYW.getMask())			);

			// MASK_XYZ
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XYZ.getMask())			);
			HK_TEST(!MASK_W.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XYZ.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XYZ.getMask())			);

			// MASK_XYZW
			HK_TEST(!MASK_NONE.anyIsSet(MASK_XYZW.getMask())			);
			HK_TEST(MASK_W.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_Z.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_ZW.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_Y.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_YW.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_YZ.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_YZW.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_X.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XW.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XZ.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XZW.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XY.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XYW.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XYZ.anyIsSet(MASK_XYZW.getMask())				);
			HK_TEST(MASK_XYZW.anyIsSet(MASK_XYZW.getMask())			);
		}
	}
}

void vector_vector4comparisons4()
{
	{
		// set and prep for other tests.
		hkVector4Comparison MASK_NONE; MASK_NONE.set(hkVector4ComparisonMask::MASK_NONE);
		hkVector4Comparison MASK_W; MASK_W.set(hkVector4ComparisonMask::MASK_W);
		hkVector4Comparison MASK_Z; MASK_Z.set(hkVector4ComparisonMask::MASK_Z);
		hkVector4Comparison MASK_ZW; MASK_ZW.set(hkVector4ComparisonMask::MASK_ZW);
		hkVector4Comparison MASK_Y; MASK_Y.set(hkVector4ComparisonMask::MASK_Y);
		hkVector4Comparison MASK_YW; MASK_YW.set(hkVector4ComparisonMask::MASK_YW);
		hkVector4Comparison MASK_YZ; MASK_YZ.set(hkVector4ComparisonMask::MASK_YZ);
		hkVector4Comparison MASK_YZW; MASK_YZW.set(hkVector4ComparisonMask::MASK_YZW);
		hkVector4Comparison MASK_X; MASK_X.set(hkVector4ComparisonMask::MASK_X);
		hkVector4Comparison MASK_XW; MASK_XW.set(hkVector4ComparisonMask::MASK_XW);
		hkVector4Comparison MASK_XZ; MASK_XZ.set(hkVector4ComparisonMask::MASK_XZ);
		hkVector4Comparison MASK_XZW; MASK_XZW.set(hkVector4ComparisonMask::MASK_XZW);
		hkVector4Comparison MASK_XY; MASK_XY.set(hkVector4ComparisonMask::MASK_XY);
		hkVector4Comparison MASK_XYW; MASK_XYW.set(hkVector4ComparisonMask::MASK_XYW);
		hkVector4Comparison MASK_XYZ; MASK_XYZ.set(hkVector4ComparisonMask::MASK_XYZ);
		hkVector4Comparison MASK_XYZW; MASK_XYZW.set(hkVector4ComparisonMask::MASK_XYZW);

		// getMask (Mask)
		{
			// MASK_NONE
			HK_TEST(MASK_NONE.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Y.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_X.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XY.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_NONE.getMask()) == MASK_NONE.getMask());

			// MASK_W
			HK_TEST(MASK_NONE.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Y.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_X.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XY.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_W.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_W.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_W.getMask()) == MASK_W.getMask());

			// MASK_Z
			HK_TEST(MASK_NONE.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_Y.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_X.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XY.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_Z.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_Z.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_Z.getMask()) == MASK_Z.getMask());

			// MASK_ZW
			HK_TEST(MASK_NONE.getMask(MASK_ZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_ZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_ZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_ZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_Y.getMask(MASK_ZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_ZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_ZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_ZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_X.getMask(MASK_ZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_ZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_ZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_ZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_XY.getMask(MASK_ZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_ZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_ZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_ZW.getMask()) == MASK_ZW.getMask());

			// MASK_Y
			HK_TEST(MASK_NONE.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Y.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_X.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_Y.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XY.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_Y.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_Y.getMask()) == MASK_Y.getMask());

			// MASK_YW
			HK_TEST(MASK_NONE.getMask(MASK_YW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_YW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_YW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_YW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Y.getMask(MASK_YW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_YW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_YW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_YW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_X.getMask(MASK_YW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_YW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_YW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_YW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XY.getMask(MASK_YW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_YW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_YW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_YW.getMask()) == MASK_YW.getMask());

			// MASK_YZ
			HK_TEST(MASK_NONE.getMask(MASK_YZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_YZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_YZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_YZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_Y.getMask(MASK_YZ.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_YZ.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_YZ.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_YZ.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_X.getMask(MASK_YZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_YZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_YZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_YZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XY.getMask(MASK_YZ.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_YZ.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_YZ.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_YZ.getMask()) == MASK_YZ.getMask());

			// MASK_YZW
			HK_TEST(MASK_NONE.getMask(MASK_YZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_YZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_YZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_YZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_Y.getMask(MASK_YZW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_YZW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_YZW.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_YZW.getMask()) == MASK_YZW.getMask());
			HK_TEST(MASK_X.getMask(MASK_YZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_XW.getMask(MASK_YZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_YZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_YZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_XY.getMask(MASK_YZW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_YZW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_YZW.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_YZW.getMask()) == MASK_YZW.getMask());

			// MASK_X
			HK_TEST(MASK_NONE.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Y.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_X.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_X.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XY.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_X.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_X.getMask()) == MASK_X.getMask());

			// MASK_XW
			HK_TEST(MASK_NONE.getMask(MASK_XW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_X.getMask(MASK_XW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XW.getMask()) == MASK_XW.getMask());

			// MASK_XZ
			HK_TEST(MASK_NONE.getMask(MASK_XZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_X.getMask(MASK_XZ.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XZ.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XZ.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XZ.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XZ.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XZ.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XZ.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XZ.getMask()) == MASK_XZ.getMask());

			// MASK_XZW
			HK_TEST(MASK_NONE.getMask(MASK_XZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_X.getMask(MASK_XZW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XZW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XZW.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XZW.getMask()) == MASK_XZW.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XZW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XZW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XZW.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XZW.getMask()) == MASK_XZW.getMask());

			// MASK_XY
			HK_TEST(MASK_NONE.getMask(MASK_XY.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XY.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XY.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XY.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XY.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XY.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XY.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XY.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_X.getMask(MASK_XY.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XY.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XY.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XY.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XY.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XY.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XY.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XY.getMask()) == MASK_XY.getMask());

			// MASK_XYW
			HK_TEST(MASK_NONE.getMask(MASK_XYW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XYW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XYW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XYW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XYW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XYW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XYW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XYW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_X.getMask(MASK_XYW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XYW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XYW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XYW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XYW.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XYW.getMask()) == MASK_XYW.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XYW.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XYW.getMask()) == MASK_XYW.getMask());

			// MASK_XYZ
			HK_TEST(MASK_NONE.getMask(MASK_XYZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XYZ.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XYZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XYZ.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XYZ.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XYZ.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XYZ.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XYZ.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_X.getMask(MASK_XYZ.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XYZ.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XYZ.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XYZ.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XYZ.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XYZ.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XYZ.getMask()) == MASK_XYZ.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XYZ.getMask()) == MASK_XYZ.getMask());

			// MASK_XYZW
			HK_TEST(MASK_NONE.getMask(MASK_XYZW.getMask()) == MASK_NONE.getMask());
			HK_TEST(MASK_W.getMask(MASK_XYZW.getMask()) == MASK_W.getMask());
			HK_TEST(MASK_Z.getMask(MASK_XYZW.getMask()) == MASK_Z.getMask());
			HK_TEST(MASK_ZW.getMask(MASK_XYZW.getMask()) == MASK_ZW.getMask());
			HK_TEST(MASK_Y.getMask(MASK_XYZW.getMask()) == MASK_Y.getMask());
			HK_TEST(MASK_YW.getMask(MASK_XYZW.getMask()) == MASK_YW.getMask());
			HK_TEST(MASK_YZ.getMask(MASK_XYZW.getMask()) == MASK_YZ.getMask());
			HK_TEST(MASK_YZW.getMask(MASK_XYZW.getMask()) == MASK_YZW.getMask());
			HK_TEST(MASK_X.getMask(MASK_XYZW.getMask()) == MASK_X.getMask());
			HK_TEST(MASK_XW.getMask(MASK_XYZW.getMask()) == MASK_XW.getMask());
			HK_TEST(MASK_XZ.getMask(MASK_XYZW.getMask()) == MASK_XZ.getMask());
			HK_TEST(MASK_XZW.getMask(MASK_XYZW.getMask()) == MASK_XZW.getMask());
			HK_TEST(MASK_XY.getMask(MASK_XYZW.getMask()) == MASK_XY.getMask());
			HK_TEST(MASK_XYW.getMask(MASK_XYZW.getMask()) == MASK_XYW.getMask());
			HK_TEST(MASK_XYZ.getMask(MASK_XYZW.getMask()) == MASK_XYZ.getMask());
			HK_TEST(MASK_XYZW.getMask(MASK_XYZW.getMask()) == MASK_XYZW.getMask());
		}
	}
}

static void vector_comparisons()
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
#if defined(HK_REAL_IS_DOUBLE)
	static HK_ALIGN_REAL( const hkUint64 mZ[2] ) = { 0x8000000000000000ull, 0x8000000000000000ull };
#else
	static HK_ALIGN_REAL( const hkUint32 mZ[4] ) = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
#endif
#else
#if defined(HK_REAL_IS_DOUBLE)
	const hkUint64 mZ = 0x8000000000000000ull;
#else
	const hkUint32 mZ = 0x80000000;
#endif
#endif
	hkSimdReal minusZero; minusZero.m_real = *(hkSingleReal*)&mZ;

	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(0,0,0,0);
		hkVector4 z; z.set(5,2,1,9);
		hkVector4 w; w.set(5,3,2,3);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		hkBool32 check = x.allEqual<3>(y,eps);
		HK_TEST( !check );

		HK_TEST( !x.allEqual<4>(y,eps));
		HK_TEST(  x.allEqual<3>(z,eps));
		HK_TEST( !x.allEqual<4>(z,eps));
		HK_TEST( !x.allEqual<3>(w,eps));
		HK_TEST( !x.allEqual<4>(w,eps));

		HK_TEST( !y.allEqual<3>(z,eps));
		HK_TEST( !y.allEqual<4>(z,eps));
		HK_TEST( !y.allEqual<3>(w,eps));
		HK_TEST( !y.allEqual<4>(w,eps));

		HK_TEST( !z.allEqual<3>(w,eps));
		HK_TEST( !z.allEqual<4>(w,eps));
	}

	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(92,4,2,-1);
		y = x;
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(x.allEqual<4>(y,eps));
	}

	// simdreal comparison
	// getComponent.
	{
		hkVector4 x;
		hkSimdReal five = hkSimdReal::getConstant<HK_QUADREAL_5>();
		hkSimdReal two  = hkSimdReal::getConstant<HK_QUADREAL_2>();
		hkSimdReal one  = hkSimdReal::getConstant<HK_QUADREAL_1>();
		hkSimdReal three= hkSimdReal::getConstant<HK_QUADREAL_3>();
		x.set(five, two, one, three);
		HK_TEST(x.getComponent(0)==five);
		HK_TEST(x.getComponent(1)==two);
		HK_TEST(x.getComponent(2)==one);
		HK_TEST(x.getComponent(3)==three);
		HK_TEST_ASSERT(0x6d0c31d7, x.getComponent(-1));
		HK_TEST_ASSERT(0x6d0c31d7, x.getComponent(4));
		HK_TEST_ASSERT(0x6d0c31d7, x.getComponent(5));

		HK_TEST(x.getComponent<0>()==five);
		HK_TEST(x.getComponent<1>()==two);
		HK_TEST(x.getComponent<2>()==one);
		HK_TEST(x.getComponent<3>()==three);
	}

	{
		hkVector4 x; x.set(1,2,6,9);
		hkVector4 y; y.set(1,2,6,99);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<3>(y,eps) );
		HK_TEST( ! x.allEqual<4>(y,eps) );
	}
	{
		hkVector4 x; x.set(1,2,6,9);
		hkVector4 y; y.set(1,2,6,99);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<3>(y,eps) );
		HK_TEST( ! x.allEqual<4>(y,eps) );
	}

	{
		hkVector4 x; x.set(1,-2,-0,99); x.setComponent<2>(minusZero);
		hkVector4 y; y.set(-1,-2,10,-99);
		int mx = x.lessZero().getMask();
		int my = y.lessZero().getMask();
		HK_TEST( mx == hkVector4ComparisonMask::MASK_Y );
		HK_TEST( my == hkVector4ComparisonMask::MASK_XYW );
	}

	// compareLessThan & lessEqual
	{
		hkVector4 x; x.set( 1, 5,-3,99);
		hkVector4 y; y.set(-1, 5,0,100);
		int cle = x.lessEqual( y ).getMask();
		HK_TEST( (cle & hkVector4ComparisonMask::MASK_X) != hkVector4ComparisonMask::MASK_X );
		HK_TEST( (cle & hkVector4ComparisonMask::MASK_Y) == hkVector4ComparisonMask::MASK_Y );
		HK_TEST( (cle & hkVector4ComparisonMask::MASK_ZW) == hkVector4ComparisonMask::MASK_ZW );

		int clt = x.less( y ).getMask();
		HK_TEST( (clt & hkVector4ComparisonMask::MASK_X) != hkVector4ComparisonMask::MASK_X );
		HK_TEST( (clt & hkVector4ComparisonMask::MASK_Y) != hkVector4ComparisonMask::MASK_Y );
		HK_TEST( (clt & hkVector4ComparisonMask::MASK_ZW) == hkVector4ComparisonMask::MASK_ZW );
	}

	// lessZero
	{
		hkVector4 x; x.set(1,-2,-0,99); x.setComponent<2>(minusZero);
		hkVector4 y; y.set(-1,-2,-10,-99);
		hkVector4Comparison mx = x.lessZero();
		hkVector4Comparison my = y.lessZero();
		HK_TEST( mx.allAreSet() == hkFalse32);
		HK_TEST( mx.allAreSet(hkVector4ComparisonMask::MASK_Y)  );
		HK_TEST( !mx.allAreSet(hkVector4ComparisonMask::MASK_X) );
		HK_TEST( !mx.allAreSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( mx.anyIsSet(hkVector4ComparisonMask::MASK_Y) );
		HK_TEST( mx.anyIsSet(hkVector4ComparisonMask::MASK_XYZW) );
		HK_TEST( !mx.anyIsSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( my.allAreSet() );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_Y)  );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_X) );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_Y) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_XYZW) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_XW) );
	}

	// greaterZero
	{
		hkVector4 x; x.set(1,-2,-0,99); x.setComponent<2>(minusZero);
		hkVector4 y; y.set(1,2,10,99);
		hkVector4Comparison mx = x.greaterZero();
		hkVector4Comparison my = y.greaterZero();
		HK_TEST( mx.allAreSet() == hkFalse32 );
		HK_TEST( !mx.allAreSet(hkVector4ComparisonMask::MASK_Y));
		HK_TEST( mx.allAreSet(hkVector4ComparisonMask::MASK_X) );
		HK_TEST( mx.allAreSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( !mx.allAreSet(hkVector4ComparisonMask::MASK_XY));
		HK_TEST( !mx.anyIsSet(hkVector4ComparisonMask::MASK_Y));
		HK_TEST( mx.anyIsSet(hkVector4ComparisonMask::MASK_XYZW) );
		HK_TEST( mx.anyIsSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( my.allAreSet() );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_Y)  );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_X) );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_Y) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_XYZW) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_XW) );
	}

	// greaterEqualZero
	{
		hkVector4 x; x.set(1,-2,-0,99); x.setComponent<2>(minusZero);
		hkVector4 y; y.set(1,0,10,99);
		hkVector4Comparison mx = x.greaterEqualZero();
		hkVector4Comparison my = y.greaterEqualZero();
		HK_TEST( !mx.allAreSet());
		HK_TEST( !mx.allAreSet(hkVector4ComparisonMask::MASK_Y));
		HK_TEST( mx.allAreSet(hkVector4ComparisonMask::MASK_Z) );
		HK_TEST( mx.allAreSet(hkVector4ComparisonMask::MASK_X) );
		HK_TEST( mx.allAreSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( !mx.anyIsSet(hkVector4ComparisonMask::MASK_Y));
		HK_TEST( mx.anyIsSet(hkVector4ComparisonMask::MASK_XYZW) );
		HK_TEST( mx.anyIsSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( my.allAreSet() );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_Y)  );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_X) );
		HK_TEST( my.allAreSet(hkVector4ComparisonMask::MASK_XW) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_Y) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_XYZW) );
		HK_TEST( my.anyIsSet(hkVector4ComparisonMask::MASK_XW) );
	}

	// Testing the functionality of equal()
	{
		hkVector4 x;
		hkVector4 y;

		x.set(1,5,6,7);
		y.set(1,2,3,4);
		int m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_X );

		x.set(7,2,8,9);
		y.set(1,2,3,4);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_Y );

		x.set(7,7,3,8);
		y.set(1,2,3,4);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_Z );

		x.set(1,2,8,7);
		y.set(1,2,3,4);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_XY );

		x.set(1,4,5,8);
		y.set(2,4,5,6);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_YZ );

		x.set(1,4,5,8);
		y.set(6,7,5,8);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_ZW );

		x.set(1,2,3,4);
		y.set(1,2,3,4);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_XYZW );

		x.set(0,2,3,4);
		y.set(-0,6,7,8); y.setComponent<0>(minusZero);
		m = x.equal(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_X );
	}

	// notEqual
	{
		hkVector4 x;
		hkVector4 y;

		x.set(1,5,6,7);
		y.set(0,5,6,7);
		int m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_X );

		x.set(7,2,8,9);
		y.set(7,1,8,9);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_Y );

		x.set(7,7,3,8);
		y.set(7,7,5,8);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_Z );

		x.set(1,2,8,7);
		y.set(4,5,8,7);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_XY );

		x.set(1,4,5,8);
		y.set(1,2,7,8);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_YZ );

		x.set(1,4,5,8);
		y.set(1,4,7,9);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_ZW );

		x.set(1,2,3,4);
		y.set(5,6,7,8);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_XYZW );

		x.set(0,2,3,4);
		y.set(-0,6,7,8); y.setComponent<0>(minusZero);
		m = x.notEqual(y).getMask();
		HK_TEST( m == hkVector4ComparisonMask::MASK_YZW );
	}

	// Testing component equal
	{
		hkVector4 a; a.set(0,0,0,1);
		hkVector4 b; b.set(0,1,1,1);
		hkVector4 c; c.setZero();
		hkVector4 d; d.set(-0,0,0,0);

		HK_TEST( a.allComponentsEqual<1>() );
		HK_TEST( a.allComponentsEqual<2>() );
		HK_TEST( a.allComponentsEqual<3>() );
		HK_TEST( !a.allComponentsEqual<4>() );

		HK_TEST( !b.allComponentsEqual<2>() );
		HK_TEST( c.allComponentsEqual<4>() );
		HK_TEST( d.allComponentsEqual<2>() ); // its a by-value operation!
		HK_TEST( d.allComponentsEqual<4>() );
	}

	// Testing the functionality of greaterEqual() and greater()
	{
		hkVector4 x;
		hkVector4 y;

		x.set(-1, 5,0,100);
		y.set(1, 5,-3,99);

		int cle = x.greaterEqual( y ).getMask();
		HK_TEST( (cle & hkVector4ComparisonMask::MASK_X) != hkVector4ComparisonMask::MASK_X );
		HK_TEST( (cle & hkVector4ComparisonMask::MASK_Y) == hkVector4ComparisonMask::MASK_Y );
		HK_TEST( (cle & hkVector4ComparisonMask::MASK_ZW) == hkVector4ComparisonMask::MASK_ZW );

		int clt = x.greater( y ).getMask();
		HK_TEST( (clt & hkVector4ComparisonMask::MASK_X) != hkVector4ComparisonMask::MASK_X );
		HK_TEST( (clt & hkVector4ComparisonMask::MASK_Y) != hkVector4ComparisonMask::MASK_Y );
		HK_TEST( (clt & hkVector4ComparisonMask::MASK_ZW) == hkVector4ComparisonMask::MASK_ZW );
	}

	// signBitSet
	{
		hkVector4 x; x.set(1,-2,0,99);
		hkVector4Comparison mx = x.signBitSet();
		HK_TEST( mx.allAreSet() == hkFalse32 );
		HK_TEST( mx.allAreSet<hkVector4ComparisonMask::MASK_Y>() );
		HK_TEST( !mx.allAreSet<hkVector4ComparisonMask::MASK_Z>());
	}
	{
		hkVector4 x; x.set(1,-2,-0.0f,99); x.setComponent<2>(minusZero);
		hkVector4Comparison mx = x.signBitSet();
		HK_TEST( mx.allAreSet<hkVector4ComparisonMask::MASK_Z>() );
	}

	// signBitClear
	{
		hkVector4 x; x.set(1,-2,0,99);
		hkVector4Comparison mx = x.signBitClear();
		HK_TEST( mx.allAreSet() == hkFalse32 );
		HK_TEST( !mx.allAreSet<hkVector4ComparisonMask::MASK_Y>() );
		HK_TEST( mx.allAreSet<hkVector4ComparisonMask::MASK_Z>());
	}
	{
		hkVector4 x; x.set(1,-2,-0.0f,99); x.setComponent<2>(minusZero);
		hkVector4Comparison mx = x.signBitClear();
		HK_TEST( !mx.allAreSet<hkVector4ComparisonMask::MASK_Z>() );
	}

	// Testing the functionality of allLess
	{
		hkVector4 x;
		hkVector4 y;
		hkVector4 z;
		hkVector4 w;

		x.set(1,2,3,100);
		y.set(2,4,5,99);

		HK_TEST(x.allLess<3>(y));
		HK_TEST(! x.allLess<4>(y));

		z.set(2,4,5,102);
		HK_TEST(x.allLess<4>(z));

		w.set(2,4,2,2);
		HK_TEST(x.allLess<2>(w));
	}

	// combined mask
	{
		hkPseudoRandomGenerator rnd(3);
		for (int i =0; i < 128; i++ )
		{
			hkVector4 a; rnd.getRandomVector11(a);
			hkVector4 b; rnd.getRandomVector11(b);
			hkVector4 c; rnd.getRandomVector11(c);
			hkVector4Comparison ca = a.greaterZero();
			hkVector4Comparison cb = b.greaterZero();
			hkVector4Comparison cc = c.greaterZero();
			int cmask = hkVector4Comparison::getCombinedMask( ca,cb,cc );
			HK_TEST( ((cmask>>0)&hkVector4ComparisonMask::MASK_XYZW) == ca.getMask() );
			HK_TEST( ((cmask>>4)&hkVector4ComparisonMask::MASK_XYZW) == cb.getMask() );
			HK_TEST( ((cmask>>8)&hkVector4ComparisonMask::MASK_XYZW) == cc.getMask() );
		}
	}
}

#if defined(HK_PLATFORM_WINRT) && defined(HK_ARCH_ARM)
#pragma optimize("", on)
#endif

static void vector_ops()
{
	// + vec
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		x.add( y );
		hkVector4 twoy; twoy.setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(), y);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<3>(twoy,eps));
		HK_TEST( x.allEqual<4>(twoy,eps));
	}

	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		x.setAdd( x, y );
		hkVector4 twoy; twoy.setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(), y);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<3>(twoy,eps));
		HK_TEST( x.allEqual<4>(twoy,eps));
	}

	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		x.addXYZ( y );
		hkVector4 twoy; twoy.setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(), y);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<3>(twoy,eps));
		HK_ON_DEBUG(checkEqualNan(x(3)));
	}

	// + real
	{
		// test hkVector4::setAdd(hkVector4, hkSimdReal)
		hkVector4 x; x.set(5,2,1,3);
		hkSimdReal y = hkSimdReal_1;
		hkVector4 z; z.setAdd( x, y );
		
		hkVector4 yVec; yVec.setAll(y);
		hkVector4 z2; z2.setAdd(x, yVec);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(z2,eps));
	}

	// - vec
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		x.sub( y );
		y.setZero();
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(y.allEqual<4>(x,eps));
	}
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		x.setSub( x, y );
		y.setZero();
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(y.allEqual<4>(x,eps));
	}
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(1,2,3,4);
		x.subXYZ( y );
		hkVector4 a; a.set(4,0,-2,-1);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(x.allEqual<3>(a,eps));
		HK_ON_DEBUG(checkEqualNan(x(3)));
	}

	// - real
	{
		// test hkVector4::setSub(hkVector4, hkSimdReal)
		hkVector4 x; x.set(5,2,1,3);
		hkSimdReal y = hkSimdReal_1;
		hkVector4 z; z.setSub( x, y );

		hkVector4 yVec; yVec.setAll(y);
		hkVector4 z2; z2.setSub(x, yVec);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(z2,eps));
	}


	// * vec
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(9,8,7,6);
		x.mul( y );
		HK_TEST(x(0)==45);
		HK_TEST(x(1)==16);
		HK_TEST(x(2)== 7);
		HK_TEST(x(3)==18);
	}
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(9,8,7,6);
		x.setMul( x,y );
		HK_TEST(x(0)==45);
		HK_TEST(x(1)==16);
		HK_TEST(x(2)== 7);
		HK_TEST(x(3)==18);
	}


	// * real
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		hkSimdReal r; r.setFromFloat(.5f);
		x.mul( r );
		HK_TEST( hkMath::equal( 2.0f*x.length<4>().getReal(), y.length<4>().getReal() ) );
		hkVector4 z; z.setAdd(x,x);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(y,eps) );
	}
	{
		hkVector4 x;
		hkVector4 y; y.set(5,2,1,3);
		hkSimdReal r; r.setFromFloat(.5f);
		x.setMul( r, y );
		HK_TEST( hkMath::equal( 2.0f*x.length<4>().getReal(), y.length<4>().getReal() ) );
	}
	{
		hkVector4 x;
		hkVector4 y; y.set(5,2,1,3);
		hkSimdReal r; r.setFromFloat(.5f);
		x.setMul( y, r );
		HK_TEST( hkMath::equal( 2.0f*x.length<4>().getReal(), y.length<4>().getReal() ) );
	}

	// component-wise division
	{
		hkVector4 x; x.set( 5.f, 2.f, 1.f, 3.f );
		hkVector4 y; y.set( 7.f, 3.f, 2.f, 1.f );
		x.div<HK_ACC_FULL,HK_DIV_SET_ZERO>( y );
		hkVector4 a; a.set( 5.f / 7.f, 2.f / 3.f, 1.f / 2.f, 3.f / 1.f );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<4>(a,eps) );
	}

	{
		hkVector4 x; x.set( 5.f, 2.f, 1.f, 3.f );
		hkVector4 y; y.set( 7.f, 3.f, 2.f, 1.f );
		hkVector4 a;
		a.setDiv<HK_ACC_FULL,HK_DIV_SET_ZERO>( x, y );
		hkVector4 b; b.set( 5.f / 7.f, 2.f / 3.f, 1.f / 2.f, 3.f / 1.f );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( a.allEqual<4>(b,eps) );
	}

	// div_12BitAccurate
	{
		hkVector4 x; x.set( 5.f, 2.f, 1.f, 3.f );
		hkVector4 y; y.set( 7.f, 3.f, 2.f, 1.f );
		x.div<HK_ACC_12_BIT,HK_DIV_SET_ZERO>( y );
		hkVector4 a; a.set( 5.f / 7.f, 2.f / 3.f, 1.f / 2.f, 3.f / 1.f );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<4>(a,eps) );
		checkEqual12Bit<4>(x, a);
	}

	{
		hkVector4 x; x.set( 5.f, 2.f, 1.f, 3.f );
		hkVector4 y; y.set( 7.f, 3.f, 2.f, 1.f );
		hkVector4 a;
		a.setDiv<HK_ACC_12_BIT,HK_DIV_SET_ZERO>( x, y );
		hkVector4 b; b.set( 5.f / 7.f, 2.f / 3.f, 1.f / 2.f, 3.f / 1.f );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( a.allEqual<4>(b,eps) );
		checkEqual12Bit<4>(a, b);
	}

	// div_23BitAccurate
	{
		hkVector4 x; x.set( 5.f, 2.f, 1.f, 3.f );
		hkVector4 y; y.set( 7.f, 3.f, 2.f, 1.f );
		x.div<HK_ACC_23_BIT,HK_DIV_SET_ZERO>( y );
		hkVector4 a; a.set( 5.f / 7.f, 2.f / 3.f, 1.f / 2.f, 3.f / 1.f );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( x.allEqual<4>(a,eps) );
		checkEqual23Bit<4>(x, a);
	}

	{
		hkVector4 x; x.set( 5.f, 2.f, 1.f, 3.f );
		hkVector4 y; y.set( 7.f, 3.f, 2.f, 1.f );
		hkVector4 a;
		a.setDiv<HK_ACC_23_BIT,HK_DIV_SET_ZERO>( x, y );
		hkVector4 b; b.set( 5.f / 7.f, 2.f / 3.f, 1.f / 2.f, 3.f / 1.f );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( a.allEqual<4>(b,eps) );
		checkEqual23Bit<4>(a, b);
	}

	// addMul
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		hkSimdReal r; r.setFromFloat(3.0f);
		x.addMul( r, y );
		HK_TEST( hkMath::equal( x.length<4>().getReal(), 4.0f * y.length<4>().getReal() ) );
	}

	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(7,3,2,1);
		hkVector4 z; z.set(1,4,2,6);
		x.addMul( y, z );
		hkVector4 a; a.set(5+7*1, 2+3*4, 1+2*2, 3+1*6);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( a.allEqual<4>(x,eps) );
	}

	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkSimdReal s; s.setFromFloat( a(1) );
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c = b;
		c.addMul(s, a);
		HK_TEST(c(0)==b(0)+a(1)*a(0));
		HK_TEST(c(1)==b(1)+a(1)*a(1));
		HK_TEST(c(2)==b(2)+a(1)*a(2));
		HK_TEST(c(3)==b(3)+a(1)*a(3));
	}

	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkSimdReal s; s.setFromFloat( a(1) );
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c = b;
		c.addMul(a, s);
		HK_TEST(c(0)==b(0)+a(1)*a(0));
		HK_TEST(c(1)==b(1)+a(1)*a(1));
		HK_TEST(c(2)==b(2)+a(1)*a(2));
		HK_TEST(c(3)==b(3)+a(1)*a(3));
	}

	// setAddMul
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(1,2,3,4);
		hkSimdReal r; r.setFromFloat(3.0f);
		hkVector4 z; z.setAddMul( x, y, r );
		hkVector4 a; a.set(8,8,10,15);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(a,eps) );
	}
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(1,2,3,4);
		hkVector4 z; z.set(10,20,30,40);
		hkVector4 w; w.setAddMul( x, y, z );
		hkVector4 a; a.set(5+1*10, 2+2*20, 1+3*30, 3+4*40);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( w.allEqual<4>(a,eps) );
	}
	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkSimdReal s; s.setFromFloat( a(1) );
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c = b;
		c.setAddMul(b, a, s);
		hkVector4 z; z.set(-100 -1*2, 555 + 2*2, 0 -6*2, 1e5f + 9*2);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(c,eps) );
	}

	// subMul
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y = x;
		hkSimdReal r; r.setFromFloat(3.0f);
		x.subMul( r, y );
		HK_TEST( hkMath::equal( x.length<4>().getReal(), 2.0f * y.length<4>().getReal() ) );
	}
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(7,3,2,1);
		hkVector4 z; z.set(1,4,2,6);
		x.subMul( y, z );
		hkVector4 a; a.set(5-7*1, 2-3*4, 1-2*2, 3-1*6);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( a.allEqual<4>(x,eps) );
	}
	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkSimdReal s; s.setFromFloat( a(1) );
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c = b;
		c.subMul(s, a);
		HK_TEST(c(0)==b(0)-a(1)*a(0));
		HK_TEST(c(1)==b(1)-a(1)*a(1));
		HK_TEST(c(2)==b(2)-a(1)*a(2));
		HK_TEST(c(3)==b(3)-a(1)*a(3));
	}
	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkSimdReal s; s.setFromFloat( a(1) );
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c = b;
		c.subMul(a, s);
		HK_TEST(c(0)==b(0)-a(1)*a(0));
		HK_TEST(c(1)==b(1)-a(1)*a(1));
		HK_TEST(c(2)==b(2)-a(1)*a(2));
		HK_TEST(c(3)==b(3)-a(1)*a(3));
	}

	// setSubMul
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 v1; random.getRandomVector11( v1 );
		hkVector4 v2; random.getRandomVector11( v2 );

		hkVector4 a; a.setSubMul( v0, v1, v2 );
		hkVector4 m; m.setMul( v1,v2);
		hkVector4 b; b.setSub( v0, m );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( a.allEqual<4>(b,eps));
	}
	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkSimdReal s; s.setFromFloat( a(1) );
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c = b;
		c.setSubMul(b, a, s);
		hkVector4 z; z.set(-100 +1*2, 555 - 2*2, 0 +6*2, 1e5f - 9*2);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(c,eps) );
	}

	// horizontal add.
	{
		hkVector4 x; x.set( 1,5,9,21 );
		HK_TEST( hkMath::equal( x.horizontalAdd<3>().getReal(), hkReal(1+5+9)) );
		HK_TEST( hkMath::equal( x.horizontalAdd<4>().getReal(), hkReal(1+5+9+21)) );
		HK_TEST( hkMath::equal( x.horizontalAdd<2>().getReal(), hkReal(1+5)) );
	}

	// set horizontal add.
	{
		hkVector4 x; x.set( 1,5,9,21 );
		hkVector4 y; y.setHorizontalAdd<3>(x);
		hkVector4 w; w.setHorizontalAdd<4>(x);
		hkVector4 z; z.setHorizontalAdd<2>(x);
		HK_TEST( hkMath::equal( y(0), hkReal(1+5+9)) );
		HK_TEST( hkMath::equal( w(0), hkReal(1+5+9+21)) );
		HK_TEST( hkMath::equal( z(0), hkReal(1+5)) );
	}


	// cross
	{
		hkVector4 x; x.set(1,0,0);
		hkVector4 y; y.set(0,1,0);
		hkVector4 z;
		z.setCross( x, y );

		HK_TEST( hkMath::equal( z(0), 0.0f ) );
		HK_TEST( hkMath::equal( z(1), 0.0f ) );
		HK_TEST( hkMath::equal( z(2), 1.0f ) );
	}
	{
		hkVector4 y; y.set(0,1,0);
		hkVector4 z; z.set(0,0,1);
		hkVector4 x;
		x.setCross( y, z );

		HK_TEST( hkMath::equal( x(0), 1.0f ) );
		HK_TEST( hkMath::equal( x(1), 0.0f ) );
		HK_TEST( hkMath::equal( x(2), 0.0f ) );
	}

	// Normalize and Cross
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(-2,6,9,7);
		x.normalize<3>();
		y.normalize<3>();

		hkVector4 z;
		z.setCross( x, y );

		HK_TEST( hkMath::equal( z.dot<3>(x).getReal(), 0.0f ) );
		HK_TEST( hkMath::equal( z.dot<3>(y).getReal(), 0.0f ) );

		// linearly dependent vectors -> cross = 0

		{
			y.setNeg<3>(x);
			z.setCross( x, y );
			HK_TEST( hkMath::equal( z(0), 0.0f ) );
			HK_TEST( hkMath::equal( z(1), 0.0f ) );
			HK_TEST( hkMath::equal( z(2), 0.0f ) );
		}

		{
			hkSimdReal ofive; ofive.setFromFloat(1.5f);
			y.setMul(ofive, x);
			z.setCross( x, y );
			HK_TEST( hkMath::equal( z(0), 0.0f ) );
			HK_TEST( hkMath::equal( z(1), 0.0f ) );
			HK_TEST( hkMath::equal( z(2), 0.0f ) );
		}

		// either vector is zero -> cross = 0
		{
			y.setZero();
			z.setCross( x, y );
			HK_TEST( hkMath::equal( z(0), 0.0f ) );
			HK_TEST( hkMath::equal( z(1), 0.0f ) );
			HK_TEST( hkMath::equal( z(2), 0.0f ) );
		}

		/*
		hkReal me1 = x.length<3>();
		hkReal me2 = y.length<3>();
		hkReal me21 = x.dot<3>(y);
		hkReal me22 = 1.0f - hkReal(x.dot<3>(y)*x.dot<3>(y));
		hkReal me222 = 1.0f - hkReal(me21*me21);
		hkReal me3 = hkMath::sqrt(me22);
		hkReal me4 = me1*me2*me3;
		*/

		hkReal area = ( (x.length<3>() * y.length<3>()) * (hkSimdReal::getConstant(HK_QUADREAL_1) - (x.dot<3>(y)*x.dot<3>(y))) ).getReal();
		hkReal zlen = z.length<3>().getReal();
		//DOUT(area); DOUT(zlen);
		HK_TEST( hkMath::equal(area, zlen ) );
	}
}


namespace vector_assign_special_structs
{
	template<int M>
	struct TestTemplatedSetSelect
	{
		static void test()
		{
			hkVector4 x; x.set(1,2,3,4);
			hkVector4 y; y.set(5,6,7,8);

			hkVector4 z; z.setSelect<(hkVector4ComparisonMask::Mask) M>(x, y);
			HK_TEST(M & hkVector4ComparisonMask::MASK_X ? z.getComponent<0>().isEqual(x.getComponent<0>()) : z.getComponent<0>().isEqual(y.getComponent<0>()));
			HK_TEST(M & hkVector4ComparisonMask::MASK_Y ? z.getComponent<1>().isEqual(x.getComponent<1>()) : z.getComponent<1>().isEqual(y.getComponent<1>()));
			HK_TEST(M & hkVector4ComparisonMask::MASK_Z ? z.getComponent<2>().isEqual(x.getComponent<2>()) : z.getComponent<2>().isEqual(y.getComponent<2>()));
			HK_TEST(M & hkVector4ComparisonMask::MASK_W ? z.getComponent<3>().isEqual(x.getComponent<3>()) : z.getComponent<3>().isEqual(y.getComponent<3>()));
		}

		static void iterTest()
		{
			TestTemplatedSetSelect<M>::test();
			TestTemplatedSetSelect<M+1>::iterTest();
		}
	};

	template<>
	void TestTemplatedSetSelect<hkVector4ComparisonMask::MASK_XYZW>::iterTest()
	{
		TestTemplatedSetSelect<hkVector4ComparisonMask::MASK_XYZW>::test();
	}
}

static void vector_assign_special()
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
#if defined(HK_REAL_IS_DOUBLE)
	static HK_ALIGN_REAL( const hkUint64 mZ[2] ) = { 0x8000000000000000ull, 0x8000000000000000ull };
#else
	static HK_ALIGN_REAL( const hkUint32 mZ[4] ) = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
#endif
#else
#if defined(HK_REAL_IS_DOUBLE)
	const hkUint64 mZ = 0x8000000000000000ull;
#else
	const hkUint32 mZ = 0x80000000;
#endif
#endif
	hkSimdReal minusZero; minusZero.m_real = *(hkSingleReal*)&mZ;


	// setNeg 1
	{
		hkVector4 x; x.set(1, -2, 3, -4);
		hkVector4 negX;
		negX.setNeg<1>(x);

		// Check last components
		HK_TEST(x(1) == negX(1));
		HK_TEST(x(2) == negX(2));
		HK_TEST(x(3) == negX(3));

		// Check negated component
		hkReal a = x(0) - negX(0);
		hkReal b = x(0) + x(0);
		HK_TEST(hkMath::equal(a, b, 1e-3f));
		hkReal c = x(0) + negX(0);
		HK_TEST(hkMath::equal(c, 0.0f));
	}

	// setNeg 2
	{
		hkVector4 x; x.set(-4, 3, -2, 1);
		hkVector4 y;
		y.setNeg<2>(x);

		hkVector4 a; a.setSub(x,y);
		hkVector4 b; b.setAdd(x,x);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( b.allEqual<2>(a,eps) );
		hkVector4 c; c.setAdd(x,y);
		HK_TEST( hkMath::equal( c.length<2>().getReal(), 0.0f ) );
	}

	// setNeg 3
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y;
		y.setNeg<3>(x);

		hkVector4 a; a.setSub(x,y);
		hkVector4 b; b.setAdd(x,x);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( b.allEqual<3>(a,eps) );
		hkVector4 c; c.setAdd(x,y);
		HK_TEST( hkMath::equal( c.length<3>().getReal(), 0.0f ) );
	}

	// setNeg 4
	{
		hkVector4 x; x.set(-1,2,6,-9);
		hkVector4 y;
		y.setNeg<4>(x);
		hkVector4 z;
		z.setMul(hkSimdReal::getConstant(HK_QUADREAL_MINUS1), x);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(y,eps) );
		HK_TEST( ! x.allEqual<4>(y,eps) );
	}

	// setFlipSign.
	{
		hkVector4 v;
		v.set(4.0f, 1.0f, 2.0f, 3.0f);
		hkVector4 val;
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		{
			hkVector4Comparison signs;
			signs.set(hkVector4ComparisonMask::MASK_YW);
			v.setFlipSign(v, signs);
			val.set(4.0f, -1.0f, 2.0f, -3.0f);
			HK_TEST(val.allEqual<4>(v,eps));
		}
		{
			hkVector4 vecSigns;
			vecSigns.set(1.0f, -4.0f, 0.0f, -0.0f); vecSigns.setComponent<3>(minusZero);
			v.setFlipSign(v, vecSigns);
			val.set(4.0f, 1.0f, 2.0f, 3.0f);
			HK_TEST(val.allEqual<4>(v,eps));
		}

		{
			hkSimdReal k; k.setFromFloat(-2.0f);
			v.setFlipSign(v, k);
			val.set(-4.0f, -1.0f, -2.0f, -3.0f);
			HK_TEST(val.allEqual<4>(v,eps));
		}

		{
			hkSimdReal k; k.setFromFloat(2.0f);
			v.setFlipSign(v, k);
			val.set(-4.0f, -1.0f, -2.0f, -3.0f);
			HK_TEST(val.allEqual<4>(v,eps));
		}
	}

	// Testing functionality of setNeg4If()
	//issue ---> function not defined
	/*{
		hkVector4 x;
		x.set(2,3,4,5);
		x.setNeg4If(1);
		HK_TEST( x(0) == -2);
		HK_TEST( x(1) == -3);
		HK_TEST( x(2) == -4);
		HK_TEST( x(3) == -5);
	}*/

	// setAbs
	{
		hkVector4 x; x.set(-1,2,-6,9);
		hkVector4 y;
		y.setAbs(x);
		HK_TEST(y(0)==1);
		HK_TEST(y(1)==2);
		HK_TEST(y(2)==6);
		HK_TEST(y(3)==9);

		x.setAll(minusZero);
		hkVector4 z; z.set(0.0f, 0.0f, 0.0f, 0.0f );
		y.setAbs(x);
		HK_TEST(z.equal(y).allAreSet());
	}

	// setMin
	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c;
		c.setMin(a,b);
		HK_TEST(c(0)==-100);
		HK_TEST(c(1)==2);
		HK_TEST(c(2)==-6);
		HK_TEST(c(3)==9);

		a.setAll(minusZero);
		b.set(0.0f, 0.0f, 0.0f, 0.0f );
		c.setMin(a, b);
		HK_TEST(b.equal(c).allAreSet());

		c.setMin(a, a);
		HK_TEST(b.equal(c).allAreSet());
	}

	// setMax
	{
		hkVector4 a; a.set(-1,2,-6,9);
		hkVector4 b; b.set(-100, 555, 0, 1e5f);
		hkVector4 c;
		c.setMax(a,b);
		HK_TEST(c(0)==-1);
		HK_TEST(c(1)==555);
		HK_TEST(c(2)==0);
		HK_TEST(c(3)==1e5f);

		a.setAll(minusZero);
		b.set(0.0f, 0.0f, 0.0f, 0.0f );
		c.setMax(a, b);
		HK_TEST(b.equal(c).allAreSet());

		c.setMax(a, a);
		HK_TEST(b.equal(c).allAreSet());
	}

	// zeroComponent, templated.
    {
    	hkVector4 a; a.set( 1,2,3,4 );
    	a.zeroComponent<0>();
    	HK_TEST( a(0) == 0 && a(1) == 2 && a(2) ==3 && a(3) == 4 );
    	a.zeroComponent<1>();
    	HK_TEST( a(0) == 0 && a(1) == 0 && a(2) ==3 && a(3) == 4 );
    	a.zeroComponent<2>();
    	HK_TEST( a(0) == 0 && a(1) == 0 && a(2) ==0 && a(3) == 4 );
    	a.zeroComponent<3>();
    	HK_TEST( a(0) == 0 && a(1) == 0 && a(2) ==0 && a(3) == 0 );
    }

	// zeroComponent, non-template
	{
		hkVector4 a; a.set( 1,2,3,4 );
		a.zeroComponent(0);
		HK_TEST( a(0) == 0 && a(1) == 2 && a(2) ==3 && a(3) == 4 );
		a.zeroComponent(1);
		HK_TEST( a(0) == 0 && a(1) == 0 && a(2) ==3 && a(3) == 4 );
		a.zeroComponent(2);
		HK_TEST( a(0) == 0 && a(1) == 0 && a(2) ==0 && a(3) == 4 );
		a.zeroComponent(3);
		HK_TEST( a(0) == 0 && a(1) == 0 && a(2) ==0 && a(3) == 0 );

		HK_TEST_ASSERT(0x3bc36625, a.zeroComponent(-1));
		HK_TEST_ASSERT(0x3bc36625, a.zeroComponent(4));
		HK_TEST_ASSERT(0x3bc36625, a.zeroComponent(5));
	}

//#if !defined(HK_COMPILER_MWERKS)
	
	// Interpolate
	{
		hkVector4 x; x.set(5,2,1,3);
		hkVector4 y; y.set(-2,6,9,7);
		hkVector4 z;

		z.setInterpolate( x, y, hkSimdReal::getConstant(HK_QUADREAL_0));
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( z.allEqual<4>(x,eps) );

		z.setInterpolate( x, y, hkSimdReal::getConstant(HK_QUADREAL_1));
		HK_TEST( z.allEqual<4>(y,eps) );

		hkSimdReal o3; o3.setFromFloat(0.3f);
		hkSimdReal o7; o7.setFromFloat(0.7f);
		z.setInterpolate( x, y, o3);
		hkVector4 w;
		w.setMul( o3, y);
		w.addMul( o7, x);
		HK_TEST( z.allEqual<4>(w,eps) );
	}
//#endif

	// setPermutation
	// identity
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XYZW>( x );
		HK_TEST( x(0) == 1 && x(1) == 2 && x(2) == 3 && x(3) == 4 );
	}
	// reverse
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WZYX>( x );
		HK_TEST( x(0) == 4 && x(1) == 3 && x(2) == 2 && x(3) == 1 );
	}

	// shift
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WXYZ>( x );
		HK_TEST( x(0) == 4 && x(1) == 1 && x(2) == 2 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZWXY>( x );
		HK_TEST( x(0) == 3 && x(1) == 4 && x(2) == 1 && x(3) == 2 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YZWX>( x );
		HK_TEST( x(0) == 2 && x(1) == 3 && x(2) == 4 && x(3) == 1 );
	}

	// swap
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XYWZ>( x );
		HK_TEST( x(0) == 1 && x(1) == 2 && x(2) == 4 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YXWZ>( x );
		HK_TEST( x(0) == 2 && x(1) == 1 && x(2) == 4 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YXZW>( x );
		HK_TEST( x(0) == 2 && x(1) == 1 && x(2) == 3 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YZXW>( x );
		HK_TEST( x(0) == 2 && x(1) == 3 && x(2) == 1 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YWZX>( x );
		HK_TEST( x(0) == 2 && x(1) == 4 && x(2) == 3 && x(3) == 1 );
	}	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WYZX>( x );
		HK_TEST( x(0) == 4 && x(1) == 2 && x(2) == 3 && x(3) == 1 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XZYW>( x );
		HK_TEST( x(0) == 1 && x(1) == 3 && x(2) == 2 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZXYW>( x );
		HK_TEST( x(0) == 3 && x(1) == 1 && x(2) == 2 && x(3) == 4 );
	}

	// pairs
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XYXY>( x );
		HK_TEST( x(0) == 1 && x(1) == 2 && x(2) == 1 && x(3) == 2 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XXYY>( x );
		HK_TEST( x(0) == 1 && x(1) == 1 && x(2) == 2 && x(3) == 2 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZWZW>( x );
		HK_TEST( x(0) == 3 && x(1) == 4 && x(2) == 3 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZYXZ>( x );
		HK_TEST( x(0) == 3 && x(1) == 2 && x(2) == 1 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZYZZ>( x );
		HK_TEST( x(0) == 3 && x(1) == 2 && x(2) == 3 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XZXZ>( x );
		HK_TEST( x(0) == 1 && x(1) == 3 && x(2) == 1 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XZYZ>( x );
		HK_TEST( x(0) == 1 && x(1) == 3 && x(2) == 2 && x(3) == 3 );
	}	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YWYW>( x );
		HK_TEST( x(0) == 2 && x(1) == 4 && x(2) == 2 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YXWW>( x );
		HK_TEST( x(0) == 2 && x(1) == 1 && x(2) == 4 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YXYX>( x );
		HK_TEST( x(0) == 2 && x(1) == 1 && x(2) == 2 && x(3) == 1 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XYWW>( x );
		HK_TEST( x(0) == 1 && x(1) == 2 && x(2) == 4 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XYZZ>( x );
		HK_TEST( x(0) == 1 && x(1) == 2 && x(2) == 3 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XWYW>( x );
		HK_TEST( x(0) == 1 && x(1) == 4 && x(2) == 2 && x(3) == 4 );
	}	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WZWZ>( x );
		HK_TEST( x(0) == 4 && x(1) == 3 && x(2) == 4 && x(3) == 3 );
	}

	// broadcasts
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XXXX>( x );
		HK_TEST( x(0) == 1 && x(1) == 1 && x(2) == 1 && x(3) == 1 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YYYY>( x );
		HK_TEST( x(0) == 2 && x(1) == 2 && x(2) == 2 && x(3) == 2 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZZZZ>( x );
		HK_TEST( x(0) == 3 && x(1) == 3 && x(2) == 3 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WWWW>( x );
		HK_TEST( x(0) == 4 && x(1) == 4 && x(2) == 4 && x(3) == 4 );
	}

	// further permutations.
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XWWW>( x );
		HK_TEST( x(0) == 1 && x(1) == 4 && x(2) == 4 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WXWW>( x );
		HK_TEST( x(0) == 4 && x(1) == 1 && x(2) == 4 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WWXW>( x );
		HK_TEST( x(0) == 4 && x(1) == 4 && x(2) == 1 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::XXZZ>( x );
		HK_TEST( x(0) == 1 && x(1) == 1 && x(2) == 3 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YYWW>( x );
		HK_TEST( x(0) == 2 && x(1) == 2 && x(2) == 4 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WYWW>( x );
		HK_TEST( x(0) == 4 && x(1) == 2 && x(2) == 4 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WWYW>( x );
		HK_TEST( x(0) == 4 && x(1) == 4 && x(2) == 2 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::WWZW>( x );
		HK_TEST( x(0) == 4 && x(1) == 4 && x(2) == 3 && x(3) == 4 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZWWW>( x );
		HK_TEST( x(0) == 3 && x(1) == 4 && x(2) == 4 && x(3) == 4 );
	}

	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::ZXYZ>( x );
		HK_TEST( x(0) == 3 && x(1) == 1 && x(2) == 2 && x(3) == 3 );
	}
	{
		hkVector4 x; x.set(1,2,3,4);
		x.setPermutation<hkVectorPermutation::YZXZ>( x );
		HK_TEST( x(0) == 2 && x(1) == 3 && x(2) == 1 && x(3) == 3 );
	}


	// Testing the functionality of select()
	{
		hkVector4Comparison comp;

		hkVector4 x;
		hkVector4 y;
		hkVector4 z;
		// Testing for equal values
		{
			x.set(1,2,3,4);
			y.set(1,2,3,4);
			comp = x.equal(y);
			z.setSelect(comp,y,x);
			HK_TEST( hkMath::equal(z(0),y(0)) );
			HK_TEST( hkMath::equal(z(1),y(1)) );
			HK_TEST( hkMath::equal(z(2),y(2)) );
			HK_TEST( hkMath::equal(z(3),y(3)) );
		}
		// Testing for unequal values
		{
			x.set(5,6,7,8);
			y.set(1,2,3,4);
			comp = x.equal(y);
			z.setSelect(comp,y,x);
			HK_TEST( hkMath::equal(z(0),x(0)) );
			HK_TEST( hkMath::equal(z(1),x(1)) );
			HK_TEST( hkMath::equal(z(2),x(2)) );
			HK_TEST( hkMath::equal(z(3),x(3)) );
		}
	}

	// Testing the functionality of templated select()
	{
		vector_assign_special_structs::TestTemplatedSetSelect<hkVector4ComparisonMask::MASK_NONE>::iterTest();
	}

	// Testing functionality of setReciprocal()
	{
		const int NUM_TIMES = 100;
		{
			hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
			hkVector4 x;
			hkVector4 y;
			hkVector4 out;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
				y.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(x);
				out.setMul(x,y);
				HK_TEST( hkMath::equal(out(0),hkReal(1)) );
				HK_TEST( hkMath::equal(out(1),hkReal(1)) );
				HK_TEST( hkMath::equal(out(2),hkReal(1)) );
				HK_TEST( hkMath::equal(out(3),hkReal(1)) );
			}
		}
	}

	// Testing functionality of setReciprocal_23BitAccurate()
	{
		const int NUM_TIMES = 100;
		{
			hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
			hkVector4 x;
			hkVector4 y;
			hkVector4 out;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),1.0f);
				y.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(x);
				out.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(x);
				checkEqual23Bit<3>(y, out);
			}
		}
	}

	// Testing functionality of setReciprocal_12BitAccurate()
	{
		const int NUM_TIMES = 100;
		{
			hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
			hkVector4 x;
			hkVector4 y;
			hkVector4 out;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),1.0f);
				y.setReciprocal<HK_ACC_12_BIT,HK_DIV_SET_ZERO>(x);
				out.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(x);
				checkEqual12Bit<3>(y, out);
			}
		}
	}

	// Check sign preserving
	{
		hkVector4 a; a.set(1,-1,0,0);
		hkVector4 b; b.setZero();
		hkVector4 c; c.setDiv<HK_ACC_12_BIT,HK_DIV_SET_MAX>(a,b);
		HK_TEST(c(0) > 0);
		HK_TEST(c(1) < 0);
		HK_TEST(c(2) > 0);
	}
	{
		hkVector4 a; a.set(1,-1,0,0);
		hkVector4 b; b.setZero();
		hkVector4 c; c.setDiv<HK_ACC_23_BIT,HK_DIV_SET_HIGH>(a,b);
		HK_TEST(c(0) > 0);
		HK_TEST(c(1) < 0);
		HK_TEST(c(2) > 0);
	}
	{
		hkVector4 a; a.set(1,-1,0,0);
		hkVector4 b; b.setZero();
		hkVector4 c; c.setDiv<HK_ACC_FULL,HK_DIV_SET_ZERO>(a,b);
		HK_TEST(c(0) == 0);
		HK_TEST(c(1) == 0);
		HK_TEST(c(2) == 0);
	}

	// setHorizontalMax
	{
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		hkPseudoRandomGenerator random(10);
		for (int i =0; i < 100; i++)
		{
			hkVector4 v0; random.getRandomVector11( v0 );
			hkVector4 v1; random.getRandomVector11( v1 );
			hkVector4 v2; random.getRandomVector11( v2 );

			// setHorizontalMax, 4 components
			{
				hkVector4 hm; hm.setHorizontalMax<4>( v0 );

				hkReal a = hkMath::max2( v0(0), v0(1) );
				hkReal b = hkMath::max2( v0(2), v0(3) );
				a = hkMath::max2( a, b );
				hkVector4 h; h.setAll( a );
				HK_TEST( h.allEqual<4>(hm,eps));
			}
		}

		// setHorizontalMax, 3 components
		{
			hkVector4 v0; v0.set(1, 2, 3, 4);
			hkVector4 hm; hm.setHorizontalMax<3>( v0 );

			hkReal a = hkMath::max2( v0(0), v0(1) );
			a = hkMath::max2( a, v0(2) );
			hkVector4 h; h.setAll( a );
			HK_TEST( h.allEqual<3>(hm,eps));
		}

		// setHorizontalMax, 2 components
		{
			hkVector4 v0; v0.set(1, 2, 3, 4);
			hkVector4 hm; hm.setHorizontalMax<2>( v0 );

			hkReal a = hkMath::max2( v0(0), v0(1) );
			hkVector4 h; h.setAll( a );
			HK_TEST( h.allEqual<2>(hm,eps));
		}
	}

	// setHorizontalMin
	{
		hkPseudoRandomGenerator random(10);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		for (int i =0; i < 100; i++)
		{
			hkVector4 v0; random.getRandomVector11( v0 );
			hkVector4 v1; random.getRandomVector11( v1 );
			hkVector4 v2; random.getRandomVector11( v2 );

			// setHorizontalMin, 4 components
			{
				hkVector4 hm; hm.setHorizontalMin<4>( v0 );

				hkReal a = hkMath::min2( v0(0), v0(1) );
				hkReal b = hkMath::min2( v0(2), v0(3) );
				a = hkMath::min2( a, b );
				hkVector4 h; h.setAll( a );
				HK_TEST( h.allEqual<4>(hm,eps));
			}
		}

		// setHorizontalMin, 3 components
		{
			hkVector4 v0; v0.set(1, 2, 3, 4);
			hkVector4 hm; hm.setHorizontalMin<3>( v0 );

			hkReal a = hkMath::min2( v0(0), v0(1) );
			a = hkMath::min2( a, v0(2) );
			hkVector4 h; h.setAll( a );
			HK_TEST( h.allEqual<3>(hm,eps));
		}

		// setHorizontalMin, 2 components
		{
			hkVector4 v0; v0.set(1, 2, 3, 4);
			hkVector4 hm; hm.setHorizontalMin<2>( v0 );

			hkReal a = hkMath::min2( v0(0), v0(1) );
			hkVector4 h; h.setAll( a );
			HK_TEST( h.allEqual<2>(hm,eps));
		}
	}

	// setClampedToMaxLength
	{
		hkVector4 v;
		hkVector4 val;
		val.set(4.0f, 1.0f, 2.0f, 3.0f);
		v.setClampedToMaxLength(val, hkSimdReal::getConstant(HK_QUADREAL_1));
		val.normalize<3>();
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(v.allEqual<3>(val,eps));
	}
	{
		hkVector4 v;
		hkVector4 val;
		val.set(4.0f, 1.0f, 2.0f, 3.0f);
		v.setClampedToMaxLength(val, hkSimdReal::getConstant(HK_QUADREAL_5)); // Should not change.
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(v.allEqual<4>(val,eps));
	}

	// constants
	{
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		for (int i = HK_QUADREAL_MINUS1; i < HK_QUADREAL_END; i++)
		{
			hkVector4 x; x = hkVector4::getConstant(hkVectorConstant(i));
			HK_TEST(x.allEqual<4>(hkVector4::getConstant(hkVectorConstant(i)), eps));
		}

		// compile time
		hkVector4 x = hkVector4::getConstant<HK_QUADREAL_5>();
		HK_TEST(x.allEqual<4>(hkVector4::getConstant<HK_QUADREAL_5>(), eps));
		x = hkVector4::getConstant<HK_QUADREAL_INV_6>();
		HK_TEST(x.allEqual<4>(hkVector4::getConstant<HK_QUADREAL_INV_6>(), eps));
	}
}

static void vector_getset_int24w()
{
	hkVector4 x; x.setZero();
	hkVector4 xyz; xyz.set(1,2,3,4);

	for (int i = 0; i < 0xffffff; i = i+1+(i>>16) )
	{
		x.setInt24W(i);
		x.setXYZ(xyz);
#if defined(HK_REAL_IS_DOUBLE)
		const hkUint64* f = (const hkUint64*)&x;
		HK_TEST( (f[3] & 0x00ffffff) == (hkUint64)i );
#else
		const hkUint32* f = (const hkUint32*)&x;
		HK_TEST( (f[3] & 0x00ffffff) == (hkUint32)i );
#endif
		HK_TEST( x.getInt24W() == i);
	}

	for (int i = 0; i < 0xffff; i = i+1+(i>>16) )
	{
		x.setInt24W(i);
		x.setXYZ(xyz);
		HK_TEST( x.getInt16W() == i );
	}
}

static void matrix3_transform_quaternion()
{
	// setRotatedDir & setRotatedInverseDir matrix3
	{
		hkVector4 c0; c0.set(4,1,7);
		hkVector4 c1; c1.set(9,5,2);
		hkVector4 c2; c2.set(8,6,4);
		hkMatrix3 m;
		m.setCols(c0,c1,c2);
		hkVector4 v0; v0.set(1,2,3);

		hkVector4 v1;
		v1.setRotatedDir(m,v0);

		HK_TEST( v1(0)==46 );
		HK_TEST( v1(1)==29 );
		HK_TEST( v1(2)==23 );
		//hkcout << v0 << '\n' << m << '\n' << v1 << '\n';

		//hkVector4 v2;
		//v2.setRotatedInverseDir(m,v1); // inline to see code
		//HK_TEST( v0.allEqual<3>(v2,hkSimdReal(1e-3f)) );
	}

	// setRotatedDir & setRotatedInverseDir hkRotation
	{
		hkRotation r;
		hkVector4 axis; axis.set(5,2,-4);
		axis.normalize<3>();
		r.setAxisAngle( axis, 0.62f);
		hkVector4 v0; v0.set(2,3,4);

		hkVector4 v1;
		v1.setRotatedDir(r,v0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( !v0.allEqual<3>(v1,eps) );

		hkVector4 v2;
		v2.setRotatedInverseDir(r,v1); // inline to see code
		HK_TEST( v0.allEqual<3>(v2,eps) );
		//		hkcout << v0 << '\n' << r << '\n' << v1 << '\n' << v2 << '\n';
	}

	// _setRotatedDir & _setRotatedInverseDir
	{
		hkRotation r;
		hkVector4 axis; axis.set(5,2,-4);
		axis.normalize<3>();
		r.setAxisAngle( axis, 0.62f);
		hkVector4 v0; v0.set(2,3,4);

		hkVector4 v1;
		v1._setRotatedDir(r,v0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( !v0.allEqual<3>(v1,eps) );

		hkVector4 v2;
		v2._setRotatedInverseDir(r,v1); // inline to see code
		HK_TEST( v0.allEqual<3>(v2,eps) );
//		hkcout << v0 << '\n' << r << '\n' << v1 << '\n' << v2 << '\n';
	}

	// setTransformedPos & setTransformedInversePos
	{
		hkVector4 axis; axis.set(5,2,-4);
		axis.normalize<3>();
		hkRotation r;
		r.setAxisAngle(axis, 0.62f);
		hkTransform t;
		t.setRotation(r);
		hkVector4 t0; t0.set(-20,-30,-40);
		t.setTranslation(t0);

		hkVector4 v0; v0.set(2,3,4);
		hkVector4 v1;
		v1.setTransformedPos(t,v0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( !v0.allEqual<3>(v1,eps) );
		hkVector4 v2;
		v2.setTransformedInversePos(t,v1);
		HK_TEST( v0.allEqual<3>(v2,eps) );
		//hkcout << v0 << '\n' << t << '\n' << v1 << '\n' << v2 << '\n';
	}

	// _setTransformedPos & _setTransformedInversePos
	{
		hkVector4 axis; axis.set(5,2,-4);
		axis.normalize<3>();
		hkRotation r;
		r.setAxisAngle(axis, 0.62f);
		hkTransform t;
		t.setRotation(r);
		hkVector4 t0; t0.set(-20,-30,-40);
		t.setTranslation(t0);

		hkVector4 v0; v0.set(2,3,4);
		hkVector4 v1;
		v1._setTransformedPos(t,v0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( !v0.allEqual<3>(v1,eps) );
		hkVector4 v2;
		v2._setTransformedInversePos(t,v1);
		HK_TEST( v0.allEqual<3>(v2,eps) );
		//hkcout << v0 << '\n' << t << '\n' << v1 << '\n' << v2 << '\n';
	}

	// setTransformedPos & setTransformedInversePos with hkQsTransform
	{
		hkVector4 axis; axis.set(5,2,-4);
		axis.normalize<3>();
		hkRotation r;
		r.setAxisAngle(axis, 0.62f);
		hkVector4 t0; t0.set(-20,-30,-40);
		hkQuaternion q;
		q.set( r );
		hkQsTransform t; t.set(t0, q);

		hkVector4 v0; v0.set(2,3,4);
		hkVector4 v1;
		v1.setTransformedPos(t,v0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( !v0.allEqual<3>(v1,eps) );
		hkVector4 v2;
		v2.setTransformedInversePos(t,v1);
		HK_TEST( v0.allEqual<3>(v2,eps) );
		//hkcout << v0 << '\n' << t << '\n' << v1 << '\n' << v2 << '\n';
	}


	// _setTransformedPos & _setTransformedInversePos with hkQsTransform
	{
		hkVector4 axis; axis.set(5,2,-4);
		axis.normalize<3>();
		hkRotation r;
		r.setAxisAngle(axis, 0.62f);
		hkVector4 t0; t0.set(-20,-30,-40);
		hkQuaternion q;
		q.set( r );
		hkQsTransform t; t.set(t0, q);

		hkVector4 v0; v0.set(2,3,4);
		hkVector4 v1;
		v1._setTransformedPos(t,v0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( !v0.allEqual<3>(v1,eps) );
		hkVector4 v2;
		v2._setTransformedInversePos(t,v1);
		HK_TEST( v0.allEqual<3>(v2,eps) );
		//hkcout << v0 << '\n' << t << '\n' << v1 << '\n' << v2 << '\n';
	}

	// _setTransformedPos & _setTransformedInversePos with hkQTransform
	 // not yet implemented.
	//{
	//	hkVector4 axis; axis.set(5,2,-4);
	//	axis.normalize_23BitAccurate<3>();
	//	hkReal angle = 0.62f;
	//	hkQuaternion q( axis, angle);

	//	hkQTransform t;
	//	t.setRotation(q);
	//	hkVector4 t0; t0.set(-20,-30,-40);
	//	t.setTranslation(t0);

	//	hkVector4 v0; v0.set(2,3,4);
	//	hkVector4 v1;
	//	v1._setTransformedPos(t,v0);
	//	HK_TEST( !v0.allEqual<3>(v1,hkSimdReal(1e-3f)) );
	//	hkVector4 v2;
	//	v2._setTransformedInversePos(t,v1);
	//	HK_TEST( v0.allEqual<3>(v2,hkSimdReal(1e-3f)) );
	//	//hkcout << v0 << '\n' << t << '\n' << v1 << '\n' << v2 << '\n';
	//}

	// setRotatedDir with quaternion
	{
		hkVector4 axis; axis.set(5,2,-4);
		hkReal angle = 0.3f;
		axis.normalize<3>();

		hkQuaternion q; q.setAxisAngle( axis, angle);
		hkRotation r;
		r.setAxisAngle(axis, angle);

		hkQuaternion q2;
		q2.set( r );

		//DOUT(q2);

		hkVector4 x; x.set(4,2,6,1);
		hkVector4 yq;
		yq.setRotatedDir(q,x);
		hkVector4 yr;
		yr.setRotatedDir(r,x);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(yq.allEqual<3>(yr,eps));
		//DOUT(q); DOUT(r); DOUT(x); DOUT(yq); DOUT(yr);
	}

	// setRotatedInverseDir with quaternions
	{
		hkVector4 axis; axis.set(5,2,-4);
		hkReal angle = 0.3f;
		axis.normalize<3>();

		hkQuaternion q; q.setAxisAngle( axis, angle);
		hkRotation r;
		r.setAxisAngle(axis, angle);

		hkQuaternion q2;
		q2.set( r );

		//DOUT(q2);

		hkVector4 x; x.set(4,2,6,1);
		hkVector4 yq;
		yq.setRotatedInverseDir(q,x);
		hkVector4 yr;
		yr.setRotatedInverseDir(r,x);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(yq.allEqual<3>(yr,eps));
//		DOUT(q); DOUT(r); DOUT(x); DOUT(yq); DOUT(yr);
	}

	// _setRotatedDir with quaternion
	{
		hkVector4 axis; axis.set(5,2,-4);
		hkReal angle = 0.3f;
		axis.normalize<3>();

		hkQuaternion q; q.setAxisAngle( axis, angle);
		hkRotation r;
		r.setAxisAngle(axis, angle);

		hkQuaternion q2;
		q2.set( r );

		//DOUT(q2);

		hkVector4 x; x.set(4,2,6,1);
		hkVector4 yq;
		yq._setRotatedDir(q,x);
		hkVector4 yr;
		yr._setRotatedDir(r,x);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(yq.allEqual<3>(yr,eps));
		//DOUT(q); DOUT(r); DOUT(x); DOUT(yq); DOUT(yr);
	}


	// _setRotatedInverseDir with quaternions
	{
		hkVector4 axis; axis.set(5,2,-4);
		hkReal angle = 0.3f;
		axis.normalize<3>();

		hkQuaternion q; q.setAxisAngle( axis, angle);
		hkRotation r;
		r.setAxisAngle(axis, angle);

		hkQuaternion q2;
		q2.set( r );

		//DOUT(q2);

		hkVector4 x; x.set(4,2,6,1);
		hkVector4 yq;
		yq._setRotatedInverseDir(q,x);
		hkVector4 yr;
		yr._setRotatedInverseDir(r,x);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(yq.allEqual<3>(yr,eps));
		//		DOUT(q); DOUT(r); DOUT(x); DOUT(yq); DOUT(yr);
	}

	// composition of rotations
	{
		hkVector4 axis; axis.set(5,2,-4);
		hkReal angle = 0.3f;
		axis.normalize<3>();

		hkQuaternion q1; q1.setAxisAngle( axis, angle);
		hkRotation r1;
		r1.setAxisAngle(axis, angle);

		hkVector4 axis2; axis2.set(2,-1,4);
		hkReal angle2 = 0.3f;
		axis2.normalize<3>();

		hkQuaternion q2; q2.setAxisAngle( axis2, angle2);
		hkRotation r2;
		r2.setAxisAngle(axis2, angle2);

		hkQuaternion q12;
		q12.setMul( q1, q2 );

		hkRotation r12;
		r12.setMul( r1, r2 );

		hkVector4 x; x.set( 4,2,6,1 );
		hkVector4 yq;
		yq.setRotatedDir(q12,x);
		hkVector4 yr;
		yr.setRotatedDir(r12,x);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(yq.allEqual<3>(yr,eps));
	}

	{
		hkVector4 c0; c0.set(-4, 1,-7,-2);
		hkVector4 c1; c1.set( 9,-5, 2,-3);
		hkVector4 c2; c2.set( 8, 6, 4,-4);
		hkTransform t;
		t.getRotation().setCols(c0,c1,c2);
		t.getTranslation().set(0,3,-1,-5);

		hkVector4 v0; v0.set(1,2,3,-1000);
		hkVector4 v1; v1._setTransformedPos(t,v0);

		HK_TEST( v1(0)== 38 );
		HK_TEST( v1(1)== 12 );
		HK_TEST( v1(2)==  8 );
		HK_TEST( v1(3)==-25 );
		//hkcout << v0 << '\n' << m << '\n' << v1 << '\n';
	}

	// transpose test
	hkPseudoRandomGenerator random(10);
	for (int i =0; i < 100; i++)
	{
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 v1; random.getRandomVector11( v1 );
		hkVector4 v2; random.getRandomVector11( v2 );

		{
			hkMatrix3 m; m.setCols( v0, v1, v2 );
			m.transpose();
			hkMatrix3 m2; m2.setRows( v0, v1, v2 );
			HK_TEST( m.isApproximatelyEqual(m2) );

			hkVector4 t0 = v0;
			hkVector4 t1 = v1;
			hkVector4 t2 = v2;
			HK_TRANSPOSE3( t0,t1,t2);
			m2.setCols( t0,t1,t2);
			HK_TEST( m.isApproximatelyEqual(m2) );
		}
	}
}

static void vector_dots_lengths()
{
	// dot, setDot - 2 components
	{
		hkVector4 a; a.set(5,2,-4,8);
		hkVector4 b; b.set(9,1,3,11);
		hkReal r = a.dot<2>(b).getReal();
		HK_TEST( hkMath::equal(r, hkReal(45+2)) );

		hkVector4 d; d.setDot<2>(a,b);
		HK_TEST( hkMath::equal(d(0), r) );
	}

	// dot, setDot - 3 components
	{
		hkVector4 a; a.set(5,2,-4,8);
		hkVector4 b; b.set(9,1,3,11);
		hkReal r = a.dot<3>(b).getReal();
		HK_TEST( hkMath::equal(r, hkReal(45+2-12)) );

		hkVector4 d; d.setDot<3>(a,b);
		HK_TEST( hkMath::equal(d(0), r) );
	}

	// dot, setDot - 4 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b; b.set(9,1,3,10);
		hkReal r = a.dot<4>(b).getReal();
		HK_TEST( hkMath::equal(r, hkReal(45+2-12-80)) );

		hkVector4 d; d.setDot<4>(a,b);
		HK_TEST( hkMath::equal(d(0), r) );
		HK_TEST( hkMath::equal(d(1), r) );
		HK_TEST( hkMath::equal(d(2), r) );
		HK_TEST( hkMath::equal(d(3), r) );
	}

	// dot with setPlaneConstant.
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b; b.set(9,1,3,10);
		hkReal w = -a.dot<3>(b).getReal();
		a.setPlaneConstant(b);
		// Make sure XYZ are unchanged and W is updated
		HK_TEST( hkMath::equal(a(0), 5.0f) );
		HK_TEST( hkMath::equal(a(1), 2.0f) );
		HK_TEST( hkMath::equal(a(2), -4.0f) );
		HK_TEST( hkMath::equal(a(3), w) );
	}

	// dot4xyz1
	{
		hkVector4 x; x.set( 1,2,3,4 );
		hkVector4 y; y.set( 3,5,7,11 );

		hkReal result = x.dot4xyz1( y ).getReal();

		HK_TEST( hkMath::equal( 38.0f, result) );
	}

	// length
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b;
		b.setAdd(a,a);

		hkReal r2 = (5.0f*5.0f) + (2.0f*2.0f);
		hkReal r3 = (5.0f*5.0f) + (2.0f*2.0f) + (-4.0f * -4.0f);
		hkReal r4 = (5.0f*5.0f) + (2.0f*2.0f) + (-4.0f * -4.0f) + (-8.0f * -8.0f);

		HK_TEST( hkMath::equal( a.lengthSquared<2>().getReal(), r2 ) );
		HK_TEST( hkMath::equal( a.lengthSquared<3>().getReal(), r3 ) );
		HK_TEST( hkMath::equal( a.lengthSquared<4>().getReal(), r4 ) );

		HK_TEST( hkMath::equal( 2.0f*a.length<2>().getReal(), b.length<2>().getReal() ) );
		HK_TEST( hkMath::equal( 2.0f*a.length<3>().getReal(), b.length<3>().getReal() ) );
		HK_TEST( hkMath::equal( 2.0f*a.length<4>().getReal(), b.length<4>().getReal() ) );

		HK_TEST( hkMath::equal( (a.length<2>()*a.length<2>()).getReal(), a.lengthSquared<2>().getReal(), 1e-4f ) );
		HK_TEST( hkMath::equal( (a.length<3>()*a.length<3>()).getReal(), a.lengthSquared<3>().getReal(), 1e-4f ) );
		HK_TEST( hkMath::equal( (a.length<4>()*a.length<4>()).getReal(), a.lengthSquared<4>().getReal(), 1e-4f ) );

		HK_TEST( hkIsEqual12BitAccurate(a.lengthInverse<2,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<2,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual12BitAccurate(a.lengthInverse<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual12BitAccurate(a.lengthInverse<4,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<4,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal() ) );

		HK_TEST( hkIsEqual12BitAccurate(a.lengthInverse<2,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<2,HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual12BitAccurate(a.lengthInverse<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual12BitAccurate(a.lengthInverse<4,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<4,HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal() ) );

		HK_TEST( hkIsEqual23BitAccurate(a.lengthInverse<2,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<2,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual23BitAccurate(a.lengthInverse<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual23BitAccurate(a.lengthInverse<4,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<4,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal() ) );

		HK_TEST( hkIsEqual23BitAccurate(a.lengthInverse<2,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<2,HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual23BitAccurate(a.lengthInverse<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal() ) );
		HK_TEST( hkIsEqual23BitAccurate(a.lengthInverse<4,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f*b.lengthInverse<4,HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal() ) );

		HK_TEST( hkMath::equal( a.lengthInverse<2>().getReal(), 2.0f / b.length<2>().getReal() ) );
		HK_TEST( hkMath::equal( a.lengthInverse<3>().getReal(), 2.0f / b.length<3>().getReal() ) );
		HK_TEST( hkMath::equal( a.lengthInverse<4>().getReal(), 2.0f / b.length<4>().getReal() ) );
	}


	// normalize - 4 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b = a;
		hkVector4 d = a;
		b.normalize<4,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		HK_TEST( hkMath::equal( b.length<4>().getReal(), 1.0f ) );

		d.normalize<4,HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		HK_TEST( hkMath::equal( d.length<4>().getReal(), 1.0f, 2e-04f ) );

		hkReal alen = a.length<4>().getReal();
		hkVector4 c = a;
		HK_TEST( hkMath::equal( c.normalizeWithLength<4>().getReal(), alen ) );
		HK_TEST( hkMath::equal( c.length<4>().getReal(), 1.0f ) );
	}

	// normalize - 3 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b = a;
		hkVector4 d = a;
		b.normalize<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		HK_TEST( hkMath::equal( b.length<3>().getReal(), 1.0f ) );

		d.normalize<3,HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		HK_TEST( hkMath::equal( d.length<3>().getReal(), 1.0f, 2e-04f ) );

		hkReal alen = a.length<3>().getReal();
		hkVector4 c = a;
		HK_TEST( hkMath::equal( c.normalizeWithLength<3>().getReal(), alen ) );
		HK_TEST( hkMath::equal( c.length<3>().getReal(), 1.0f ) );
	}

	// normalize - 2 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b = a;
		hkVector4 d = a;
		b.normalize<2,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		HK_TEST( hkMath::equal( b.length<2>().getReal(), 1.0f ) );

		d.normalize<2,HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		HK_TEST( hkMath::equal( d.length<2>().getReal(), 1.0f, 2e-04f ) );

		hkReal alen = a.length<2>().getReal();
		hkVector4 c = a;
		HK_TEST( hkMath::equal( c.normalizeWithLength<2>().getReal(), alen ) );
		HK_TEST( hkMath::equal( c.length<2>().getReal(), 1.0f ) );
	}

	// normalize_23BitAccurate - 4 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b = a;
		hkVector4 d = a;
		b.normalize<4,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		HK_TEST( hkMath::equal( b.length<4>().getReal(), 1.0f ) );

		d.normalize<4,HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		HK_TEST( hkMath::equal( d.length<4>().getReal(), 1.0f, 2e-04f ) );

		hkReal alen = a.length<4>().getReal();
		hkVector4 c = a;
		HK_TEST( hkMath::equal( c.normalizeWithLength<4,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), alen ) );
		HK_TEST( hkMath::equal( c.length<4>().getReal(), 1.0f ) );
	}

	// normalize_23BitAccurate - 3 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b = a;
		hkVector4 d = a;
		b.normalize<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		HK_TEST( hkMath::equal( b.length<3>().getReal(), 1.0f ) );

		d.normalize<3,HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		HK_TEST( hkMath::equal( d.length<3>().getReal(), 1.0f, 2e-04f ) );

		hkReal alen = a.length<3>().getReal();
		hkVector4 c = a;
		HK_TEST( hkMath::equal( c.normalizeWithLength<3,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), alen ) );
		HK_TEST( hkMath::equal( c.length<3>().getReal(), 1.0f ) );
	}

	// normalize_12BitAccurate - 3 components
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b = a;
		hkVector4 d = a;
		b.normalize<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
		HK_TEST( hkMath::equal( b.length<3>().getReal(), 1.0f, 2e-04f ) );

		d.normalize<3,HK_ACC_12_BIT,HK_SQRT_IGNORE>();
		HK_TEST( hkMath::equal( d.length<3>().getReal(), 1.0f, 2e-04f ) );

		hkReal alen = a.length<3>().getReal();
		hkVector4 c = a;
		HK_TEST( hkMath::equal( c.normalizeWithLength<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), alen, 11e-04f ) );
		HK_TEST( hkMath::equal( c.length<3>().getReal(), 1.0f, 2e-04f ) );
	}

	//Testing functionality of normalizeIfNotZero()
	{
		hkVector4 x;
		x.set(4,5,-1,-2);
		hkVector4 out(x) ;
		HK_TEST( x.normalizeIfNotZero<3>() );
		hkSimdReal ss; ss.setFromFloat(hkMath::sqrtInverse( out.lengthSquared<3>().getReal() ));
		out.mul(ss);
		HK_TEST( hkMath::equal(x(0),out(0)) );
		HK_TEST( hkMath::equal(x(1),out(1)) );
		HK_TEST( hkMath::equal(x(2),out(2)) );

		x.set(4,5,-1,-2);
		HK_TEST( x.normalizeIfNotZero<4>() );
		ss.setFromFloat(hkMath::sqrtInverse( out.lengthSquared<4>().getReal() ));
		out.mul(ss);
		HK_TEST( hkMath::equal(x(0),out(0)) );
		HK_TEST( hkMath::equal(x(1),out(1)) );
		HK_TEST( hkMath::equal(x(2),out(2)) );
		HK_TEST( hkMath::equal(x(3),out(3)) );

		x.set(4,5,-1,-2);
		HK_TEST( x.normalizeIfNotZero<2>() );
		ss.setFromFloat(hkMath::sqrtInverse( out.lengthSquared<2>().getReal() ));
		out.mul(ss);
		HK_TEST( hkMath::equal(x(0),out(0)) );
		HK_TEST( hkMath::equal(x(1),out(1)) );

		hkVector4 y;
		y.set(0,0,0,1);
		HK_TEST( y.normalizeIfNotZero<3>() == hkFalse32 );
		HK_TEST( y.normalizeIfNotZero<4>() != hkFalse32 );

		y.set(0,0,1,1);
		HK_TEST( y.normalizeIfNotZero<2>() == hkFalse32 );
		HK_TEST( y.normalizeIfNotZero<3>() != hkFalse32 );
	}

	// Testing functionality of isNormalized()
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		const int NUM_TIMES = 100;
		for(int i = 0; i < NUM_TIMES; i++)
		{
			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
			x.normalize<3>();
			HK_TEST(x.isNormalized<3>());
			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
			x.normalize<4>();
			HK_TEST(x.isNormalized<4>());
			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
			x.normalize<2>();
			HK_TEST(x.isNormalized<2>());
		}
	}

	// length
	{
		hkVector4 c; c.set(0,0,0,1);
		hkReal cl3 = c.length<3>().getReal();
		hkReal cl4 = c.length<4>().getReal();
		HK_TEST( hkMath::equal( 0.0f, cl3, 1e-04f ) );
		HK_TEST( hkMath::equal( 1.0f, cl4, 1e-04f ) );

		hkVector4 d; d.set(0,0,0,0);
		hkReal dl3 = d.length<3>().getReal();
		hkReal dl4 = d.length<4>().getReal();
		HK_TEST( hkMath::equal( 0.0f, dl3, 1e-04f ) );
		HK_TEST( hkMath::equal( 0.0f, dl4, 1e-04f ) );
	}

	// Check that zero-length vectors give finite values for lengthInverse_23BitAccurate 3&4 components
	{
		hkVector4 zero; zero.set(0,0,0);

		HK_TEST( hkMath::isFinite(zero.lengthInverse<3>().getReal()) );
		HK_TEST( hkMath::isFinite(zero.lengthInverse<4>().getReal()) );
		HK_TEST( hkMath::isFinite(zero.length<3>().getReal()) );
		HK_TEST( hkMath::isFinite(zero.length<4>().getReal()) );
	}

	// Check distanceTo and distanceToSquared
	{
		hkVector4 a; a.set(5,2,-4,-8);
		hkVector4 b; b.set(5,5,-8, 0); // a + (0, 3, 4)

		HK_TEST( hkMath::equal( a.distanceTo<HK_ACC_FULL,HK_SQRT_SET_ZERO>(b).getReal(), 5.0f) );
		HK_TEST( hkMath::equal( a.distanceToSquared(b).getReal(), 25.0f) );
		HK_TEST( hkIsEqual23BitAccurate( a.distanceTo<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>(b).getReal(), 5.0f) );
		HK_TEST( hkIsEqual12BitAccurate( a.distanceTo<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>(b).getReal(), 5.0f) );
	}
}

#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED
#define ON_SIMD_CPU(x) HK_ON_CPU(x)
#else
#define ON_SIMD_CPU(x)
#endif

static void vector_square_roots()
{
	// Note that several of these checks are disabled on SPU and no-SIMD, since they never generate INF/NAN values.
	// We can check that something IS ok, but not that it's NOT ok.
	hkSimdReal eps; eps.setFromFloat(1e-3f);
	// setSqrt
	{
		hkVector4 a; a.set(4, 9, 16, 0);
		hkVector4 b; b.setSqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>(a);

		b.mul(b);
		HK_TEST( b.allEqual<4>(a, eps));
	}
	{
		hkVector4 a; a.set(-4.0f, 9, 16, 25);
		hkVector4 b; b.setSqrt<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(0)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}
	{
		hkVector4 a; a.set(4.0f, 9, 16, -25);
		hkVector4 b; b.setSqrt<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
		HK_TEST(b.isOk<3>());
	}
	{
		hkVector4 a; a.set(-4.0f, -9, -16, -25);
		hkVector4 b; b.setSqrt<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(0)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(2)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<1>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<2>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<3>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}


	// setSqrtInverse
	{
		hkVector4 a; a.set(4, 9, 16, 25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		hkVector4 c; c.set(1.0f/2.0f, 1.0f/3.0f, 1.0f/4.0f, 1.0f/5.0f);
		HK_TEST( b.allEqual<4>(c,eps) );
	}
	{
		hkVector4 a; a.set(0.0f, -9.0f, 16, 25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		//HK_TEST(hkMath::equal(b(0), 0.0f));  // not working
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}
	{
		hkVector4 a; a.set(4, 9, 16, -25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
		HK_TEST(b.isOk<3>());
	}
	{
		hkVector4 a; a.set(-4, -9, -16, -25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(0)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(2)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<1>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<2>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<3>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}

	// setSqrtInverse_23BitAccurate
	{
		hkVector4 a; a.set(4, 9, 16, 25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(a);

		hkVector4 c; c.setSqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>(a);
		checkEqual23Bit<4>(b, c);
	}
	{
		hkVector4 a; a.set(0.0f, -9.0f, 16, 25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(a);

		//HK_TEST(hkMath::equal(b(0), 0.0f));  // not working
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}
	{
		hkVector4 a; a.set(4, 9, 16, -25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
		HK_TEST(b.isOk<3>());
	}
	{
		hkVector4 a; a.set(-4, -9, -16, -25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(0)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(2)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<1>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<2>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<3>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}

	// setSqrtInverse_12BitAccurate
	{
		hkVector4 a; a.set(4, 9, 16, 25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>(a);

		hkVector4 c; c.setSqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>(a);
		checkEqual12Bit<4>(b, c);
	}
	{
		hkVector4 a; a.set(0.0f, -9.0f, 16, 25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>(a);

		//HK_TEST(hkMath::equal(b(0), 0.0f));  // not working
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}
	{
		hkVector4 a; a.set(4, 9, 16, -25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
		HK_TEST(b.isOk<3>());
	}
	{
		hkVector4 a; a.set(-4, -9, -16, -25);
		hkVector4 b; b.setSqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>(a);

		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(0)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(1)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(2)) == false) );
		ON_SIMD_CPU( HK_TEST(hkMath::isFinite(b(3)) == false) );
		ON_SIMD_CPU( HK_TEST(b.isOk<1>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<2>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<3>() == hkFalse32) );
		ON_SIMD_CPU( HK_TEST(b.isOk<4>() == hkFalse32) );
	}
}

static void vector_broadcast()
{
	{
		hkVector4 x; x.set( 5, 2,-4, 9);
		hkVector4 y; y.set(-3, 1, 8, 2);
		hkVector4 z; z.set(-1, 7, 6, 3);
		for( int i = 0; i < 4; ++i )
		{
			hkVector4 a = x;
			a.broadcast(i);
			HK_TEST( a(0) == x(i) );
			HK_TEST( a(1) == x(i) );
			HK_TEST( a(2) == x(i) );
			HK_TEST( a(3) == x(i) );

			hkVector4 b;
			b.setBroadcast(i,y);
			HK_TEST( b(0) == y(i) );
			HK_TEST( b(1) == y(i) );
			HK_TEST( b(2) == y(i) );
			HK_TEST( b(3) == y(i) );
		}

		{
			hkVector4 c;
			hkVector4 d;
			hkSimdReal eps; eps.setFromFloat(1e-3f);

			c = z;
			d.setAll(z(0));
			c.broadcast<0>();
			HK_TEST( c.allEqual<4>(d, eps));

			c = z;
			d.setAll(z(1));
			c.broadcast<1>();
			HK_TEST( c.allEqual<4>(d, eps));

			c = z;
			d.setAll(z(2));
			c.broadcast<2>();
			HK_TEST( c.allEqual<4>(d, eps));

			c = z;
			d.setAll(z(3));
			c.broadcast<3>();
			HK_TEST( c.allEqual<4>(d, eps));
		}
		{
			hkVector4 c;
			hkVector4 d;
			hkSimdReal eps; eps.setFromFloat(1e-3f);

			d.setAll(z(0));
			c.setBroadcast<0>(z);
			HK_TEST( c.allEqual<4>(d, eps));

			c = z;
			d.setAll(z(1));
			c.setBroadcast<1>(z);
			c.broadcast<1>();
			HK_TEST( c.allEqual<4>(d, eps));

			c = z;
			d.setAll(z(2));
			c.setBroadcast<2>(z);
			c.broadcast<2>();
			HK_TEST( c.allEqual<4>(d, eps));

			c = z;
			d.setAll(z(3));
			c.setBroadcast<3>(z);
			HK_TEST( c.allEqual<4>(d, eps));
		}

		// setBroadcastXYZ
		{
			hkVector4 c;
			hkVector4 d; d.setZero();
			hkSimdReal eps; eps.setFromFloat(1e-3f);

			d.setXYZ(z(0));
			c.setBroadcastXYZ(0, z);
			HK_TEST( c.allEqual<3>(d, eps));
			HK_ON_DEBUG(checkEqualNan(c(3)));
		}
	}
}

#define SETUP_SHUFFLE_BASE(S, P) { const int shuf = ( (S) & 0x03) | (( (S) & 0x0c) << 2) | (( (S) & 0x30) << 4) | (( (S) & 0xc0) << 6); res[ (S) ].setPermutation<(hkVectorPermutation::Permutation)shuf>(P); }

#define SETUP_SHUFFLE_0(S, P) SETUP_SHUFFLE_BASE( (S) , P) SETUP_SHUFFLE_BASE( (S)|(1<<0) , P)
#define SETUP_SHUFFLE_1(S, P) SETUP_SHUFFLE_0   ( (S) , P) SETUP_SHUFFLE_0   ( (S)|(1<<1) , P)
#define SETUP_SHUFFLE_2(S, P) SETUP_SHUFFLE_1   ( (S) , P) SETUP_SHUFFLE_1   ( (S)|(1<<2) , P)
#define SETUP_SHUFFLE_3(S, P) SETUP_SHUFFLE_2   ( (S) , P) SETUP_SHUFFLE_2   ( (S)|(1<<3) , P)
#define SETUP_SHUFFLE_4(S, P) SETUP_SHUFFLE_3   ( (S) , P) SETUP_SHUFFLE_3   ( (S)|(1<<4) , P)
#define SETUP_SHUFFLE_5(S, P) SETUP_SHUFFLE_4   ( (S) , P) SETUP_SHUFFLE_4   ( (S)|(1<<5) , P)
#define SETUP_SHUFFLE_6(S, P) SETUP_SHUFFLE_5   ( (S) , P) SETUP_SHUFFLE_5   ( (S)|(1<<6) , P)
#define SETUP_SHUFFLE(P)   SETUP_SHUFFLE_6(0, P); SETUP_SHUFFLE_6(128, P);

static void vector_shuffle()
{
#if 0
	// todo
	//defined(HK_COMPILER_HAS_INTRINSICS_ALTIVEC)
	hkVector4 output1, output2, test1, test2;
	hkVector4 result1, result2;

	hkPseudoRandomGenerator random(22);

	random.getRandomVector11( test1 );
	random.getRandomVector11( test2 );

	HK_VECTOR4_PERM1( output1, test1, HK_VECTOR4_PERM1ARG(3,2,0,1) ) ;
	result1.set(test1(3),test1(2),test1(0),test1(1));
	HK_VECTOR4_PERM2( output2, test1, test2, VPERMWI_CONST(Z,B,D,C) ) ;
	result2.set(test1(2),test2(1),test2(3),test2(2));

	HK_TEST( output1.equals4(result1) );
	HK_TEST( output2.equals4(result2) );

//#elif defined(HK_COMPILER_HAS_INTRINSICS_IA32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4 a, b, output, result;
	a.set(2.0f,3.0f,5.0f,7.0f);
	b.set(11.0f,13.0f,17.0f,19.0f);

	HK_VECTOR4_SHUF(output.m_quad,a.m_quad,b.m_quad,HK_VECTOR4_SHUFFLE(2,3,1,2));
	result.set(a(2),a(3),b(1),b(2));

	HK_TEST( output.approxEqual<4>(result,hkSimdReal(1e-3f)) );
#endif

	// Test setFlipSign
	if (0){
		// todo rewrite for changed mask
		for (int i = 0; i < 16; i++)
		{
			hkVector4 v; v.set(1, 2, 3, 4);
			hkVector4 r1;
			hkVector4Comparison cmp; cmp.set((hkVector4ComparisonMask::Mask)i);

			r1.setFlipSign(v, cmp);

			// Set by hand
			hkVector4 r2;
			for (int j = 0; j < 4; j++)
			{
				r2(j) = (i & (1 << j)) ? -v(j) : v(j);
			}

			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( r1.allEqual<4>(r2,eps));
		}
	}


	// Shuffle
	{
		hkVector4 v; v.set(1, 2, 3, 4);
		hkVector4 res[256];

		SETUP_SHUFFLE(v);

		for (int i = 0; i < 256; i++)
		{
			const int x = i & 3;
			const int y = (i >> 2) & 3;
			const int z = (i >> 4) & 3;
			const int w = (i >> 6) & 3;

			hkVector4 r1;
			r1.set(v(w), v(z), v(y), v(x));

			const hkVector4& r2 = res[i];

			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( r1.allEqual<4>(r2,eps));
		}
	}
}

static void vector_load_store()
{
	const int NUM_TIMES = 100;

	// Testing functionality of HK_IO_NATIVE_ALIGNED and isOk() - 3 components
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		for(int i = 0; i < NUM_TIMES; i++)
		{
			hkReal p[4] = {rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01()};
			x.setAll(0);
			x.load<3,HK_IO_NATIVE_ALIGNED>(p);
			HK_TEST( hkMath::equal(x(0),p[0]) );
			HK_TEST( hkMath::equal(x(1),p[1]) );
			HK_TEST( hkMath::equal(x(2),p[2]) );
			// HK_TEST( !hkMath::equal(x(3),p[3]) ); cannot really check this as docs say its undef
			HK_TEST( x.isOk<3>() );
			HK_ON_DEBUG(checkEqualNan(x(3)));
		}
	}

	// Testing functionality of HK_IO_NATIVE_ALIGNED and isOk() - 4 components
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		for(int i = 0; i < NUM_TIMES; i++)
		{
			hkReal p[4] = {rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01()};
			x.setAll(0);
			x.load<4,HK_IO_NATIVE_ALIGNED>(p);
			HK_TEST( hkMath::equal(x(0),p[0]) );
			HK_TEST( hkMath::equal(x(1),p[1]) );
			HK_TEST( hkMath::equal(x(2),p[2]) );
			HK_TEST( hkMath::equal(x(3),p[3]) );
			HK_TEST( x.isOk<4>() );
		}
	}

	// Testing functionality of HK_IO_NATIVE_ALIGNED and isOk() - 2 components
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		for(int i = 0; i < NUM_TIMES; i++)
		{
			hkReal p[4] = {rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01()};
			x.setAll(0);
			x.load<2,HK_IO_NATIVE_ALIGNED>(p);
			HK_TEST( hkMath::equal(x(0),p[0]) );
			HK_TEST( hkMath::equal(x(1),p[1]) );
			HK_TEST( x.isOk<2>() );
			HK_ON_DEBUG(checkEqualNan(x(2)));
			HK_ON_DEBUG(checkEqualNan(x(3)));
		}
	}

	{
		// Test native aligned loads
		const hkReal testArray[] = {0,1,2,3,4,5,6,7,8,9,10};
		for (int i=0; i<=7; i++)
		{
			hkVector4 v;
			v.load<4,HK_IO_NATIVE_ALIGNED>(&testArray[i]);
			HK_TEST( v(0) == testArray[i] && v(1) == testArray[i+1] && v(2) == testArray[i+2] && v(3) == testArray[i+3] );
		}
	}

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_RVL) && !defined(HK_PLATFORM_WIIU) && !defined(HK_REAL_IS_DOUBLE) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_IOS_SIM) && !defined(HK_PLATFORM_TIZEN)
	{
		// Test byte aligned load<4>
		const hkUint8 testArray[40] = {0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9};
		for (int i=3; i<10; ++i)
		{
			const hkUint8* ptr = testArray+i;
			hkVector4 vec;
			vec.load<4,HK_IO_BYTE_ALIGNED>((const hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<16; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
		}
		// Test byte aligned load<3>
		for (int i=3; i<10; ++i)
		{
			const hkUint8* ptr = testArray+i;
			hkVector4 vec;
			vec.load<3,HK_IO_BYTE_ALIGNED>((const hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<12; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
		}
		// Test byte aligned load<2>
		for (int i=3; i<10; ++i)
		{
			const hkUint8* ptr = testArray+i;
			hkVector4 vec;
			vec.load<2,HK_IO_BYTE_ALIGNED>((const hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<8; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
		}
		// Test byte aligned load<1>
		for (int i=3; i<10; ++i)
		{
			const hkUint8* ptr = testArray+i;
			hkVector4 vec;
			vec.load<1,HK_IO_BYTE_ALIGNED>((const hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<4; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
		}
	}
#endif

	// aligned loads
	{
		HK_ALIGN_REAL(const hkReal testArray[]) = {0,1,2,3,4,5,6,7,8,9,10};

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_IOS_SIM) && !defined(HK_PLATFORM_TIZEN)
		// neon does not align

		// Test that alignment asserts are triggered when loading unaligned data.
		for (int i=1; i<=3; i++)
		{
			hkVector4 v;
			if ( i != 2 )
			{
				HK_TEST_ASSERT(0x64211c2f, v.load<2>(&testArray[i]));
				HK_TEST_ASSERT(0x64211c2f, v.load<3>(&testArray[i]));
				HK_TEST_ASSERT(0x64211c2f, v.load<4>(&testArray[i]));
			}
			else
			{
				// verify that it doesn't assert for 8 byte alignments on load<2>
				v.load<2>(&testArray[i]);
#if !defined(HK_REAL_IS_DOUBLE)
				// max 16 byte alignment necessary
				HK_TEST_ASSERT(0x64211c2f, v.load<3>(&testArray[i]));
				HK_TEST_ASSERT(0x64211c2f, v.load<4>(&testArray[i]));
#endif
			}
		}

		// Test for load<1> that values unaligned to 4 bytes will cause an alignment assert.
#if !defined(HK_REAL_IS_DOUBLE)
		// Removing this test from SNC Compiler as it removes the HK_ASSERT2(0x64211c2f) in the load<1> case
		// since it presumes that hkReal is 4 byte aligned
#if !defined(HK_COMPILER_SNC)
		{
			hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
			HK_ALIGN_REAL(char testCharArray[32]);
			float randomFloat = rng.getRandReal01();
			for (int i = 1; i <=3; i++)
			{
				float* targetFloat = (float*)(&testCharArray[i]);
				hkMemUtil::memCpy(targetFloat, &randomFloat, sizeof(randomFloat));
				HK_ON_TEST_ASSERT_ENABLED(hkVector4 v);
				HK_TEST_ASSERT(0x64211c2f, v.load<1>(targetFloat));
			}
		}
#endif
#endif
#endif

		{
			hkVector4 v;
			v.load<1>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] );
			HK_ON_DEBUG(checkEqualNan(v(1)));
			HK_ON_DEBUG(checkEqualNan(v(2)));
			HK_ON_DEBUG(checkEqualNan(v(3)));

			v.load<1>(&testArray[1]);
			HK_TEST( v(0) == testArray[1] );
			HK_ON_DEBUG(checkEqualNan(v(1)));
			HK_ON_DEBUG(checkEqualNan(v(2)));
			HK_ON_DEBUG(checkEqualNan(v(3)));

			v.load<1>(&testArray[2]);
			HK_TEST( v(0) == testArray[2] );
			HK_ON_DEBUG(checkEqualNan(v(1)));
			HK_ON_DEBUG(checkEqualNan(v(2)));
			HK_ON_DEBUG(checkEqualNan(v(3)));

			v.load<1>(&testArray[3]);
			HK_TEST( v(0) == testArray[3] );
			HK_ON_DEBUG(checkEqualNan(v(1)));
			HK_ON_DEBUG(checkEqualNan(v(2)));
			HK_ON_DEBUG(checkEqualNan(v(3)));
		}
		{
			hkVector4 v;
			v.load<2>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] && v(1) == testArray[1]);
			HK_ON_DEBUG(checkEqualNan(v(2)));
			HK_ON_DEBUG(checkEqualNan(v(3)));

			v.load<2>(&testArray[2]);
			HK_TEST( v(0) == testArray[2] && v(1) == testArray[3]);
			HK_ON_DEBUG(checkEqualNan(v(2)));
			HK_ON_DEBUG(checkEqualNan(v(3)));
		}
		{
			hkVector4 v;
			v.load<3>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] && v(1) == testArray[1] && v(2) == testArray[2] );
			HK_ON_DEBUG(checkEqualNan(v(3)));
		}
		{
			hkVector4 v;
			v.load<4>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] && v(1) == testArray[1] && v(2) == testArray[2] && v(3) == testArray[3] );
		}
	}

	// uncached loads
	{
		HK_ALIGN_REAL(const hkReal testArray[]) = {0,1,2,3,4,5,6,7,8,9,10};

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_IOS_SIM) && !defined(HK_PLATFORM_TIZEN)
		// neon does not align

		// Test alignments
		for (int i=1; i<=3; i++)
		{
			hkVector4 v;
			if ( i != 2 )
			{
				HK_TEST_ASSERT(0x64211c2f, (v.load<2,HK_IO_NOT_CACHED>(&testArray[i])));
				HK_TEST_ASSERT(0x64211c2f, (v.load<3,HK_IO_NOT_CACHED>(&testArray[i])));
				HK_TEST_ASSERT(0x64211c2f, (v.load<4,HK_IO_NOT_CACHED>(&testArray[i])));
			}
			else
			{
				// verify that it doesn't assert for 8 byte alignments on load<2>
				v.load<2,HK_IO_NOT_CACHED>(&testArray[i]);
#if !defined(HK_REAL_IS_DOUBLE)
				// max 16 byte alignment necessary
				HK_TEST_ASSERT(0x64211c2f, (v.load<3,HK_IO_NOT_CACHED>(&testArray[i])));
				HK_TEST_ASSERT(0x64211c2f, (v.load<4,HK_IO_NOT_CACHED>(&testArray[i])));
#endif
			}
		}

		// Test alignments for loadNotCached<1>. Values unaligned to 4 bytes will cause an alignment assert.
#if !defined(HK_REAL_IS_DOUBLE)
		// Removing this test from SNC Compiler as it removes the HK_ASSERT2(0x64211c2f) in the loadNotCached<1> case
		// since it presumes that hkReal is 4 byte aligned
#if !defined(HK_COMPILER_SNC)
		{
			hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
			HK_ALIGN_REAL(char testCharArray[32]);
			float randomFloat = rng.getRandReal01();
			for (int i = 1; i <=3; i++)
			{
				float* targetFloat = (float*)(&testCharArray[i]);
				hkMemUtil::memCpy(targetFloat, &randomFloat, sizeof(randomFloat));
				HK_ON_TEST_ASSERT_ENABLED(hkVector4 v);
				HK_TEST_ASSERT(0x64211c2f, (v.load<1,HK_IO_NOT_CACHED>(targetFloat)));
			}
		}
#endif
#endif
#endif

		{
			hkVector4 v;
			v.load<1,HK_IO_NOT_CACHED>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] );
		}
		{
			hkVector4 v;
			v.load<2,HK_IO_NOT_CACHED>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] && v(1) == testArray[1]);
		}
		{
			hkVector4 v;
			v.load<3,HK_IO_NOT_CACHED>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] && v(1) == testArray[1] && v(2) == testArray[2] );
		}
		{
			hkVector4 v;
			v.load<4,HK_IO_NOT_CACHED>(&testArray[0]);
			HK_TEST( v(0) == testArray[0] && v(1) == testArray[1] && v(2) == testArray[2] && v(3) == testArray[3] );
		}
	}

	// Testing functionality of store() with HK_IO_NATIVE_ALIGNED
	// Make sure it doesn't write outside the three values
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		const int arraySize = 32;
		const hkReal guardVal = 123.0f;
		hkVector4 x;
		hkReal p[arraySize];
		for(int i = 0; i < arraySize - 4; i++)
		{
			// set the array to guard values
			for (int j=0; j<arraySize; j++)
			{
				p[j] = guardVal;
			}

			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
			x.store<3,HK_IO_NATIVE_ALIGNED>(p + i);

			for(int j=0; j<arraySize; j++)
			{
				if( j>=i && j<=i+2)
				{
					// Make sure we wrote the write value
					HK_TEST(hkMath::equal(x(j-i),p[j]));
				}
				else
				{
					HK_TEST(p[j] == guardVal);
				}
			}
		}
	}

	// Testing functionality of store() with HK_IO_NATIVE_ALIGNED
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		hkReal p[4];
		for(int i = 0; i < NUM_TIMES; i++)
		{
			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());

			x.store<4,HK_IO_NATIVE_ALIGNED>(p);
			HK_TEST( hkMath::equal(x(0),p[0]) );
			HK_TEST( hkMath::equal(x(1),p[1]) );
			HK_TEST( hkMath::equal(x(2),p[2]) );
			HK_TEST( hkMath::equal(x(3),p[3]) );
		}
	}

	// TODO add guard checks around store4 like for store3.

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_RVL) && !defined(HK_PLATFORM_WIIU) && !defined(HK_REAL_IS_DOUBLE) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_IOS_SIM) && !defined(HK_PLATFORM_TIZEN)
	{
		// Test byte aligned store<4>
		hkUint8 testArray[40];
		hkVector4 vec; vec.set(0.0f,1.0f,2.0f,3.0f);
		for (int i=3; i<10; ++i)
		{
			for (int k=0; k<40; ++k) testArray[k] = 99;
			hkUint8* ptr = testArray+i;
			vec.store<4,HK_IO_BYTE_ALIGNED>((hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<i; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
			for (int k=0; k<16; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
			for (int k=i+16; k<40; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
		}
		// Test byte aligned store<3>
		for (int i=3; i<10; ++i)
		{
			for (int k=0; k<40; ++k) testArray[k] = 99;
			hkUint8* ptr = testArray+i;
			vec.store<3,HK_IO_BYTE_ALIGNED>((hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<i; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
			for (int k=0; k<12; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
			for (int k=i+12; k<40; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
		}
		// Test byte aligned store<2>
		for (int i=3; i<10; ++i)
		{
			for (int k=0; k<40; ++k) testArray[k] = 99;
			hkUint8* ptr = testArray+i;
			vec.store<2,HK_IO_BYTE_ALIGNED>((hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<i; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
			for (int k=0; k<8; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
			for (int k=i+8; k<40; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
		}
		// Test byte aligned store<1>
		for (int i=3; i<10; ++i)
		{
			for (int k=0; k<40; ++k) testArray[k] = 99;
			hkUint8* ptr = testArray+i;
			vec.store<1,HK_IO_BYTE_ALIGNED>((hkReal*)ptr);
			const hkUint8* v = (const hkUint8*)&vec;
			for (int k=0; k<i; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
			for (int k=0; k<4; ++k)
			{
				HK_TEST( v[k] == ptr[k] );
			}
			for (int k=i+4; k<40; ++k)
			{
				HK_TEST( testArray[k] == 99 );
			}
		}
	}
#endif

	//Testing functionality of storeUncached()
	{
		HK_ALIGN_REAL(hkReal data[8]);
		for(hkReal i = 0; i < NUM_TIMES; i++)
		{
			{
				hkVector4 vec;
				vec.set( i, i+1, i+2, i+3 );
				vec.store<4,HK_IO_NOT_CACHED>(data);
				for (int j = 0 ; j < 4 ; j++)
				{
					HK_TEST( hkMath::equal(vec(j),data [j]) );
				}
			}
			{
				hkVector4 vec;
				vec.set( i, i+1, i+2, i+3 );
				vec.store<3,HK_IO_NOT_CACHED>(data);
				for (int j = 0 ; j < 3 ; j++)
				{
					HK_TEST( hkMath::equal(vec(j),data [j]) );
				}
			}
			{
				hkVector4 vec;
				vec.set( i, i+1, i+2, i+3 );
				vec.store<2,HK_IO_NOT_CACHED>(data);
				for (int j = 0 ; j < 2 ; j++)
				{
					HK_TEST( hkMath::equal(vec(j),data [j]) );
				}
			}

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_IOS_SIM) && !defined(HK_PLATFORM_TIZEN)
		// neon does not align
#if !defined(HK_REAL_IS_DOUBLE)
		// cannot run misaligned double code
			{
				HK_ON_TEST_ASSERT_ENABLED(hkVector4 vec);
				HK_ON_TEST_ASSERT_ENABLED(vec.set( i, i+1, i+2, i+3 ));
				HK_TEST_ASSERT(0x64211c2f, (vec.store<4,HK_IO_NOT_CACHED>(&data[1])));
				HK_TEST_ASSERT(0x64211c2f, (vec.store<4,HK_IO_NOT_CACHED>(&data[2])));
				HK_TEST_ASSERT(0x64211c2f, (vec.store<4,HK_IO_NOT_CACHED>(&data[3])));
				//HK_TEST_ASSERT(0x64211c2f, (vec.store<1,HK_IO_NOT_CACHED>(&data[1])));
				HK_TEST_ASSERT(0x64211c2f, (vec.store<2,HK_IO_NOT_CACHED>(&data[1])));
				HK_TEST_ASSERT(0x64211c2f, (vec.store<3,HK_IO_NOT_CACHED>(&data[1])));
			}
#endif
#endif
		}
	}

	//Testing functionality of store()
	{
		HK_ALIGN_REAL(hkReal data[8]);
		for(hkReal i = 0; i < NUM_TIMES; i++)
		{
			{
				hkVector4 vec;
				vec.set( i, i+1, i+2, i+3 );
				vec.store<4>(data);
				for (int j = 0 ; j < 4 ; j++)
				{
					HK_TEST( hkMath::equal(vec(j),data [j]) );
				}
			}
			{
				hkVector4 vec;
				vec.set( i, i+1, i+2, i+3 );
				vec.store<3>(data);
				for (int j = 0 ; j < 3 ; j++)
				{
					HK_TEST( hkMath::equal(vec(j),data [j]) );
				}
			}
			{
				hkVector4 vec;
				vec.set( i, i+1, i+2, i+3 );
				vec.store<2>(data);
				for (int j = 0 ; j < 2 ; j++)
				{
					HK_TEST( hkMath::equal(vec(j),data [j]) );
				}
			}

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_IOS_SIM) && !defined(HK_PLATFORM_TIZEN)
		// neon does not align
			{
				HK_ON_TEST_ASSERT_ENABLED(hkVector4 vec);
				HK_ON_TEST_ASSERT_ENABLED(vec.set( i, i+1, i+2, i+3 ));
				HK_TEST_ASSERT(0x64211c2f, vec.store<4>(&data[1]));
#if !defined(HK_REAL_IS_DOUBLE)
				// max 16 byte alignment
				HK_TEST_ASSERT(0x64211c2f, vec.store<4>(&data[2]));
#endif
				HK_TEST_ASSERT(0x64211c2f, vec.store<4>(&data[3]));
				//HK_TEST_ASSERT(0x64211c2f, vec.store<1>(&data[1]));
				HK_TEST_ASSERT(0x64211c2f, vec.store<2>(&data[1]));
				HK_TEST_ASSERT(0x64211c2f, vec.store<3>(&data[1]));
			}
#endif
		}
	}

}

static void vector_length0()
{
	// COM-710
	// The following functions shouldn't assert for zero-length vectors, and should graciously handle the results (no NAN or INF)
	//   lengthInverse3
	//   lengthInverse4
	//   normalize3
	//   normalizeWithLength3
	//   normalize4
	//   normalizeWithLength4

	hkVector4 zero; zero.setZero();

	{
		hkReal lengthInverse3Result = zero.lengthInverse<3>().getReal();
		HK_TEST( lengthInverse3Result == 0.0f );
	}

	{
		hkReal lengthInverse4Result = zero.lengthInverse<4>().getReal();
		HK_TEST( lengthInverse4Result == 0.0f );
	}

	{
		hkVector4 normalize3Result = zero;
		normalize3Result.normalize<3>();
		HK_TEST( normalize3Result(0) == 0.0f && normalize3Result(1) == 0.0f && normalize3Result(2) == 0.0f ) ;
	}

	{
		hkVector4 normalize4Result = zero;
		normalize4Result.normalize<4>();
		HK_TEST( normalize4Result(0) == 0.0f && normalize4Result(1) == 0.0f && normalize4Result(2) == 0.0f && normalize4Result(3) == 0.0f);
	}

 	{
 		hkVector4 normLen3Result = zero;
 		hkReal len3 = normLen3Result.normalizeWithLength<3>().getReal();
 		HK_TEST(len3 == 0.0f);
 		HK_TEST( normLen3Result(0) == 0.0f && normLen3Result(1) == 0.0f && normLen3Result(2) == 0.0f);
 	}

 	{
 		hkVector4 normLen4Result = zero;
 		hkReal len4 = normLen4Result.normalizeWithLength<4>().getReal();
 		HK_TEST(len4 == 0.0f);
 		HK_TEST( normLen4Result(0) == 0.0f && normLen4Result(1) == 0.0f && normLen4Result(2) == 0.0f && normLen4Result(3) == 0.0f);
 	}
}


static void vector_accessors()
{
	// getW
	{
		hkVector4 a; a.set(1.0f, 2.0f, 3.0f, 4.0f);
		HK_TEST( hkMath::equal(a.getW().getReal(), 4.0f));

		hkPseudoRandomGenerator random(10);
		for (int i = 0; i < 100; i++)
		{
			hkSimdReal r; r.setFromFloat(random.getRandReal11());
			a.setComponent<3>(r);
			HK_TEST( hkMath::equal(a.getW().getReal(), r.getReal()));
		}
	}

	// getMaxComponentIndex
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 ma; ma.setHorizontalMax<4>( v0 );
		int maxIndex = v0.getIndexOfMaxComponent<4>();
		HK_TEST( v0(maxIndex) == ma(0) );
	}
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 ma; ma.setHorizontalMax<3>( v0 );
		int maxIndex = v0.getIndexOfMaxComponent<3>();
		HK_TEST( v0(maxIndex) == ma(0) );
	}
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 ma; ma.setHorizontalMax<2>( v0 );
		int maxIndex = v0.getIndexOfMaxComponent<2>();
		HK_TEST( v0(maxIndex) == ma(0) );
	}

	{
		hkVector4 v0; v0.set(1.0f, 0.5f, 1.0f, 1.0f);
		hkVector4 ma; ma.setHorizontalMax<4>( v0 );
		HK_TEST( v0.getIndexOfMaxComponent<4>() == 3 );
		HK_TEST( v0.getIndexOfMaxComponent<3>() == 2 );
		HK_TEST( v0.getIndexOfMaxComponent<2>() == 0 );
	}

	// getMinComponentIndex
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 mi; mi.setHorizontalMin<4>( v0 );
		int minIndex = v0.getIndexOfMinComponent<4>();
		HK_TEST( v0(minIndex) == mi(0) );
	}
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 mi; mi.setHorizontalMin<3>( v0 );
		int minIndex = v0.getIndexOfMinComponent<3>();
		HK_TEST( v0(minIndex) == mi(0) );
	}
	{
		hkPseudoRandomGenerator random(10);
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 mi; mi.setHorizontalMin<2>( v0 );
		int minIndex = v0.getIndexOfMinComponent<2>();
		HK_TEST( v0(minIndex) == mi(0) );
	}

	{
		hkVector4 v0; v0.set(0.5f, 0.5f, 1.0f, 0.5f);
		hkVector4 mi; mi.setHorizontalMin<4>( v0 );
		HK_TEST( v0.getIndexOfMinComponent<4>() == 0 );
		HK_TEST( v0.getIndexOfMinComponent<3>() == 0 );
		HK_TEST( v0.getIndexOfMinComponent<2>() == 0 );
	}

	// getConstant
	{
		HK_TEST( hkVector4::getConstant( HK_QUADREAL_0)(0)		== 0.0f );
		HK_TEST( hkVector4::getConstant( HK_QUADREAL_256)(0)	== 256.0f );
		HK_TEST( hkVector4::getConstant( HK_QUADREAL_INV_4)(0)	== 0.25f );
		HK_TEST( hkVector4::getConstant( HK_QUADREAL_m11m11)(3) == 1.0f );
		HK_TEST( hkVector4::getConstant( HK_QUADREAL_8421)(2)	== 2.0f );
	}


	// Testing functionality of getZero()
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		hkVector4 x;
		hkVector4 y;
		const int NUM_TIMES = 100;
		for(int i = 0; i < NUM_TIMES; i++)
		{
			x.set(rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01(),rng.getRandReal01());
			y = x.getZero();
			HK_TEST( hkMath::equal(y(0),0.0f) );
			HK_TEST( hkMath::equal(y(1),0.0f) );
			HK_TEST( hkMath::equal(y(2),0.0f) );
			HK_TEST( hkMath::equal(y(3),0.0f) );
		}
	}

	// getAbsMaxComponentIndex.
	{
		hkVector4 a; a.set(.9f,.8f,.8f);
		HK_TEST( a.getIndexOfMaxAbsComponent<3>() == 0 );
		hkVector4 b; b.set(.8f,.9f,.8f);
		HK_TEST( b.getIndexOfMaxAbsComponent<3>() == 1 );
		hkVector4 c; c.set(.8f,.8f,.9f);
		HK_TEST( c.getIndexOfMaxAbsComponent<3>() == 2 );
		HK_TEST( c.getIndexOfMaxAbsComponent<2>() == 1 );
		hkVector4 d; d.set(.8f,.8f,.8f,.9f);
		HK_TEST( d.getIndexOfMaxAbsComponent<3>() == 2 );
		HK_TEST( d.getIndexOfMaxAbsComponent<4>() == 3 );
	}
	{
		hkVector4 a; a.set(-.9f,.8f,.8f);
		HK_TEST( a.getIndexOfMaxAbsComponent<3>() == 0 );
		hkVector4 b; b.set(.8f,-.9f,.8f);
		HK_TEST( b.getIndexOfMaxAbsComponent<3>() == 1 );
		hkVector4 c; c.set(.8f,.8f,-.9f);
		HK_TEST( c.getIndexOfMaxAbsComponent<3>() == 2 );
		HK_TEST( c.getIndexOfMaxAbsComponent<2>() == 1 );
		hkVector4 d; d.set(.8f,.8f,.8f,-.9f);
		HK_TEST( d.getIndexOfMaxAbsComponent<3>() == 2 );
		HK_TEST( d.getIndexOfMaxAbsComponent<4>() == 3 );
	}

	// getAbsMinComponentIndex.
	{
		hkVector4 a; a.set(.9f,.8f,.8f);
		HK_TEST( a.getIndexOfMinAbsComponent<3>() == 1 );
		hkVector4 b; b.set(.8f,.9f,.8f);
		HK_TEST( b.getIndexOfMinAbsComponent<3>() == 0 );
		hkVector4 c; c.set(.8f,.8f,.9f);
		HK_TEST( c.getIndexOfMinAbsComponent<3>() == 0 );
		HK_TEST( c.getIndexOfMinAbsComponent<2>() == 0 );
		hkVector4 d; d.set(.8f,.8f,.8f,.9f);
		HK_TEST( d.getIndexOfMinAbsComponent<3>() == 0 );
		HK_TEST( d.getIndexOfMinAbsComponent<4>() == 0 );
	}
	{
		hkVector4 a; a.set(-.9f,.8f,.8f);
		HK_TEST( a.getIndexOfMinAbsComponent<3>() == 1 );
		hkVector4 b; b.set(.8f,-.9f,.8f);
		HK_TEST( b.getIndexOfMinAbsComponent<3>() == 0 );
		hkVector4 c; c.set(.8f,.8f,-.9f);
		HK_TEST( c.getIndexOfMinAbsComponent<3>() == 0 );
		HK_TEST( c.getIndexOfMinAbsComponent<2>() == 0 );
		hkVector4 d; d.set(.8f,.8f,.8f,-.9f);
		HK_TEST( d.getIndexOfMinAbsComponent<3>() == 0 );
		HK_TEST( d.getIndexOfMinAbsComponent<4>() == 0 );
	}

	// getComponent, getReal.
	{
		hkVector4 a; a.set(5,2,-4,9);
		HK_TEST( a(0) == a.getComponent<0>().getReal() );
		HK_TEST( a(1) == a.getComponent<1>().getReal() );
		HK_TEST( a(2) == a.getComponent<2>().getReal() );
		HK_TEST( a(3) == a.getComponent<3>().getReal() );
	}

	// horizontalMax, horizontalMin
	{
		hkVector4 v;

		v.set(1.0f, 2.0f, 3.0f, 4.0f);
		HK_TEST(v.horizontalMax<3>().getReal() == 3.0f);
		HK_TEST(v.horizontalMin<3>().getReal() == 1.0f);
		HK_TEST(v.horizontalMax<4>().getReal() == 4.0f);
		HK_TEST(v.horizontalMax<2>().getReal() == 2.0f);

		v.set(2.0f, 3.0f, 4.0f, 1.0f);
		HK_TEST(v.horizontalMax<3>().getReal() == 4.0f);
		HK_TEST(v.horizontalMin<3>().getReal() == 2.0f);
		HK_TEST(v.horizontalMin<4>().getReal() == 1.0f);

		v.set(3.0f, 4.0f, 1.0f, 2.0f);
		HK_TEST(v.horizontalMax<3>().getReal() == 4.0f);
		HK_TEST(v.horizontalMin<3>().getReal() == 1.0f);
		HK_TEST(v.horizontalMin<2>().getReal() == 3.0f);

		v.set(4.0f, 1.0f, 2.0f, 3.0f);
		HK_TEST(v.horizontalMax<3>().getReal() == 4.0f);
		HK_TEST(v.horizontalMin<3>().getReal() == 1.0f);
	}
}


static void checkPackedVector3( hkVector4Parameter check )
{
	hkPackedVector3 cv;	cv.pack(check);

	hkVector4 a; a.setAbs( check );
	hkVector4 ma; ma.setHorizontalMax<3>(a);
	hkVector4 ref; cv.unpack(ref);
	const hkReal eps = (1.0f / 0x7f00 );
	hkReal m = hkMath::max2( ma(0), HK_REAL_EPSILON * HK_REAL_EPSILON );
	hkSimdReal meps; meps.setFromFloat(m * eps);
	HK_TEST( ref.allEqual<3>( check, meps ) );
}

static void checkPackedVector8_3( hkVector4Parameter check )
{
	hkPackedVector8_3 cv;	cv.pack(check);

	hkVector4 a; a.setAbs( check );
	hkVector4 ma; ma.setHorizontalMax<4>(a);
	hkVector4 ref; cv.unpack(ref);
	hkReal eps = (1.0f / 0x7f );
	hkReal m = hkMath::max2( ma(0), HK_REAL_EPSILON * HK_REAL_EPSILON );
	hkSimdReal meps; meps.setFromFloat(m * eps);
	HK_TEST( ref.allEqual<3>( check, meps ) );
}

static void vector_packed()
{

	checkPackedVector3( hkVector4::getConstant( HK_QUADREAL_0) );
	checkPackedVector3( hkVector4::getConstant( HK_QUADREAL_1) );
	checkPackedVector3( hkVector4::getConstant( HK_QUADREAL_0100) );
	checkPackedVector3( hkVector4::getConstant( HK_QUADREAL_m11m11) );
	checkPackedVector3( hkVector4::getConstant( HK_QUADREAL_EPS) );
	//checkPackedVector3( hkVector4::getConstant( HK_QUADREAL_MAX) ); // does not work

	checkPackedVector8_3( hkVector4::getConstant( HK_QUADREAL_0) );
	checkPackedVector8_3( hkVector4::getConstant( HK_QUADREAL_1) );
	checkPackedVector8_3( hkVector4::getConstant( HK_QUADREAL_0100) );
	checkPackedVector8_3( hkVector4::getConstant( HK_QUADREAL_m11m11) );
	checkPackedVector8_3( hkVector4::getConstant( HK_QUADREAL_EPS) );

	{
		hkVector4 test; test.set(0.99999f,-0.1f,0.1f);
		checkPackedVector3( test );
		checkPackedVector8_3( test );
	}

	hkPseudoRandomGenerator random(10);
	for (int i =0; i < 100; i++)
	{
		hkVector4 v0;

		for (int c = 0; c < 4; c++)
		{
			v0(c) = ((i&3)-1) * hkMath::pow( hkReal(10), random.getRandRange( -6.0f, 10.0f ) );
		}
		checkPackedVector3( v0 );
		checkPackedVector8_3( v0 );
	}
	// check tiny numbers
	{
		hkVector4 x; x.setAll(1.0f);
#if defined(HK_REAL_IS_DOUBLE)
		for (int i = 0; i < 16; i++) // very small numbers still handled correctly by double precision, will not truncate to zero
#else
		for (int i = 0; i < 40; i++)
#endif
		{
			x.setMul(x, hkVector4::getConstant(HK_QUADREAL_INV_7) );
			checkPackedVector3( x );
			checkPackedVector8_3( x );
		}
	}
}

static void vector_half8()
{
	// Test native alignment
	{
		hkVector4 vIn; vIn.set(1, 2, 3, 4);
		HK_ALIGN16(hkHalf h[16]);

		// Test all different offsets within a quad word
		for (int offset = 0; offset < 8; ++offset)
		{
			{
				const int N = 1;

				// Test store
				hkString::memClear16(h, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(h + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(h[i].isZero()); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(h[i].isZero()); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(h + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}

			{
				const int N = 2;

				// Test store
				hkString::memClear16(h, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(h + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(h[i].isZero()); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(h[i].isZero()); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(h + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}

			{
				const int N = 3;

				// Test store
				hkString::memClear16(h, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(h + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(h[i].isZero()); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(h[i].isZero()); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(h + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}

			{
				const int N = 4;

				// Test store
				hkString::memClear16(h, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(h + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(h[i].isZero()); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(h[i].isZero()); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(h + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}
		}
	}

	hkPseudoRandomGenerator random(10);
	for (int i =0; i < 100; i++)
	{
		hkVector4 v0; random.getRandomVector11( v0 );
		hkVector4 v1; random.getRandomVector11( v1 );

		// unpack first, second
		{
			hkHalf h[8];
			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);

			hkVector4 r0;
			hkVector4 r1;
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}
		}

		// pack first, second.
		{
			hkHalf h[8]; for (int k=0; k<8; ++k) h[k].setZero();
			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			hkVector4 r0;
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);

			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);
			hkVector4 r1;
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/127.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/127.f ));
			}
		}


		// pack, unpack.
		{
			hkHalf h[8];
			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);

			hkVector4 r0, r1;
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}
		}

		// unpack first, unpack second.
		{
			hkHalf h0[8*3];
			hkHalf h1[8*3];
			for (int k=0; k<8*3; ++k) { h0[k].setZero(); h1[k].setZero(); }
			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h0+8);
			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h1+8+4);

			hkVector4 r0;
			hkVector4 r1;
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h0+8);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h1+8+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}

			// check for overspill
#if defined(HK_HALF_IS_FLOAT)
			union h2s { hkHalf* h; hkUint32* s; };
#else
			union h2s { hkHalf* h; hkUint16* s; };
#endif
			h2s hs;
			for (int t=0; t < 8*3; t++)
			{
				if ( t<8 && t >=12 )
				{
					hs.h = h0;
					HK_TEST( (hs.s)[t] == 0 );
				}
				if ( t<12 && t >=16 )
				{
					hs.h = h1;
					HK_TEST( (hs.s)[t] == 0 );
				}
			}
		}

		// unpack first, unpack second. Check for overwrites.
		{
			hkHalf h[8]; for (int k=0; k<8; ++k) h[k].setZero();
			hkVector4 r0, r1;

			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);
			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}
		}

		// unpack first, unpack second. Check for overwrites.
		{
			hkHalf h[8]; for (int k=0; k<8; ++k) h[k].setZero();
			hkVector4 r0, r1;

			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}
		}

		// unpack first, unpack second. Check for overwrites.
		{
			hkHalf h[8]; for (int k=0; k<8; ++k) h[k].setZero();
			hkVector4 r0, r1;

			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);
			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}
		}

		// unpack first, unpack second. Check for overwrites.
		{
			hkHalf h[8]; for (int k=0; k<8; ++k) h[k].setZero();
			hkVector4 r0, r1;

			v1.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h+4);
			v0.store<4,HK_IO_NATIVE_ALIGNED,HK_ROUND_NEAREST>(h);
			r1.load<4,HK_IO_NATIVE_ALIGNED>(h+4);
			r0.load<4,HK_IO_NATIVE_ALIGNED>(h);

			for (int k=0; k < 4; k++ )
			{
				HK_TEST( hkMath::equal( v0(k), r0(k), hkMath::fabs(v0(k))/255.f ));
				HK_TEST( hkMath::equal( v1(k), r1(k), hkMath::fabs(v1(k))/255.f ));
			}
		}

#if 0
		// aligned operations not yet implemented



		// Checking functionality and alignment on loadPacked.
		{
			hkHalf8 half;
			HK_ALIGN16(hkHalf testArray[11]);
			for (int k=0; k < 11; k++)
			{
				testArray[k] = float(k);
			}

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_RVL) && !defined(HK_PLATFORM_IOS_SIM)
		// neon does not align
			// Alignment checks.
			for (int j=1; j<=3; j++)
			{
				HK_TEST_ASSERT(0x64211c2f, half.loadPacked(&testArray[j]));
			}
#endif

			{
				half.loadPacked(&testArray[0]);
				for (int j = 0; j < 8; j++)
				{
					HK_TEST( hkMath::equal(half.getComponent(j), testArray[j]) );
				}
			}
		}

		//Testing functionality and alignment of storePacked()
		{
			hkHalf8 half8;
			half8.pack<true>(v0, v1);

			HK_ALIGN16(hkHalf data[12]);
			{
				half8.storePacked(&data[0]);
				for (int j = 0 ; j < 8 ; j++)
				{
					HK_TEST( hkMath::equal(half8.getComponent(j), data [j]) );
				}
			}

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_RVL) && !defined(HK_PLATFORM_IOS_SIM)
		// neon does not align
			// Alignment checks.
			{
				HK_TEST_ASSERT(0x64211c2f, half8.storePacked(&data[1]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePacked(&data[2]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePacked(&data[3]));
			}
#endif
		}

		// StorePackedFirst, StorePackedSecond.
		{
			hkHalf8 half8;
			half8.pack<true>(v0, v1);

			HK_ALIGN16(hkHalf data[12]);
			{
				half8.storePackedFirst(&data[0]);
				half8.storePackedSecond(&data[4]);
				for (int j = 0 ; j < 8 ; j++)
				{
					HK_TEST( hkMath::equal(half8.getComponent(j), data [j]) );
				}
			}

#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_RVL) && !defined(HK_PLATFORM_IOS_SIM)
		// neon does not align
			// Alignment checks.
			{
				HK_TEST_ASSERT(0x64211c2f, half8.storePackedFirst(&data[1]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePackedFirst(&data[2]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePackedFirst(&data[3]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePackedSecond(&data[1]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePackedSecond(&data[2]));
				HK_TEST_ASSERT(0x64211c2f, half8.storePackedSecond(&data[3]));
			}
#endif
		}

#endif

	} // 100 times

#if !defined(HK_REAL_IS_DOUBLE) && !defined(HK_HALF_IS_FLOAT)
	{
		HK_ALIGN_REAL(hkUint32 a[4]) = { 0x12345678, 0x43215678, 0x00005678, 0x87655678 };
		hkVector4* x = (hkVector4*)a;

		x->reduceToHalfPrecision();

		HK_TEST(a[0] == 0x12340000);
		HK_TEST(a[1] == 0x43210000);
		HK_TEST(a[2] == 0x00000000);
		HK_TEST(a[3] == 0x87650000);
	}
#endif
}


static void vector_unimplemented()
{

	// TODO:
	// Accuracy asserts on reciprocal and invsqrt. (produce a binary rep of a sqrt and do a binary compare). Allow difference in the last bit.


	// Multiple component copy
// 	{ todo
// 		hkVector4 vA, vB, vC, vD;
// 		vA.set(1.0f, 2.0f, 3.0f, 4.0f);
// 		vB.set(8.0f, 9.0f, 10.0f, 11.0f);
//
// 		vC = vB;	vC(0) = 5.0f;	vC(1) = 6.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(1) = 6.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XY>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(0) = 5.0f;	vC(2) = 6.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(2) = 6.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XZ>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(0) = 5.0f;	vC(3) = 6.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(3) = 6.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XW>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(1) = 5.0f;	vC(2) = 6.0f;
// 		vD = vA;	vD(1) = 5.0f;	vD(2) = 6.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_YZ>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(1) = 5.0f;	vC(3) = 6.0f;
// 		vD = vA;	vD(1) = 5.0f;	vD(3) = 6.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_YW>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(2) = 5.0f;	vC(3) = 6.0f;
// 		vD = vA;	vD(2) = 5.0f;	vD(3) = 6.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_ZW>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(0) = 5.0f;	vC(1) = 6.0f;	vC(2) = 7.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(1) = 6.0f;	vD(2) = 7.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XYZ>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(0) = 5.0f;	vC(1) = 6.0f;	vC(3) = 7.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(1) = 6.0f;	vD(3) = 7.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XYW>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(0) = 5.0f;	vC(2) = 6.0f;	vC(3) = 7.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(2) = 6.0f;	vD(3) = 7.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XZW>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(1) = 5.0f;	vC(2) = 6.0f;	vC(3) = 7.0f;
// 		vD = vA;	vD(1) = 5.0f;	vD(2) = 6.0f;	vD(3) = 7.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_YZW>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(0) = 5.0f;	vC(1) = 6.0f;	vC(2) = 7.0f;	vC(3) = -1.0f;
// 		vD = vA;	vD(0) = 5.0f;	vD(1) = 6.0f;	vD(2) = 7.0f;	vD(3) = -1.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_XYZW>(vC);
// 		HK_TEST(v.equals4(vD));
// 	}

	// Test component copy
// 	{ todo
// 		hkVector4 vA, vB, vC, vD;
// 		vA.set(1.0f, 2.0f, 3.0f, 4.0f);
// 		vB.set(8.0f, 9.0f, 10.0f, 11.0f);
//
// 		vC = vB;	vC(0) = 5.0f;
// 		vD = vA;	vD(0) = 5.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_X>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(1) = 5.0f;
// 		vD = vA;	vD(1) = 5.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_Y>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(2) = 5.0f;
// 		vD = vA;	vD(2) = 5.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_Z>(vC);
// 		HK_TEST(v.equals4(vD));
//
// 		vC = vB;	vC(3) = 5.0f;
// 		vD = vA;	vD(3) = 5.0f;
// 		v = vA;		v.setComponent<hkVector4ComparisonMask::MASK_W>(vC);
// 		HK_TEST(v.equals4(vD));
// 	}
}

//
//	Tests findComponent

static void vector_find_component()
{
	for (int i = 0; i < 16; i++)
	{
		// Init vector
		hkVector4 v;
		v(0) = (i & 1) ? hkReal(1) : hkReal(0);
		v(1) = (i & 2) ? hkReal(1) : hkReal(0);
		v(2) = (i & 4) ? hkReal(1) : hkReal(0);
		v(3) = (i & 8) ? hkReal(1) : hkReal(0);

		// Get value index
		int idxA = v.findComponent<3>(hkSimdReal::getConstant<HK_QUADREAL_1>());
		HK_TEST((idxA >= -1) && (idxA <= 2));

		int idxB = -1;
		for (int k = 0; k < 3; k++)
		{
			if ( v(k) == hkReal(1) )
			{
				idxB = k;
				break;
			}
		}

		HK_TEST(idxA == idxB);
	}
}

namespace hkMath
{
	static hkQuadReal HK_CALL quadReciprocalTwoIter( const hkQuadReal& r );
	static hkQuadReal HK_CALL quadReciprocalSquareRootTwoIter( const hkQuadReal& r );
}

#if (HK_ENDIAN_BIG)
#define _iexponent_ 0
#else
#define _iexponent_ 1
#endif

HK_FORCE_INLINE static hkDouble64 preciseInvSqrt(const hkDouble64 y)
{
	hkDouble64 x, z;
	union {
		hkDouble64 tempd;
		hkUint32   tempi[2];
	} t;

	t.tempd = y;
	t.tempi[_iexponent_] = (0xbfcdd90a - t.tempi[_iexponent_])>>1; // estimate of 1/sqrt(y)
	x =  t.tempd;
	z =  y*0.5;                        // hoist out the '/2'
	x = (1.5*x) - (x*x)*(x*z);         // iteration formula
	x = (1.5*x) - (x*x)*(x*z);
	x = (1.5*x) - (x*x)*(x*z);
	x = (1.5*x) - (x*x)*(x*z);
	x = (1.5*x) - (x*x)*(x*z);			// 5 NR give 51 Bit
	return x;
}
#undef _iexponent_

static void vector_accuracy_tests()
{
	const int COUNT = 10000;
	{
		double avgErrorSqrt = 0;
		double avgErrorSqrt23 = 0;
		double avgErrorSqrt12 = 0;
		double avgErrorInvSqrt = 0;
		double avgErrorInvSqrt23 = 0;
		double avgErrorInvSqrt12 = 0;

		double absMaxErrorSqrt = 0;
		double absMaxErrorSqrt23 = 0;
		double absMaxErrorSqrt12 = 0;
		double absMaxErrorInvSqrt = 0;
		double absMaxErrorInvSqrt23 = 0;
		double absMaxErrorInvSqrt12 = 0;

		hkPseudoRandomGenerator random(10);
		for (int i =0; i < COUNT; i++)
		{
			const hkReal value = random.getRandRange(hkReal(0.000001f), hkReal(i));
			const double referenceInv = preciseInvSqrt( double(value) );
			const double reference = 1.0 / referenceInv;
			hkSimdReal sv; sv.setFromFloat(value);

			// full accuracy - sqrt
			{
				hkSimdReal ssqrt = sv.sqrt<HK_ACC_FULL,HK_SQRT_IGNORE>();
				double error = double(ssqrt.getReal()) - reference;
				double rError = hkMath::abs(error * referenceInv);
				avgErrorSqrt += rError;
				absMaxErrorSqrt = hkMath::max2(absMaxErrorSqrt, rError);
			}

			// sqrt 23 bit accuracy
			{
				hkSimdReal ssqrt = sv.sqrt<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
				double error = double(ssqrt.getReal()) - reference;
				double rError = hkMath::abs(error * referenceInv);
				avgErrorSqrt23 += rError;
				absMaxErrorSqrt23 = hkMath::max2(absMaxErrorSqrt23, rError);
			}

			// sqrt 12 bit accuracy
			{
				hkSimdReal ssqrt = sv.sqrt<HK_ACC_12_BIT,HK_SQRT_IGNORE>();
				double error = double(ssqrt.getReal()) - reference;
				double rError = hkMath::abs(error * referenceInv);
				avgErrorSqrt12 += rError;
				absMaxErrorSqrt12 = hkMath::max2(absMaxErrorSqrt12, rError);
			}

			// full accuracy inv sqrt
			{
				hkSimdReal ssqrt = sv.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>();
				double error = double(ssqrt.getReal()) - referenceInv;
				double rError = hkMath::abs(error * reference);
				avgErrorInvSqrt += rError;
				absMaxErrorInvSqrt = hkMath::max2(absMaxErrorInvSqrt, rError);
			}

			// inv sqrt 23 bit accuracy
			{
				hkSimdReal ssqrt = sv.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
				double error = double(ssqrt.getReal()) - referenceInv;
				double rError = hkMath::abs(error * reference);
				avgErrorInvSqrt23 += rError;
				absMaxErrorInvSqrt23 = hkMath::max2( absMaxErrorInvSqrt23, rError );
			}

			// inv sqrt 12 bit accuracy
			{
				hkSimdReal ssqrt = sv.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
				double error = double(ssqrt.getReal()) - referenceInv;
				double rError = hkMath::abs(error * reference);
				avgErrorInvSqrt12 += rError;
				absMaxErrorInvSqrt12 = hkMath::max2( absMaxErrorInvSqrt12, rError );
			}
		}

		const double cnt = 1.0 / double(COUNT);
		avgErrorSqrt *= cnt;
		avgErrorSqrt23 *= cnt;
		avgErrorSqrt12 *= cnt;
		avgErrorInvSqrt *= cnt;
		avgErrorInvSqrt23 *= cnt;
		avgErrorInvSqrt12 *= cnt;

#if !defined(HK_REAL_IS_DOUBLE)
		// cannot compute ground truth with same precision as approximation
		HK_TEST(avgErrorSqrt <= avgErrorSqrt23);
		HK_TEST(absMaxErrorSqrt <= absMaxErrorSqrt23);
#if !defined(HK_PLATFORM_PS3) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		HK_TEST(avgErrorInvSqrt <= avgErrorInvSqrt23);
		HK_TEST(absMaxErrorInvSqrt <= absMaxErrorInvSqrt23);
#endif
#endif

		// check approximations are sane
		HK_TEST(avgErrorSqrt23 <= avgErrorSqrt12);
		HK_TEST(absMaxErrorSqrt23 <= absMaxErrorSqrt12);
		HK_TEST(avgErrorInvSqrt23 <= avgErrorInvSqrt12);
		HK_TEST(absMaxErrorInvSqrt23 <= absMaxErrorInvSqrt12);
	}
#if 0
	//TODO
	// Testing double iterations for inv
	{
		double avgError = 0;
		double avgError23 = 0;
		double avgError2Iter = 0;
		double maxError = 0;
		double maxError23 = 0;
		double maxError2Iter = 0;

		hkPseudoRandomGenerator random(10);
		for (int i = 0; i < COUNT; i++)
		{
			hkReal value = random.getRandRange(0.000001f, hkReal(i));
			double correct = 1.0 / double(value);

			// full accuracy sqrt
			{
				hkSimdReal result; result.setFromFloat(1.0f);
				hkSimdReal sv; sv.setFromFloat(value);
				result.div(sv);
				double error = result.getReal() - correct;
				double rError = hkMath::fabs(error)/correct;
				avgError += rError;
				maxError = hkMath::max2( maxError, rError );
			}

			// 23 bit accuracy sqrt
			{
				hkVector4 result; result.setAll(value);
				result.m_quad = hkMath::quadReciprocal(result.m_quad);
				double error = result.getComponent<0>().getReal() - correct;
				double rError = hkMath::fabs(error)/correct;
				avgError23 += rError;
				maxError23 = hkMath::max2( maxError23, rError );
			}

			// 2iter accuracy sqrt
			{
				hkVector4 result; result.setAll(value);
				result.m_quad = hkMath::quadReciprocalTwoIter(result.m_quad);

				double error = result.getComponent<0>().getReal() - correct;
				double rError = hkMath::fabs(error)/correct;
				avgError2Iter += rError;
				maxError2Iter = hkMath::max2( maxError2Iter, rError );
			}
		}

		avgError /= COUNT;
		avgError23 /= COUNT;
		avgError2Iter /= COUNT;

		//HK_TEST(avgError <= avgError2Iter);
		//HK_TEST(avgError2Iter <= avgError23);

		//HK_TEST(maxError <= maxError2Iter);
		//HK_TEST(maxError2Iter <= maxError23);
	}

	// testing double iter for inv square root
	{

		double avgError = 0;
		double avgError23 = 0;
		double avgError2Iter = 0;
		double maxError = 0;
		double maxError23 = 0;
		double maxError2Iter = 0;

		hkPseudoRandomGenerator random(10);
		for (int i = 0; i < COUNT; i++)
		{
			hkReal value = random.getRandRange(0.000001f, hkReal(i));
			double correct = preciseSqrt( double(value) );

			// full accuracy recip sqrt
			{
				hkSimdReal input; input.setFromFloat(value);
				hkSimdReal result = input.sqrtInverse() * input;

				double error = result.getReal() - correct;
				double rError = hkMath::fabs(error)/correct;
				avgError += rError;
				maxError = hkMath::max2( maxError, rError );
			}

			// 23 bit accuracy recip sqrt
			{
				hkSimdReal input; input.setFromFloat(value);
				hkSimdReal result = input.sqrtInverse_23BitAccurate() * input;
				double error = result.getReal() - correct;
				double rError = hkMath::fabs(error)/correct;
				avgError23 += rError;
				maxError23 = hkMath::max2( maxError23, rError );
			}

			// 2iter accuracy recip sqrt
			{
				hkVector4 input; input.setAll(value);
				hkVector4 result;
				result.m_quad = hkMath::quadReciprocalSquareRootTwoIter(input.m_quad);
				result.mul(input);

				double error = result.getComponent<0>().getReal() - correct;
				double rError = hkMath::fabs(error)/correct;
				avgError2Iter += rError;
				maxError2Iter = hkMath::max2( maxError2Iter, rError );
			}
		}

		avgError /= COUNT;
		avgError23 /= COUNT;
		avgError2Iter /= COUNT;

		//HK_TEST(avgErrorSqrt <= avgError2Iter);
		//HK_TEST(avgError2Iter <= avgError23);

		//HK_TEST(maxError <= maxError2Iter);
		//HK_TEST(maxError2Iter <= maxError23);
	}
#endif
}

namespace hkFloat16Test
{

hkUint32 g_mantissatable[2048];
hkUint32 g_exponenttable[64];
hkUint16 g_offsettable[64];
hkUint16 g_basetable[512];
hkUint8 g_shifttable[512];

static hkUint32 convertmantissa(hkUint32 i)
{
	hkUint32 m = i<<13; // Zero pad mantissa bits
	hkUint32 e = 0; // Zero exponent
	while(!(m&0x00800000))
	{ // While not normalized
		e -= 0x00800000; // Decrement exponent (1<<23)
		m <<= 1; // Shift mantissa
	}
	m &= ~0x00800000; // Clear leading 1 bit
	e += 0x38800000; // Adjust bias ((127-14)<<23)
	return m | e; // Return combined number
}

static void setupFloat16Tables()
{
	hkUint32 i;

	g_mantissatable[0] = 0;
	for (i=1; i<=1023; ++i) g_mantissatable[i] = convertmantissa(i);
	for (i=1024; i<=2047; ++i) g_mantissatable[i] = 0x38000000 + ((i-1024)<<13);

	g_exponenttable[0] = 0;
	for (i=1; i<=30; ++i) g_exponenttable[i] = i<<23;
	g_exponenttable[31]= 0x47800000;
	g_exponenttable[32]= 0x80000000;
	for (i=33; i<=62; ++i) g_exponenttable[i] = 0x80000000 + ((i-32)<<23);
	g_exponenttable[63]= 0xC7800000;

	g_offsettable[0] = 0;
	for (i=1; i<=31; ++i) g_offsettable[i] = 1024;
	g_offsettable[32]= 0;
	for (i=33; i<=63; ++i) g_offsettable[i] = 1024;

	for (i=0; i<256; ++i)
	{
		int e = i-127;
		if (e < -24)
		{ // Very small numbers map to zero
			g_basetable[i|0x000] = 0x0000;
			g_basetable[i|0x100] = 0x8000;
			g_shifttable[i|0x000] = 24;
			g_shifttable[i|0x100] = 24;
		}
		else if (e < -14)
		{ // Small numbers map to denorms
			g_basetable[i|0x000] = (0x0400 >> (-e-14));
			g_basetable[i|0x100] = (0x0400 >> (-e-14)) | 0x8000;
			g_shifttable[i|0x000] = hkUint8(-e-1);
			g_shifttable[i|0x100] = hkUint8(-e-1);
		}
		else if (e <= 15)
		{ // Normal numbers just lose precision
			g_basetable[i|0x000] = hkUint16((e+15)<<10);
			g_basetable[i|0x100] = hkUint16(((e+15)<<10) | 0x8000);
			g_shifttable[i|0x000] = 13;
			g_shifttable[i|0x100] = 13;
		}
		else if (e < 128)
		{ // Large numbers map to Infinity
			g_basetable[i|0x000] = 0x7C00;
			g_basetable[i|0x100] = 0xFC00;
			g_shifttable[i|0x000] = 24;
			g_shifttable[i|0x100] = 24;
		}
		else
		{ // Infinity and NaN's stay Infinity and NaN's
			g_basetable[i|0x000] = 0x7C00;
			g_basetable[i|0x100] = 0xFC00;
			g_shifttable[i|0x000] = 13;
			g_shifttable[i|0x100] = 13;
		}
	}
}

union i2f { hkUint32 i; hkFloat32 f; };

static hkFloat32 hkFloat16_conv_16_32(hkFloat16 h)
{
	i2f c;
	const hkUint16 expIndex = h.getBits() >> 10;
	const hkUint16 exponent = h.getBits() & 0x3ff;
	c.i = g_mantissatable[g_offsettable[expIndex] + exponent] + g_exponenttable[expIndex];
	return c.f;
}


static hkFloat16 hkFloat16_conv_32_16(hkFloat32 f)
{
	i2f c;
	c.f = f;
	const hkUint32 expIndex = (c.i >> 23) & 0x1ff;
	const hkUint32 mantissa = c.i & 0x007fffff;
	hkFloat16 f16; f16.setBits( (hkUint16)( g_basetable[expIndex] + (mantissa >> g_shifttable[expIndex]) ) );
	return f16;
}
} // namespace

static void vector_float16()
{
	hkFloat16Test::setupFloat16Tables();

	// check FPU and SIMD produce reference output for good numbers
	{
		HK_ALIGN_REAL(hkFloat16 qa16[4]);

		hkFloat32 a = 0.0f;
		hkFloat32 b = 1.0f;
		hkFloat32 c = -1.0f;
		hkFloat32 d = 123.45f;

		// pack
		hkFloat16 a16; a16.setReal<false>(a);
		hkFloat16 b16; b16.setReal<false>(b);
		hkFloat16 c16; c16.setReal<false>(c);
		hkFloat16 d16; d16.setReal<false>(d);

		hkFloat16 ref_a16 = hkFloat16Test::hkFloat16_conv_32_16(a);
		hkFloat16 ref_b16 = hkFloat16Test::hkFloat16_conv_32_16(b);
		hkFloat16 ref_c16 = hkFloat16Test::hkFloat16_conv_32_16(c);
		hkFloat16 ref_d16 = hkFloat16Test::hkFloat16_conv_32_16(d);

		HK_TEST(a16.getBits() == ref_a16.getBits());
		HK_TEST(b16.getBits() == ref_b16.getBits());
		HK_TEST(c16.getBits() == ref_c16.getBits());
		HK_TEST(d16.getBits() == ref_d16.getBits());

		hkVector4 qa; qa.set(a,b,c,d);
		qa.store<4,HK_IO_SIMD_ALIGNED,HK_ROUND_TRUNCATE>(qa16);

		HK_TEST(qa16[0].getBits() == ref_a16.getBits());
		HK_TEST(qa16[1].getBits() == ref_b16.getBits());

		// Compiler issue, to revisit later. Works in fulldebug on PlayStation(R)3 SNC, fails in debug/release.
#if !defined(HK_COMPILER_SNC)
		HK_TEST(qa16[2].getBits() == ref_c16.getBits());
		HK_TEST(qa16[3].getBits() == ref_d16.getBits());
#endif

		// unpack
		hkFloat32 a32 = hkFloat32(a16.getReal());
		hkFloat32 b32 = hkFloat32(b16.getReal());
		hkFloat32 c32 = hkFloat32(c16.getReal());
		hkFloat32 d32 = hkFloat32(d16.getReal());

		hkFloat32 ref_a32 = hkFloat16Test::hkFloat16_conv_16_32(ref_a16);
		hkFloat32 ref_b32 = hkFloat16Test::hkFloat16_conv_16_32(ref_b16);

		hkFloat32 ref_c32 = hkFloat16Test::hkFloat16_conv_16_32(ref_c16);
		hkFloat32 ref_d32 = hkFloat16Test::hkFloat16_conv_16_32(ref_d16);

		HK_TEST(a32 == ref_a32);
		HK_TEST(b32 == ref_b32);
		HK_TEST(c32 == ref_c32);
		HK_TEST(d32 == ref_d32);

		hkVector4 qa32; qa32.load<4,HK_IO_SIMD_ALIGNED>(qa16);

		HK_TEST(qa32(0) == ref_a32);
		HK_TEST(qa32(1) == ref_b32);
		HK_TEST(qa32(2) == ref_c32);
		HK_TEST(qa32(3) == ref_d32);
	}

	// check FPU and SIMD produce same output for nasty numbers
	{
		HK_ALIGN_REAL(const hkUint32 n[4]) = { 0x80000000, 0x48127C00, 0xC664A800, 0xB8D1B717 }; // -0.0f, 150000.0f, -14634.0f, -0.0001f
		HK_ALIGN_REAL(hkFloat16 qa16[4]);

		const hkFloat32* a = (const hkFloat32*)n;
		const hkFloat32* b = (const hkFloat32*)n+1;
		const hkFloat32* c = (const hkFloat32*)n+2;
		const hkFloat32* d = (const hkFloat32*)n+3;

		qa16[0].setReal<true>(*a);
		qa16[1].setReal<true>(*b);
		qa16[2].setReal<true>(*c);
		qa16[3].setReal<true>(*d);

		hkFloat32 a32 = hkFloat32(qa16[0].getReal());
#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_IOS_SIM)
		hkFloat32 b32 = hkFloat32(qa16[1].getReal());
#endif
		hkFloat32 c32 = hkFloat32(qa16[2].getReal());
		hkFloat32 d32 = hkFloat32(qa16[3].getReal());

		hkVector4 qa; qa.set(*a,*b,*c,*d);
		qa.store<4,HK_IO_SIMD_ALIGNED,HK_ROUND_NEAREST>(qa16);
		hkVector4 qa32;
		qa32.load<4>(qa16);

		HK_TEST(qa32(0) == a32);
#if !defined(HK_COMPILER_HAS_INTRINSICS_NEON) && !defined(HK_PLATFORM_IOS_SIM)
		
		
		
		HK_TEST(qa32(1) == b32);
#endif
		HK_TEST(qa32(2) == c32);
		HK_TEST(qa32(3) == d32);
	}

	// Skip this test on SPU to avoid running out of space on the unit test elf.
#if !defined(HK_PLATFORM_SPU)

	// Test native alignment
	{
		hkVector4 vIn; vIn.set(1, 2, 3, 4);
		HK_ALIGN16(hkFloat16 f16[16]);

		// Test all different offsets within a quad word
		for (int offset = 0; offset < 8; ++offset)
		{
			{
				const int N = 1;

				// Test store
				hkString::memClear16(f16, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(f16 + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(f16[i].getBits() == 0); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(f16[i].getBits() == 0); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(f16 + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}

			{
				const int N = 2;

				// Test store
				hkString::memClear16(f16, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(f16 + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(f16[i].getBits() == 0); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(f16[i].getBits() == 0); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(f16 + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}

			{
				const int N = 3;

				// Test store
				hkString::memClear16(f16, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(f16 + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(f16[i].getBits() == 0); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(f16[i].getBits() == 0); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(f16 + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}

			{
				const int N = 4;

				// Test store
				hkString::memClear16(f16, 2);
				vIn.store<N, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(f16 + offset);
				for (int i = 0; i < offset; ++i) { HK_TEST(f16[i].getBits() == 0); }
				for (int i = offset + N; i < 16; ++i) { HK_TEST(f16[i].getBits() == 0); }

				// Test load
				hkVector4 vOut; vOut.load<N, HK_IO_NATIVE_ALIGNED>(f16 + offset);
				for (int k = 0; k < N; k++)
				{
					HK_TEST(hkMath::equal(vIn(k), vOut(k), hkMath::fabs(vIn(k)) / 255.f));
				}
			}
		}
	}
#endif
}

//
//	Tests hkVector4Comparison::horizontalOr

template <int N>
static void vector4comparison_horizontalOr()
{
	static const hkVector4ComparisonMask::Mask allMasks[5] =
	{
		hkVector4ComparisonMask::MASK_NONE, hkVector4ComparisonMask::MASK_X, hkVector4ComparisonMask::MASK_XY, hkVector4ComparisonMask::MASK_XYZ, hkVector4ComparisonMask::MASK_XYZW,
	};
	const hkUint32 M = allMasks[N];

	for (int k = hkVector4ComparisonMask::MASK_NONE; k <= hkVector4ComparisonMask::MASK_XYZW; k++)
	{
		hkVector4Comparison mask;		mask.set((hkVector4ComparisonMask::Mask)k);
		const hkUint32 maskVal			= mask.getMask();
		const hkVector4Comparison hor	= mask.horizontalOr<N>();
		const hkUint32 actualVal		= hor.getMask();
		const hkUint32 expectedVal		= (maskVal & M) ? hkVector4ComparisonMask::MASK_XYZW : hkVector4ComparisonMask::MASK_NONE;

		HK_TEST(actualVal == expectedVal);
	}
}

static void hkVector4Comparison_horizontalOr()
{
	vector4comparison_horizontalOr<1>();
	vector4comparison_horizontalOr<2>();
	vector4comparison_horizontalOr<3>();
	vector4comparison_horizontalOr<4>();
}

int vector3_main()
{
	{
		vector_assign_basic();

		vector_vector4comparisons();
	    vector_vector4comparisons2();
		vector_vector4comparisons3();
		vector_vector4comparisons4();
		vector_comparisons();

		vector_ops();
		vector_assign_special();
		matrix3_transform_quaternion();
		vector_dots_lengths();
		vector_square_roots();
		vector_broadcast();
		vector_shuffle();
		vector_accessors();
		vector_length0();
		vector_load_store();
		vector_getset_int24w();
		vector_packed();
		vector_float16();
		vector_half8();
		vector_find_component();

		vector_unimplemented();

		vector_accuracy_tests();
		hkVector4Comparison_horizontalOr();
	}
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(vector3_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/vector3.cpp"     );

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
