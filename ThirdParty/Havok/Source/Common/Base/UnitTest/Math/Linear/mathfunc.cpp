/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <math.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

static inline hkBool approx_equals(hkReal x, hkReal y)
{
	const hkReal tol = 1e-4f;
	hkReal f = x - y;
	if(-tol < f && f < tol)
	{
		return true;
	}
	else
	{
		//printf("values were %f, %f\n", x, y);
		return false;
	}
}

extern hkReal mathfunc_test_zero;

static void mathfunc_test()
{
	{
        // try to fool the compilers div by zero check
		hkReal y = 1.0f / mathfunc_test_zero;
		HK_TEST(!hkMath::isFinite(y));
	}

	const int NLOOP = 10;

	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		for(int i=0; i<NLOOP*NLOOP; ++i)
		{
			hkReal r = rng.getRandRange(0,HK_REAL_MAX);
			HK_TEST( hkMath::isFinite(r) );
		}
	}

	{
		for(int i = 0; i < NLOOP; ++i)
		{
			hkReal r = i - (0.5f*NLOOP);
			hkReal fabsR = hkMath::fabs(r);
			if( r >= 0)
			{
				HK_TEST( fabsR == r );
			}
			else
			{
				HK_TEST( fabsR == -r );
			}
		}
	}

	const hkReal tolerance = 1e-6f;

	HK_TEST( hkMath::equal( hkMath::acos(hkReal( 1.00001f)), hkReal(0), tolerance) );
	HK_TEST( hkMath::equal( hkMath::acos(hkReal(-1.00001f)), HK_REAL_PI, tolerance) );

	HK_TEST( hkMath::equal( hkMath::asin(hkReal( 1.00001f)),  0.5f * HK_REAL_PI, tolerance) );
	HK_TEST( hkMath::equal( hkMath::asin(hkReal(-1.00001f)), -0.5f * HK_REAL_PI, tolerance) );

#if !defined(HK_REAL_IS_DOUBLE)

		// Now test the "closest" values to 1 and -1
	HK_COMPILE_TIME_ASSERT(sizeof(hkReal) == sizeof(unsigned int));
	union realIntUnion{hkReal			m_real;
						unsigned int	m_int;};

	{
		
		realIntUnion smallestGreaterThanOne;
		smallestGreaterThanOne.m_int = 0x3F800001;	// This is the hex IEEE representation of 1.00000012. See http://www.markworld.com/scripts/showfloat.exe

		HK_TEST( hkMath::equal( hkMath::acos(smallestGreaterThanOne.m_real), 0.0f, tolerance) );
		HK_TEST( hkMath::equal( hkMath::asin(smallestGreaterThanOne.m_real),  0.5f * HK_REAL_PI, tolerance) );

	}

	{
		realIntUnion smallestLessThanMinusOne;
		smallestLessThanMinusOne.m_int = 0xBF800001;	// This is the hex IEEE representation of -1.00000012. See http://www.markworld.com/scripts/showfloat.exe

		HK_TEST( hkMath::equal( hkMath::acos(smallestLessThanMinusOne.m_real), HK_REAL_PI, tolerance) );
		HK_TEST( hkMath::equal( hkMath::asin(smallestLessThanMinusOne.m_real), -0.5f * HK_REAL_PI, tolerance) );
	}
#endif

	/*
	{
		for(int i=0; i<NLOOP; ++i)
		{
			hkReal r = i - (0.5f*NLOOP);
			hkReal sinR = hkMath::sin(r);
			HK_TEST( approx_equals(::sin(r), sinR) );
		}
	}

	{
		for(int i=0; i<NLOOP; ++i)
		{
			hkReal r = i - (0.5f*NLOOP);
			hkReal s = hkMath::cos(r);
			HK_TEST( approx_equals(::cos(r), s) );
		}
	}

	{
		for(int i=0; i<NLOOP; ++i)
		{
			hkReal r = (2*hkReal(i) - NLOOP) / NLOOP;
			hkReal s = hkMath::acos(r);
			HK_TEST( approx_equals(::acos(r), s) );
		}
	}

	{
		for(int i=0; i<NLOOP; ++i)
		{
			hkReal r = (2*hkReal(i) - NLOOP) / NLOOP;
			hkReal s = hkMath::asin(r);
			HK_TEST( approx_equals(::asin(r), s) );
		}
	}
	*/
	

	{
		for(int i=0; i<NLOOP; ++i)
		{
			hkReal r = hkReal(i+1);
			hkReal s = hkMath::sqrtInverse(r);
			HK_TEST( approx_equals(s*s*r, 1) );
		}
	}

	// Test sqrt around 1
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');

		for(int i=0; i<NLOOP; ++i)
		{
			for (int j=0; j < NLOOP; j++)
			{
				hkReal r = hkReal(rng.getRandRange(0,1));
				hkReal s = hkMath::sqrt(r);
				hkReal diff = hkMath::fabs(s*s)/r;
				HK_TEST( approx_equals(diff, 1.0f));
			}
		}
	}

	// Test sqrt around larger ranges
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');

		for(int i=0; i<NLOOP; ++i)
		{
			for (int j=0; j < NLOOP; j++)
			{
				hkReal r = hkReal(rng.getRandRange(0,1000000));
				hkReal s = hkMath::sqrt(r);
				hkReal diff = hkMath::fabs(s*s)/r;
				HK_TEST( approx_equals(diff, 1.0f));
			}
		}
	}

	// Test Sqrt 0
	{
		hkReal r = 0.0f;
		hkReal s = hkMath::sqrt(r);
		HK_TEST(approx_equals(s,0));
	}

	// Test recip Sqrt 0
	{
		hkReal r = 0.0f;
		hkReal s = hkMath::sqrtInverse(r);
		HK_TEST( !hkMath::isFinite(s) );
	}

	// Test Sqrt of -ve
	{
		hkReal r = -1.0f;
		hkReal s = hkMath::sqrt(r);
		HK_TEST(!hkMath::isFinite(s));
	}

	// Test recip Sqrt of -ve
	{
		hkReal r = -1.0f;
		hkReal s = hkMath::sqrtInverse(r);
		HK_TEST( !hkMath::isFinite(s) );
	}

	// Test results of hkMath::fselect*
	{
		HK_TEST(hkMath::fselectEqualZero( 1.0f, 2.0f, 3.0f) == 3.0f);
		HK_TEST(hkMath::fselectEqualZero( 0.0f, 2.0f, 3.0f) == 2.0f);
		HK_TEST(hkMath::fselectEqualZero(-1.0f, 2.0f, 3.0f) == 3.0f);

		HK_TEST(hkMath::fselectGreaterEqualZero( 1.0f, 2.0f, 3.0f) == 2.0f );
		HK_TEST(hkMath::fselectGreaterEqualZero( 0.0f, 2.0f, 3.0f) == 2.0f );
		HK_TEST(hkMath::fselectGreaterEqualZero(-1.0f, 2.0f, 3.0f) == 3.0f );

		HK_TEST(hkMath::fselectGreaterZero( 1.0f, 2.0f, 3.0f) == 2.0f );
		HK_TEST(hkMath::fselectGreaterZero( 0.0f, 2.0f, 3.0f) == 3.0f );
		HK_TEST(hkMath::fselectGreaterZero(-1.0f, 2.0f, 3.0f) == 3.0f );
	}

	// Test clamp
	{
		hkReal mone = hkReal(-1);
		hkReal mtwo = hkReal(-2);
		hkReal one = hkReal(1);
		hkReal two = hkReal(2);
		hkReal f = hkReal(0.252f);

		HK_TEST(hkMath::clamp(mtwo, mone, one) == mone);
		HK_TEST(hkMath::clamp(mone, mone, one) == mone);
		HK_TEST(hkMath::clamp(f, mone, one) == f);
		HK_TEST(hkMath::clamp(f, mtwo, two) == f);
		HK_TEST(hkMath::clamp(one, mtwo, two) == one);
		HK_TEST(hkMath::clamp(two, mtwo, one) == one);
	}
}

hkReal mathfunc_test_zero = 0;

union fi
{
	hkReal f;
#if defined( HK_REAL_IS_DOUBLE )
	hkUint64 i;
#else
	hkUint32 i;
#endif
};

// types_test() assumes 'precise' float control for non simd mode,
// and can fail if model is set to 'fast'. (See issue COM-1673)
#ifdef HK_PLATFORM_WIN32
#if (HK_CONFIG_SIMD==HK_CONFIG_SIMD_DISABLED)
#pragma float_control(precise, on) 
#pragma float_control(except, on) 
#endif
#endif
static void types_test()
{
	fi test1;
	fi test2;
	fi test3;
	fi test4;
	fi test5;
	fi test6;

#if defined( HK_REAL_IS_DOUBLE )
	test1.i = 0x4000000000000000ull;
	test2.i = 0x401A000000000000ull;
	test3.i = 0xC01A000000000000ull;
	test4.i = 0x3810000000000000ull;
	test5.i = 0x0080000000000000ull;
	test6.i = 0x0000000000000001ull;
#else
	test1.i = 0x40000000;
	test2.i = 0x40D00000;
	test3.i = 0xC0D00000;
	test4.i = 0x00800000;
	test5.i = 0x00400000;
	test6.i = 0x00000001;
#endif

	hkReal o = hkReal(-0.0004f);
	hkReal p = hkReal(-1.1f);
	hkReal q = hkReal(10000000000.0f);
	hkReal r = hkReal(-1.0f);
	hkReal s = hkReal(0.5f);
	hkReal t = hkReal(-9.29999f);
	hkReal u = hkReal(-16646142.000000f);
	hkReal v = hkReal(0.0000f);
	hkReal w = hkReal(-0.0f);
	hkReal x = hkReal(9.32345f);
	hkReal y = hkReal(1.0f);
	hkReal z = hkReal(-0.5f);

	int o1 = hkMath::hkFloatToInt(o);
	int p1 = hkMath::hkFloatToInt(p);
	//int q1 = hkMath::hkFloatToInt(q);
	int r1 = hkMath::hkFloatToInt(r);
	int s1 = hkMath::hkFloatToInt(s);
	int t1 = hkMath::hkFloatToInt(t);
	int u1 = hkMath::hkFloatToInt(u);
	int v1 = hkMath::hkFloatToInt(v);
	int w1 = hkMath::hkFloatToInt(w);
	int x1 = hkMath::hkFloatToInt(x);
	int y1 = hkMath::hkFloatToInt(y);
	int z1 = hkMath::hkFloatToInt(z);
	int test11 = hkMath::hkFloatToInt(test1.f);
	int test12 = hkMath::hkFloatToInt(test2.f);
	int test13 = hkMath::hkFloatToInt(test3.f);
	int test14 = hkMath::hkFloatToInt(test4.f);
	int test15 = hkMath::hkFloatToInt(test5.f);
	int test16 = hkMath::hkFloatToInt(test6.f);

	HK_TEST(o1 == (int)o);
	HK_TEST(p1 == (int)p);
	//this test fails as the default implementation
	//is not as sturdy on Win32!
	//HK_TEST(q1 == (int)q);
	HK_TEST(r1 == (int)r);
	HK_TEST(s1 == (int)s);
	HK_TEST(t1 == (int)t);
	HK_TEST(u1 == (int)u);
	HK_TEST(v1 == (int)v);
	HK_TEST(w1 == (int)w);
	HK_TEST(x1 == (int)x);
	HK_TEST(y1 == (int)y);
	HK_TEST(z1 == (int)z);
	HK_TEST(test11 == (int)test1.f);
	HK_TEST(test12 == (int)test2.f);
	HK_TEST(test13 == (int)test3.f);
	HK_TEST(test14 == (int)test4.f);
	HK_TEST(test15 == (int)test5.f);
	HK_TEST(test16 == (int)test6.f);

	hkReal o2 = hkMath::hkFloor(o);
	hkReal p2 = hkMath::hkFloor(p);
	hkReal q2 = hkMath::hkFloor(q);
	hkReal r2 = hkMath::hkFloor(r);
	hkReal s2 = hkMath::hkFloor(s);
	hkReal t2 = hkMath::hkFloor(t);
	hkReal u2 = hkMath::hkFloor(u);
	hkReal v2 = hkMath::hkFloor(v);
	hkReal w2 = hkMath::hkFloor(w);
	hkReal x2 = hkMath::hkFloor(x);
	hkReal y2 = hkMath::hkFloor(y);
	hkReal z2 = hkMath::hkFloor(z);
	hkReal test21 = hkMath::hkFloor(test1.f);
	hkReal test22 = hkMath::hkFloor(test2.f);
	hkReal test23 = hkMath::hkFloor(test3.f);
	hkReal test24 = hkMath::hkFloor(test4.f);
	// These aren't used in PlayStation(R)3 SNC fulldebug (see below).
#if !defined(HK_COMPILER_SNC) || !defined(HK_DEBUG)
	hkReal test25 = hkMath::hkFloor(test5.f);
	hkReal test26 = hkMath::hkFloor(test6.f);
#endif

	HK_TEST(o2 == hkMath::floor(o));
	HK_TEST(p2 == hkMath::floor(p));
	HK_TEST(q2 == hkMath::floor(q));
	HK_TEST(r2 == hkMath::floor(r));
	HK_TEST(s2 == hkMath::floor(s));
	HK_TEST(t2 == hkMath::floor(t));
	HK_TEST(u2 == hkMath::floor(u));
	HK_TEST(v2 == hkMath::floor(v));
	HK_TEST(w2 == hkMath::floor(w));
	HK_TEST(x2 == hkMath::floor(x));
	HK_TEST(y2 == hkMath::floor(y));
	HK_TEST(z2 == hkMath::floor(z));
	HK_TEST(test21 == hkMath::floor(test1.f));
	HK_TEST(test22 == hkMath::floor(test2.f));
	HK_TEST(test23 == hkMath::floor(test3.f));
	HK_TEST(test24 == hkMath::floor(test4.f));
	// These two use denormalized numbers and may fail on PlayStation(R)3 SNC in debug due to a compiler issue.
#if !defined(HK_COMPILER_SNC) || !defined(HK_DEBUG)
	HK_TEST(test25 == hkMath::floor(test5.f));
	HK_TEST(test26 == hkMath::floor(test6.f));
#endif

	int o3 = hkMath::hkFloorToInt(o);
	int p3 = hkMath::hkFloorToInt(p);
	//int q3 = hkMath::hkFloorToInt(q);
	int r3 = hkMath::hkFloorToInt(r);
	int s3 = hkMath::hkFloorToInt(s);
	int t3 = hkMath::hkFloorToInt(t);
	int u3 = hkMath::hkFloorToInt(u);
	int v3 = hkMath::hkFloorToInt(v);
	int w3 = hkMath::hkFloorToInt(w);
	int x3 = hkMath::hkFloorToInt(x);
	int y3 = hkMath::hkFloorToInt(y);
	int z3 = hkMath::hkFloorToInt(z);
	int test31 = hkMath::hkFloorToInt(test1.f);
	int test32 = hkMath::hkFloorToInt(test2.f);
	int test33 = hkMath::hkFloorToInt(test3.f);
	int test34 = hkMath::hkFloorToInt(test4.f);
	int test35 = hkMath::hkFloorToInt(test5.f);
	int test36 = hkMath::hkFloorToInt(test6.f);

	HK_TEST(o3 == (int)(hkMath::floor(o)));
	HK_TEST(p3 == (int)(hkMath::floor(p)));
	//this test fails as the default implementation
	//is not as sturdy on PlayStation(R)2!
	//HK_TEST(q3 == (int)(hkMath::floor(q)));
	HK_TEST(r3 == (int)(hkMath::floor(r)));
	HK_TEST(s3 == (int)(hkMath::floor(s)));
	HK_TEST(t3 == (int)(hkMath::floor(t)));
	HK_TEST(u3 == (int)(hkMath::floor(u)));
	HK_TEST(v3 == (int)(hkMath::floor(v)));
	HK_TEST(w3 == (int)(hkMath::floor(w)));
	HK_TEST(x3 == (int)(hkMath::floor(x)));
	HK_TEST(y3 == (int)(hkMath::floor(y)));
	HK_TEST(z3 == (int)(hkMath::floor(z)));
	HK_TEST(test31 == (int)(hkMath::floor(test1.f)));
	HK_TEST(test32 == (int)(hkMath::floor(test2.f)));
	HK_TEST(test33 == (int)(hkMath::floor(test3.f)));
	HK_TEST(test34 == (int)(hkMath::floor(test4.f)));
	HK_TEST(test35 == (int)(hkMath::floor(test5.f)));
	HK_TEST(test36 == (int)(hkMath::floor(test6.f)));
	{
		int count = 0;
		for ( hkReal f = hkReal(1000000000.0f); f > hkReal(-1000000000.0f); f -= hkReal(0.001f) )
		{
			int i = hkMath::hkFloorToInt( f );
			HK_TEST( i == (int)( hkMath::floor( f ) ) );

			if ( ++count == 10 )
			{
				count = 0;
				f -= hkReal(1000000.0f);
			}
		}
	}

	// Check that float2int does truncate rounding
	{
		hkPseudoRandomGenerator rng('h'+'a'+'v'+'o'+'k');
		const int NUM_ITERATIONS = 1000;
		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			int rnd = int(rng.getRandRange(0.0f, 1048576.0f)); // 2^20

			for (hkReal j = 0.0f; j < 1.0f; j += 0.05f)
			{
				int resultFromFloat  = hkMath::hkFloatToInt(hkFloat32(rnd) + j);
				HK_TEST2( rnd == resultFromFloat, "r = " << rnd << ", increment = " << j << ", result = " << resultFromFloat);
				int resultFromFloat2  = hkMath::hkToIntFast(hkFloat32(rnd) + j);
				HK_TEST2( rnd == resultFromFloat2, "r = " << rnd << ", increment = " << j << ", result = " << resultFromFloat2);
			}

			rnd = int(rng.getRandRange(-1048576.0f, 0.0f)); // 2^20

			for (hkReal j = 0.0f; j < 1.0f; j += 0.05f)
			{
				int resultFromFloat  = hkMath::hkFloatToInt(hkFloat32(rnd) - j);
				HK_TEST2( rnd == resultFromFloat, "r = " << rnd << ", increment = " << j << ", result = " << resultFromFloat);
				int resultFromFloat2  = hkMath::hkToIntFast(hkFloat32(rnd) - j);
				HK_TEST2( rnd == resultFromFloat2, "r = " << rnd << ", increment = " << j << ", result = " << resultFromFloat2);
			}
		}
	}
}
#ifdef HK_PLATFORM_WIN32
#if (HK_CONFIG_SIMD==HK_CONFIG_SIMD_DISABLED)
#pragma float_control(except, off) 
#pragma float_control(precise, off) 
#endif
#endif


#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_REAL_IS_DOUBLE) && !defined(HK_ARCH_ARM)
#include <Common/Base/System/Stopwatch/hkStopwatch.h>

static void HK_CALL evalError(double ln_0, double ln_1, double ln_2, double ln_3, 
							 double ref0, double ref1, double ref2, double ref3, 
							 double& aErr, double& mErr)
{
	double err0 = ln_0 - ref0;
	double rErr0 = ::fabs(err0 / ref0);
	aErr += rErr0;
	if (rErr0 > mErr) mErr = rErr0;

	double err1 = ln_1 - ref1;
	double rErr1 = ::fabs(err1 / ref1);
	aErr += rErr1;
	if (rErr1 > mErr) mErr = rErr1;

	double err2 = ln_2 - ref2;
	double rErr2 = ::fabs(err2 / ref2);
	aErr += rErr2;
	if (rErr2 > mErr) mErr = rErr2;

	double err3 = ln_3 - ref3;
	double rErr3 = ::fabs(err3 / ref3);
	aErr += rErr3;
	if (rErr3 > mErr) mErr = rErr3;
}

static void HK_CALL evalError(const __m128& ln,
							 double ref0, double ref1, double ref2, double ref3, 
							 double& aErr, double& mErr)
{
	double err0 = double(ln.m128_f32[0]) - ref0;
	double rErr0 = ::fabs(err0 / ref0);
	aErr += rErr0;
	if (rErr0 > mErr) mErr = rErr0;

	double err1 = double(ln.m128_f32[1]) - ref1;
	double rErr1 = ::fabs(err1 / ref1);
	aErr += rErr1;
	if (rErr1 > mErr) mErr = rErr1;

	double err2 = double(ln.m128_f32[2]) - ref2;
	double rErr2 = ::fabs(err2 / ref2);
	aErr += rErr2;
	if (rErr2 > mErr) mErr = rErr2;

	double err3 = double(ln.m128_f32[3]) - ref3;
	double rErr3 = ::fabs(err3 / ref3);
	aErr += rErr3;
	if (rErr3 > mErr) mErr = rErr3;
}

static void HK_CALL evalError(const __m128d& ln,
							  double ref0, double ref1,
							  double& aErr, double& mErr)
{
	double err0 = ln.m128d_f64[0] - ref0;
	double rErr0 = ::fabs(err0 / ref0);
	aErr += rErr0;
	if (rErr0 > mErr) mErr = rErr0;

	double err1 = ln.m128d_f64[1] - ref1;
	double rErr1 = ::fabs(err1 / ref1);
	aErr += rErr1;
	if (rErr1 > mErr) mErr = rErr1;
}


void atan2Accuracy()
{
	double aErrR = 0.0;
	double aErrQ = 0.0;
	double aErrQT = 0.0;
	double aErr2 = 0.0;
	double aErr2f = 0.0;
	double aErrV = 0.0;
	double mErrR = 0.0;
	double mErrQ = 0.0;
	double mErrQT = 0.0;
	double mErr2 = 0.0;
	double mErr2f = 0.0;
	double mErrV = 0.0;

	unsigned int count = 0;

	for (hkReal vy = -HK_REAL_PI; vy < HK_REAL_PI; vy = vy+0.01f)
	{
		for (hkReal vx = -HK_REAL_PI; vx < HK_REAL_PI; vx = vx+0.01f)
		{
			hkQuadReal y;
			y.m128_f32[0] = vy;
			y.m128_f32[1] = vy+0.001f;
			y.m128_f32[2] = vy+0.003f;
			y.m128_f32[3] = vy+0.007f;

			hkQuadReal rx;
			rx.m128_f32[0] = vx;
			rx.m128_f32[1] = vx+0.001f;
			rx.m128_f32[2] = vx+0.003f;
			rx.m128_f32[3] = vx+0.007f;

			double ref0 = ::atan2(double(y.m128_f32[0]),double(rx.m128_f32[0]));
			double ref1 = ::atan2(double(y.m128_f32[1]),double(rx.m128_f32[1]));
			double ref2 = ::atan2(double(y.m128_f32[2]),double(rx.m128_f32[2]));
			double ref3 = ::atan2(double(y.m128_f32[3]),double(rx.m128_f32[3]));

			{
				hkReal ln3_0 = hkMath::atan2fApproximation(y.m128_f32[0], rx.m128_f32[0]);
				hkReal ln3_1 = hkMath::atan2fApproximation(y.m128_f32[1], rx.m128_f32[1]);
				hkReal ln3_2 = hkMath::atan2fApproximation(y.m128_f32[2], rx.m128_f32[2]);
				hkReal ln3_3 = hkMath::atan2fApproximation(y.m128_f32[3], rx.m128_f32[3]);

				evalError(	double(ln3_0), double(ln3_1), double(ln3_2), double(ln3_3),
							ref0, ref1, ref2, ref3,
							aErr2f, mErr2f);
			}

			{
				hkReal ln4_0 = hkMath::atan2Approximation(y.m128_f32[0], rx.m128_f32[0]);
				hkReal ln4_1 = hkMath::atan2Approximation(y.m128_f32[1], rx.m128_f32[1]);
				hkReal ln4_2 = hkMath::atan2Approximation(y.m128_f32[2], rx.m128_f32[2]);
				hkReal ln4_3 = hkMath::atan2Approximation(y.m128_f32[3], rx.m128_f32[3]);

				evalError(	double(ln4_0), double(ln4_1), double(ln4_2), double(ln4_3),
							ref0, ref1, ref2, ref3,
							aErr2, mErr2);
			}

			{
				hkQuadReal aln2 = hkMath::quadAtan2(y,rx);

				evalError(	aln2,
							ref0, ref1, ref2, ref3,
							aErrQ, mErrQ);
			}

			{
				__m128d yt = _mm_cvtps_pd(y);
				__m128d rxt = _mm_cvtps_pd(rx);
				__m128d aln2 = hkMath::twoAtan2(yt,rxt);

				evalError(	aln2,
							ref0, ref1,
							aErrQT, mErrQT);
			}

			{
				hkVector4 vecX, vecY, vecR;
				vecX.m_quad = rx;
				vecY.m_quad = y;
				hkVector4Util::linearAtan2Approximation(vecY,vecX,vecR);

				evalError(	vecR.m_quad,
							ref0, ref1, ref2, ref3,
							aErrV, mErrV);
			}

			{
				hkVector4 vecX, vecY, vecR;
				vecX.m_quad = rx;
				vecY.m_quad = y;
				hkVector4Util::linearAtan2ApproximationRough(vecY,vecX,vecR);

				evalError(	vecR.m_quad,
							ref0, ref1, ref2, ref3,
							aErrR, mErrR);
			}

			count += 4;
		}
	}

	HK_REPORT("old atan2fA                           av="<<aErr2f/double(count)<<" max="<<mErr2f);
	HK_REPORT("new atan2A                            av="<<aErr2/double(count)<<" max="<<mErr2);
	HK_REPORT("hkVector4Util::linearAtan2Approx      av="<<aErrV/double(count)<<" max="<<mErrV);
	HK_REPORT("hkVector4Util::linearAtan2ApproxRough av="<<aErrR/double(count)<<" max="<<mErrR);
	HK_REPORT("hkMath::quadAtan2                     av="<<aErrQ/double(count)<<" max="<<mErrQ);
	HK_REPORT("hkMath::twoAtan2                      av="<<aErrQT/double(count/2)<<" max="<<mErrQT);
}

void atan2Perf()
{
	hkReal sum = 0.0f;

	hkArray<hkVector4> data;
	for (hkReal vx = -HK_REAL_PI; vx < (HK_REAL_PI-0.0008f); vx = vx+0.01f)
	{
		hkVector4 d; d.set(vx, vx+0.0001f, vx+0.0003f, vx+0.0007f);
		data.pushBack(d);
	}
	hkReal invNumData = hkReal(1)/hkReal(data.getSize());

	hkStopwatch sw;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkReal ln2_0 = hkMath::atan2fApproximation(rx.m128_f32[0], rx.m128_f32[0]);
		hkReal ln2_1 = hkMath::atan2fApproximation(rx.m128_f32[1], rx.m128_f32[1]);
		hkReal ln2_2 = hkMath::atan2fApproximation(rx.m128_f32[2], rx.m128_f32[2]);
		hkReal ln2_3 = hkMath::atan2fApproximation(rx.m128_f32[3], rx.m128_f32[3]);

		sum += ln2_0 + ln2_1 +ln2_2 + ln2_3;
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks old atan2fApp sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkReal ln2_0 = hkMath::atan2Approximation(rx.m128_f32[0], rx.m128_f32[0]);
		hkReal ln2_1 = hkMath::atan2Approximation(rx.m128_f32[1], rx.m128_f32[1]);
		hkReal ln2_2 = hkMath::atan2Approximation(rx.m128_f32[2], rx.m128_f32[2]);
		hkReal ln2_3 = hkMath::atan2Approximation(rx.m128_f32[3], rx.m128_f32[3]);

		sum += ln2_0 + ln2_1 +ln2_2 + ln2_3;
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks new atan2App sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		hkSimdReal vecX = data[i].getW(), vecY = data[i].getW();
		hkSimdReal vecR = hkVector4Util::linearAtan2Approximation(vecY,vecX);

		sum += vecR.getReal();
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks hkVector4Util::linearAtan2Approximation(SimdReal) sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkVector4 vecX, vecY, vecR;
		vecX.m_quad = rx;
		vecY.m_quad = rx;
		hkVector4Util::linearAtan2Approximation(vecY,vecX,vecR);

		sum += vecR(0);
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks hkVector4Util::linearAtan2Approximation sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkVector4 vecX, vecY, vecR;
		vecX.m_quad = rx;
		vecY.m_quad = rx;
		hkVector4Util::linearAtan2ApproximationRough(vecY,vecX,vecR);

		sum += vecR(0);
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks hkVector4Util::linearAtan2ApproximationRough sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadAtan2(rx,rx);

		sum += aln2.m128_f32[0];
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks hkMath::quadAtan2 sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		__m128d rx = _mm_cvtps_pd(data[i].m_quad);
		__m128d aln = hkMath::twoAtan2(rx,rx);
		__m128d rrx = _mm_cvtps_pd(_mm_shuffle_ps(data[i].m_quad,data[i].m_quad,_MM_SHUFFLE(0,1,2,3)));
		__m128d aln2 = hkMath::twoAtan2(rrx,rrx);

		sum += hkReal(aln.m128d_f64[0] + aln2.m128d_f64[0]);
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks hkMath::twoAtan2 sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkFloat32 ref0 = ::atan2f(rx.m128_f32[0],rx.m128_f32[0]);
		hkFloat32 ref1 = ::atan2f(rx.m128_f32[1],rx.m128_f32[1]);
		hkFloat32 ref2 = ::atan2f(rx.m128_f32[2],rx.m128_f32[2]);
		hkFloat32 ref3 = ::atan2f(rx.m128_f32[3],rx.m128_f32[3]);

		sum += hkReal(ref0 + ref1 + ref2 + ref3);
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::atan2f sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkDouble64 ref0 = ::atan2(hkDouble64(rx.m128_f32[0]),hkDouble64(rx.m128_f32[0]));
		hkDouble64 ref1 = ::atan2(hkDouble64(rx.m128_f32[1]),hkDouble64(rx.m128_f32[1]));
		hkDouble64 ref2 = ::atan2(hkDouble64(rx.m128_f32[2]),hkDouble64(rx.m128_f32[2]));
		hkDouble64 ref3 = ::atan2(hkDouble64(rx.m128_f32[3]),hkDouble64(rx.m128_f32[3]));

		sum += hkReal(ref0 + ref1 + ref2 + ref3);
	}
	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::atan2 sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));
}


void logAccuracy()
{
	double aErrQ = 0.0;
	double mErrQ = 0.0;

	unsigned int count = 0;
// 	hkOstream outQ("logQuad.csv");

	for (hkReal vx = HK_REAL_EPSILON; vx < 1e4f; vx = vx+0.01f)
	{
		//if ((vx > 0.9f) && (vx < 1.1f)) continue;

		hkQuadReal rx;
		rx.m128_f32[0] = vx;
		rx.m128_f32[1] = vx+0.001f;
		rx.m128_f32[2] = vx+0.003f;
		rx.m128_f32[3] = vx+0.007f;

		double ref0 = ::log(double(rx.m128_f32[0]));
		double ref1 = ::log(double(rx.m128_f32[1]));
		double ref2 = ::log(double(rx.m128_f32[2]));
		double ref3 = ::log(double(rx.m128_f32[3]));

		hkQuadReal aln2 = hkMath::quadLog(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = hkMath::fabs(err0 / ref0);
// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrQ += rErr0;
			if (rErr0 > mErrQ) mErrQ = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = hkMath::fabs(err1 / ref1);
// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrQ += rErr1;
			if (rErr1 > mErrQ) mErrQ = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = hkMath::fabs(err2 / ref2);
// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrQ += rErr2;
			if (rErr2 > mErrQ) mErrQ = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = hkMath::fabs(err3 / ref3);
// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrQ += rErr3;
			if (rErr3 > mErrQ) mErrQ = rErr3;
		}

		count += 4;
	}

	HK_REPORT("quadLog   av="<<aErrQ/double(count)<<" max="<<mErrQ);
}

void logPerf()
{
	hkReal sum = 0.0f;

	hkStopwatch sw;
	sw.reset();
	sw.start();

	for (hkReal vx = HK_REAL_EPSILON; vx < 1e4f; vx = vx+0.01f)
	{
		hkQuadReal rx;
		rx.m128_f32[0] = vx;
		rx.m128_f32[1] = vx+0.0001f;
		rx.m128_f32[2] = vx+0.0003f;
		rx.m128_f32[3] = vx+0.0007f;

		hkReal ln2_0 = hkMath::log(rx.m128_f32[0]);
		hkReal ln2_1 = hkMath::log(rx.m128_f32[1]);
		hkReal ln2_2 = hkMath::log(rx.m128_f32[2]);
		hkReal ln2_3 = hkMath::log(rx.m128_f32[3]);

		sum += ln2_0 + ln2_1 +ln2_2 + ln2_3;
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks hkMath::log sum="<<sum);

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (hkReal vx = HK_REAL_EPSILON; vx < 1e4f; vx = vx+0.01f)
	{
		hkQuadReal rx;
		rx.m128_f32[0] = vx;
		rx.m128_f32[1] = vx+0.0001f;
		rx.m128_f32[2] = vx+0.0003f;
		rx.m128_f32[3] = vx+0.0007f;

		hkQuadReal aln2 = hkMath::quadLog(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadLog sum="<<sum);
}

void asinAcosAccuracy()
{
	double aErrS = 0.0;
	double mErrS = 0.0;
	double aErrSD = 0.0;
	double mErrSD = 0.0;
	double aErrC = 0.0;
	double mErrC = 0.0;
	double aErrCD = 0.0;
	double mErrCD = 0.0;

	unsigned int count = 0;

	for (hkReal vx = -1.0f; vx < (1.0f-0.008f); vx = vx+0.01f)
	{
		hkQuadReal rx;
		rx.m128_f32[0] = vx;
		rx.m128_f32[1] = vx+0.001f;
		rx.m128_f32[2] = vx+0.003f;
		rx.m128_f32[3] = vx+0.007f;

		double ref0 = HK_STD_NAMESPACE::asin(double(rx.m128_f32[0]));
		double ref1 = HK_STD_NAMESPACE::asin(double(rx.m128_f32[1]));
		double ref2 = HK_STD_NAMESPACE::asin(double(rx.m128_f32[2]));
		double ref3 = HK_STD_NAMESPACE::asin(double(rx.m128_f32[3]));

		hkQuadReal aln2 = hkMath::quadAsin(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrS += rErr0;
			if (rErr0 > mErrS) mErrS = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrS += rErr1;
			if (rErr1 > mErrS) mErrS = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = (ref2 == 0.0) ? hkMath::fabs(err2) : hkMath::fabs(err2 / ref2);
			// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrS += rErr2;
			if (rErr2 > mErrS) mErrS = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = (ref3 == 0.0) ? hkMath::fabs(err3) : hkMath::fabs(err3 / ref3);
			// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrS += rErr3;
			if (rErr3 > mErrS) mErrS = rErr3;
		}

		__m128d twoRX = _mm_cvtps_pd(rx);
		__m128d twoS  = hkMath::twoAsin(twoRX);

		{
			double err0 = double(twoS.m128d_f64[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrSD += rErr0;
			if (rErr0 > mErrSD) mErrSD = rErr0;

			double err1 = double(twoS.m128d_f64[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrSD += rErr1;
			if (rErr1 > mErrSD) mErrSD = rErr1;
		}

		ref0 = HK_STD_NAMESPACE::acos(double(rx.m128_f32[0]));
		ref1 = HK_STD_NAMESPACE::acos(double(rx.m128_f32[1]));
		ref2 = HK_STD_NAMESPACE::acos(double(rx.m128_f32[2]));
		ref3 = HK_STD_NAMESPACE::acos(double(rx.m128_f32[3]));

		aln2 = hkMath::quadAcos(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrC += rErr0;
			if (rErr0 > mErrC) mErrC = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrC += rErr1;
			if (rErr1 > mErrC) mErrC = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = (ref2 == 0.0) ? hkMath::fabs(err2) : hkMath::fabs(err2 / ref2);
			// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrC += rErr2;
			if (rErr2 > mErrC) mErrC = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = (ref3 == 0.0) ? hkMath::fabs(err3) : hkMath::fabs(err3 / ref3);
			// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrC += rErr3;
			if (rErr3 > mErrC) mErrC = rErr3;
		}

		__m128d twoC  = hkMath::twoAcos(twoRX);

		{
			double err0 = double(twoC.m128d_f64[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrCD += rErr0;
			if (rErr0 > mErrCD) mErrCD = rErr0;

			double err1 = double(twoC.m128d_f64[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrCD += rErr1;
			if (rErr1 > mErrCD) mErrCD = rErr1;
		}

		count += 4;
	}

	HK_REPORT("quadAsin   av="<<aErrS/double(count)<<" max="<<mErrS);
	HK_REPORT("twoAsin    av="<<aErrSD/double(count/2)<<" max="<<mErrSD);
	HK_REPORT("quadAcos   av="<<aErrC/double(count)<<" max="<<mErrC);
	HK_REPORT("twoAcos    av="<<aErrCD/double(count/2)<<" max="<<mErrCD);
}

void asinAcosPerf()
{
	hkReal sum = 0.0f;

	hkArray<hkVector4> data;
	for (hkReal vx = -1.0f; vx < (1.0f-0.008f); vx = vx+0.01f)
	{
		hkVector4 d; d.set(vx, vx+0.0001f, vx+0.0003f, vx+0.0007f);
		data.pushBack(d);
	}
	hkReal invNumData = hkReal(1)/hkReal(data.getSize());

	hkStopwatch sw;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkFloat32 ln2_0 = HK_STD_NAMESPACE::asinf(rx.m128_f32[0]);
		hkFloat32 ln2_1 = HK_STD_NAMESPACE::asinf(rx.m128_f32[1]);
		hkFloat32 ln2_2 = HK_STD_NAMESPACE::asinf(rx.m128_f32[2]);
		hkFloat32 ln2_3 = HK_STD_NAMESPACE::asinf(rx.m128_f32[3]);

		sum += hkReal(ln2_0 + ln2_1 +ln2_2 + ln2_3);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::asinf sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadAsin(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadAsin sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkDouble64 ln2_0 = HK_STD_NAMESPACE::asin(hkDouble64(rx.m128_f32[0]));
		hkDouble64 ln2_1 = HK_STD_NAMESPACE::asin(hkDouble64(rx.m128_f32[1]));
		hkDouble64 ln2_2 = HK_STD_NAMESPACE::asin(hkDouble64(rx.m128_f32[2]));
		hkDouble64 ln2_3 = HK_STD_NAMESPACE::asin(hkDouble64(rx.m128_f32[3]));

		sum += hkReal(ln2_0 + ln2_1 +ln2_2 + ln2_3);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::asin sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		__m128d rrx0 = _mm_cvtps_pd(rx);
		__m128d sin0 = hkMath::twoAsin(rrx0);
		__m128d rrx1 = _mm_cvtps_pd(_mm_shuffle_ps(rx,rx,_MM_SHUFFLE(0,1,2,3)));
		__m128d sin1 = hkMath::twoAsin(rrx1);

		sum += hkReal(sin0.m128d_f64[0] + sin1.m128d_f64[0]);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks twoAsin sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkFloat32 ln2_0 = HK_STD_NAMESPACE::acosf(rx.m128_f32[0]);
		hkFloat32 ln2_1 = HK_STD_NAMESPACE::acosf(rx.m128_f32[1]);
		hkFloat32 ln2_2 = HK_STD_NAMESPACE::acosf(rx.m128_f32[2]);
		hkFloat32 ln2_3 = HK_STD_NAMESPACE::acosf(rx.m128_f32[3]);

		sum += hkReal(ln2_0 + ln2_1 +ln2_2 + ln2_3);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::acosf sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadAcos(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadAcos sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkDouble64 ln2_0 = HK_STD_NAMESPACE::acos(hkDouble64(rx.m128_f32[0]));
		hkDouble64 ln2_1 = HK_STD_NAMESPACE::acos(hkDouble64(rx.m128_f32[1]));
		hkDouble64 ln2_2 = HK_STD_NAMESPACE::acos(hkDouble64(rx.m128_f32[2]));
		hkDouble64 ln2_3 = HK_STD_NAMESPACE::acos(hkDouble64(rx.m128_f32[3]));

		sum += hkReal(ln2_0 + ln2_1 +ln2_2 + ln2_3);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::acos sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		__m128d rrx0 = _mm_cvtps_pd(rx);
		__m128d sin0 = hkMath::twoAcos(rrx0);
		__m128d rrx1 = _mm_cvtps_pd(_mm_shuffle_ps(rx,rx,_MM_SHUFFLE(0,1,2,3)));
		__m128d sin1 = hkMath::twoAcos(rrx1);

		sum += hkReal(sin0.m128d_f64[0] + sin1.m128d_f64[0]);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks twoAcos sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));
}

void sinCosAccuracy()
{
	double aErrQSD = 0.0;
	double mErrQSD = 0.0;
	double aErrQS = 0.0;
	double mErrQS = 0.0;
	double aErrQC = 0.0;
	double mErrQC = 0.0;
	double aErrQSF = 0.0;
	double mErrQSF = 0.0;
	double aErrQCF = 0.0;
	double mErrQCF = 0.0;

	unsigned int count = 0;
	// 	hkOstream outQ("logQuad.csv");

	for (hkReal vx = 0.0f; vx < 2.0f*HK_REAL_PI; vx = vx+0.01f)
	{
		//if ((vx > 0.9f) && (vx < 1.1f)) continue;

		hkQuadReal rx;
		rx.m128_f32[0] = vx;
		rx.m128_f32[1] = vx+0.001f;
		rx.m128_f32[2] = vx+0.003f;
		rx.m128_f32[3] = vx+0.007f;

		double ref0 = HK_STD_NAMESPACE::cos(double(rx.m128_f32[0]));
		double ref1 = HK_STD_NAMESPACE::cos(double(rx.m128_f32[1]));
		double ref2 = HK_STD_NAMESPACE::cos(double(rx.m128_f32[2]));
		double ref3 = HK_STD_NAMESPACE::cos(double(rx.m128_f32[3]));

		hkQuadReal aln2 = hkMath::quadCos(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrQC += rErr0;
			if (rErr0 > mErrQC) mErrQC = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrQC += rErr1;
			if (rErr1 > mErrQC) mErrQC = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = (ref2 == 0.0) ? hkMath::fabs(err2) : hkMath::fabs(err2 / ref2);
			// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrQC += rErr2;
			if (rErr2 > mErrQC) mErrQC = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = (ref3 == 0.0) ? hkMath::fabs(err3) : hkMath::fabs(err3 / ref3);
			// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrQC += rErr3;
			if (rErr3 > mErrQC) mErrQC = rErr3;
		}

		aln2 = hkMath::quadCosApproximation(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrQCF += rErr0;
			if (rErr0 > mErrQCF) mErrQCF = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrQCF += rErr1;
			if (rErr1 > mErrQCF) mErrQCF = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = (ref2 == 0.0) ? hkMath::fabs(err2) : hkMath::fabs(err2 / ref2);
			// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrQCF += rErr2;
			if (rErr2 > mErrQCF) mErrQCF = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = (ref3 == 0.0) ? hkMath::fabs(err3) : hkMath::fabs(err3 / ref3);
			// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrQCF += rErr3;
			if (rErr3 > mErrQCF) mErrQCF = rErr3;
		}

		ref0 = HK_STD_NAMESPACE::sin(double(rx.m128_f32[0]));
		ref1 = HK_STD_NAMESPACE::sin(double(rx.m128_f32[1]));
		ref2 = HK_STD_NAMESPACE::sin(double(rx.m128_f32[2]));
		ref3 = HK_STD_NAMESPACE::sin(double(rx.m128_f32[3]));

		__m128d twoRX = _mm_cvtps_pd(rx);
		__m128d twoS  = hkMath::twoSin(twoRX);

		{
			double err0 = double(twoS.m128d_f64[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrQSD += rErr0;
			if (rErr0 > mErrQSD) mErrQSD = rErr0;

			double err1 = double(twoS.m128d_f64[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrQSD += rErr1;
			if (rErr1 > mErrQSD) mErrQSD = rErr1;
		}

		aln2 = hkMath::quadSin(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrQS += rErr0;
			if (rErr0 > mErrQS) mErrQS = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrQS += rErr1;
			if (rErr1 > mErrQS) mErrQS = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = (ref2 == 0.0) ? hkMath::fabs(err2) : hkMath::fabs(err2 / ref2);
			// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrQS += rErr2;
			if (rErr2 > mErrQS) mErrQS = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = (ref3 == 0.0) ? hkMath::fabs(err3) : hkMath::fabs(err3 / ref3);
			// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrQS += rErr3;
			if (rErr3 > mErrQS) mErrQS = rErr3;
		}

		aln2 = hkMath::quadSinApproximation(rx);

		{
			double err0 = double(aln2.m128_f32[0]) - ref0;
			double rErr0 = (ref0 == 0.0) ? hkMath::fabs(err0) : hkMath::fabs(err0 / ref0);
			// 			outQ << rx.m128_f32[0] << " , " << rErr0 << "\n";
			aErrQSF += rErr0;
			if (rErr0 > mErrQSF) mErrQSF = rErr0;

			double err1 = double(aln2.m128_f32[1]) - ref1;
			double rErr1 = (ref1 == 0.0) ? hkMath::fabs(err1) : hkMath::fabs(err1 / ref1);
			// 			outQ << rx.m128_f32[1] << " , " << rErr1 << "\n";
			aErrQSF += rErr1;
			if (rErr1 > mErrQSF) mErrQSF = rErr1;

			double err2 = double(aln2.m128_f32[2]) - ref2;
			double rErr2 = (ref2 == 0.0) ? hkMath::fabs(err2) : hkMath::fabs(err2 / ref2);
			// 			outQ << rx.m128_f32[2] << " , " << rErr2 << "\n";
			aErrQSF += rErr2;
			if (rErr2 > mErrQSF) mErrQSF = rErr2;

			double err3 = double(aln2.m128_f32[3]) - ref3;
			double rErr3 = (ref3 == 0.0) ? hkMath::fabs(err3) : hkMath::fabs(err3 / ref3);
			// 			outQ << rx.m128_f32[3] << " , " << rErr3 << "\n";
			aErrQSF += rErr3;
			if (rErr3 > mErrQSF) mErrQSF = rErr3;
		}

		count += 4;
	}

	HK_REPORT("twoSin   av="<<aErrQSD/double(count/2)<<" max="<<mErrQSD);

	HK_REPORT("quadCos   av="<<aErrQC/double(count)<<" max="<<mErrQC);
	HK_REPORT("quadSin   av="<<aErrQS/double(count)<<" max="<<mErrQS);

	HK_REPORT("quadCosApproximation    av="<<aErrQCF/double(count)<<" max="<<mErrQCF);
	HK_REPORT("quadSinApproximation    av="<<aErrQSF/double(count)<<" max="<<mErrQSF);
}

void sinCosPerf()
{
	hkReal sum = 0.0f;

	hkArray<hkVector4> data;
	for (hkReal vx = 0.0f; vx < 2.0f*HK_REAL_PI; vx = vx+0.01f)
	{
		hkVector4 d; d.set(vx, vx+0.0001f, vx+0.0003f, vx+0.0007f);
		data.pushBack(d);
	}
	hkReal invNumData = hkReal(1)/hkReal(data.getSize());

	hkStopwatch sw;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkDouble64 ln2_0 = HK_STD_NAMESPACE::sin(hkDouble64(rx.m128_f32[0]));
		hkDouble64 ln2_1 = HK_STD_NAMESPACE::sin(hkDouble64(rx.m128_f32[1]));
		hkDouble64 ln2_2 = HK_STD_NAMESPACE::sin(hkDouble64(rx.m128_f32[2]));
		hkDouble64 ln2_3 = HK_STD_NAMESPACE::sin(hkDouble64(rx.m128_f32[3]));

		sum += hkReal(ln2_0 + ln2_1 +ln2_2 + ln2_3);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::sin sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkFloat32 ln2_0 = HK_STD_NAMESPACE::sinf(hkFloat32(rx.m128_f32[0]));
		hkFloat32 ln2_1 = HK_STD_NAMESPACE::sinf(hkFloat32(rx.m128_f32[1]));
		hkFloat32 ln2_2 = HK_STD_NAMESPACE::sinf(hkFloat32(rx.m128_f32[2]));
		hkFloat32 ln2_3 = HK_STD_NAMESPACE::sinf(hkFloat32(rx.m128_f32[3]));

		sum += hkReal(ln2_0 + ln2_1 +ln2_2 + ln2_3);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::sinf sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		__m128d rrx0 = _mm_cvtps_pd(rx);
		__m128d sin0 = hkMath::twoSin(rrx0);
		__m128d rrx1 = _mm_cvtps_pd(_mm_shuffle_ps(rx,rx,_MM_SHUFFLE(0,1,2,3)));
		__m128d sin1 = hkMath::twoSin(rrx1);

		sum += hkReal(sin0.m128d_f64[0] + sin1.m128d_f64[0]);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks twoSin sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadSin(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadSin sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		__m128d rrx0 = _mm_cvtps_pd(rx);
		__m128d sin0 = hkMath::twoSinApproximation(rrx0);
		__m128d rrx1 = _mm_cvtps_pd(_mm_shuffle_ps(rx,rx,_MM_SHUFFLE(0,1,2,3)));
		__m128d sin1 = hkMath::twoSinApproximation(rrx1);

		sum += hkReal(sin0.m128d_f64[0] + sin1.m128d_f64[0]);
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks twoSinApproximation sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadSinApproximation(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadSinApproximation sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

#if defined(HK_REAL_IS_DOUBLE)
		hkReal ln2_0 = HK_STD_NAMESPACE::cos(rx.m128_f32[0]);
		hkReal ln2_1 = HK_STD_NAMESPACE::cos(rx.m128_f32[1]);
		hkReal ln2_2 = HK_STD_NAMESPACE::cos(rx.m128_f32[2]);
		hkReal ln2_3 = HK_STD_NAMESPACE::cos(rx.m128_f32[3]);
#else
		hkReal ln2_0 = HK_STD_NAMESPACE::cosf(rx.m128_f32[0]);
		hkReal ln2_1 = HK_STD_NAMESPACE::cosf(rx.m128_f32[1]);
		hkReal ln2_2 = HK_STD_NAMESPACE::cosf(rx.m128_f32[2]);
		hkReal ln2_3 = HK_STD_NAMESPACE::cosf(rx.m128_f32[3]);
#endif

		sum += ln2_0 + ln2_1 +ln2_2 + ln2_3;
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks ::cos sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadCos(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadCos sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));

	sum = 0.0f;
	sw.reset();
	sw.start();

	for (int i=0; i<data.getSize(); ++i)
	{
		const hkQuadReal& rx = data[i].m_quad;

		hkQuadReal aln2 = hkMath::quadCosApproximation(rx);

		sum += aln2.m128_f32[0];
	}

	sw.stop();
	HK_REPORT(sw.getElapsedTicks()<<" ticks quadCosApproximation sum="<<sum<<" ticks per call: "<<int(sw.getElapsedTicks()*invNumData));
}
#endif

int mathfunc_main()
{
//  	logAccuracy();
//  	logPerf();
//  	atan2Accuracy();
//  	atan2Perf();
//  	sinCosAccuracy();
//  	sinCosPerf();
// 		asinAcosAccuracy();
// 		asinAcosPerf();
	mathfunc_test();
	types_test();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(mathfunc_main, "Slow", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/mathfunc.cpp"     );

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
