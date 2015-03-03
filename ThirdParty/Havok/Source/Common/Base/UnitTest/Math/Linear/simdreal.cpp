/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/UnitTest/Math/Linear/mathtestutils.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

#define checkEqual23Bit(x,y) HK_TEST( hkIsEqual23BitAccurate(x.getReal(), y.getReal()))
#define checkEqual12Bit(x,y) HK_TEST( hkIsEqual12BitAccurate(x.getReal(), y.getReal()))

static void simdreal_test()
{
		{ // construct from float
			const float refValue = 62.0f;
			hkSimdReal value; value.setFromFloat(refValue);

#	if defined(HK_ARCH_IA32) || defined(HK_ARCH_X64)
#		if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
#			if defined(HK_REAL_IS_DOUBLE)
			{
				double* cast = (double*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
				HK_TEST(hkMath::equal(cast[1], refValue));
			}
#			else
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
				HK_TEST(hkMath::equal(cast[1], refValue));
				HK_TEST(hkMath::equal(cast[2], refValue));
				HK_TEST(hkMath::equal(cast[3], refValue));
			}
#			endif
#		else
#			if defined(HK_REAL_IS_DOUBLE)
			{
				double* cast = (double*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
			}
#			else
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
			}
#			endif
#		endif

			// PSP(R) (PlayStation(R)Portable)
#	elif defined(HK_PLATFORM_PSP)
			{
				union SimdUnion
				{
					hkSimdReal value;
					float cast;
				};

				HK_ALIGN16(SimdUnion u);
				HK_TEST(hkMath::equal(u.cast, refValue));
			}
			// PlayStation(R)3
#	elif defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_PS3_SPU)
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
				HK_TEST(hkMath::equal(cast[1], refValue));
				HK_TEST(hkMath::equal(cast[2], refValue));
				HK_TEST(hkMath::equal(cast[3], refValue));
			}
			// Xbox 360
#	elif defined(HK_PLATFORM_XBOX360)
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
				HK_TEST(hkMath::equal(cast[1], refValue));
				HK_TEST(hkMath::equal(cast[2], refValue));
				HK_TEST(hkMath::equal(cast[3], refValue));
			}
			// Gamecube
#	elif defined(HK_PLATFORM_GC)
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
			}
#	elif defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_PSVITA) 
#		if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
				HK_TEST(hkMath::equal(cast[1], refValue));
			}
#		else 
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
			}
#		endif
#	elif defined(HK_PLATFORM_WIIU)
#		if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
			{
				union SimdUnion
				{
					hkSimdReal value;
					float cast[4];
				};

				HK_ALIGN16(SimdUnion u);
				u.value.setFromFloat(refValue);
				HK_TEST(hkMath::equal(u.cast[0], refValue));
			}
#		else 
			{
				float* cast = (float*)&value;
				HK_TEST(hkMath::equal(cast[0], refValue));
			}
#		endif
#	else
			{
				HK_WARN_ALWAYS(0x99000ff, "SimdReal platform-specific data storage verification test not implemented for this platform");
			}
#	endif
		}

		{
			hkReal a = 0.0f;
			hkReal b = 2.0f;
			hkSimdReal sa; sa.setFromFloat(a);
			hkSimdReal sb; sb.setFromFloat(b);
			HK_TEST( (0!=(	sa < sb )) == (a<b));
			HK_TEST( (0!=(  sa > sb )) == (a>b));
			HK_TEST( (0!=(	sa == sb )) == (a==b));
		}
		{
			hkReal a = 2.0f;
			hkReal b = -2.0f;
			hkSimdReal sa; sa.setFromFloat(a);
			hkSimdReal sb; sb.setFromFloat(b);
			HK_TEST( (0!=(	sa < sb )) == (a<b));
			HK_TEST( (0!=(  sa > sb )) == (a>b));
			HK_TEST( (0!=(	sa == sb )) == (a==b));
		}
		{
			hkReal a = 2.0f;
			hkReal b = 2.0f;
			hkSimdReal sa; sa.setFromFloat(a);
			hkSimdReal sb; sb.setFromFloat(b);
			HK_TEST( (0!=(	sa < sb )) == (a<b));
			HK_TEST( (0!=(  sa > sb )) == (a>b));
			HK_TEST( (0!=(	sa == sb )) == (a==b));
		}


#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED
		//
		//	setSelect
		//
		{
			hkSimdReal one = hkSimdReal::getConstant<HK_QUADREAL_1>();
			hkSimdReal two = hkSimdReal::getConstant<HK_QUADREAL_2>();
			hkVector4Comparison oneGreaterThanTwo = one.greater(two); // false
			hkVector4Comparison oneLessThanTwo = one.less(two); // true
			hkSimdReal res;  res.setSelect ( oneGreaterThanTwo, one, two );
			hkSimdReal res2; res2.setSelect( oneLessThanTwo,    one, two );
			HK_TEST( res == two );
			HK_TEST( res2 == one );
		}
#endif
}

static void simdreal_setget_basic()
{
	// from real
	{
		hkSimdReal a;
		hkReal f(2.0f);
		a.setFromFloat(f);
		HK_TEST( a.getReal() == f );
	}

	// half
	{
		hkSimdReal a;
		hkHalf f; f.setReal<true>(hkReal(2));
		a.setFromHalf(f);
		HK_TEST( a.getReal() == f );
		hkHalf r[2]; r[1].setZero();
		a.store<1>(&r[0]);
		HK_TEST( r[0] == f );
		HK_TEST( r[1] == 0.0f );
	}

	// uint16
	{
		hkSimdReal a;
		hkUint16 f(31);
		a.setFromUint16(f);
		hkUint16 result; a.storeSaturateUint16(&result);
		HK_TEST( result == f );
		HK_TEST( a.getReal() == hkReal(f) );
	}

	// uint8
	{
		hkSimdReal a;
		hkUint8 f(31);
		a.setFromUint8(f);
		hkUint16 result; a.storeSaturateUint16(&result);
		HK_TEST( hkUint8(result) == f );
		HK_TEST( a.getReal() == hkReal(f) );
	}

	// uint32
	{
		hkSimdReal a;
		hkInt32 f(31);
		a.setFromInt32(f);
		hkInt32 result; a.storeSaturateInt32(&result);
		HK_TEST( result == f );
		HK_TEST( a.getReal() == hkReal(f) );
	}
	{
		hkSimdReal a;
		hkInt32 f(-47);
		a.setFromInt32(f);
		hkInt32 result; a.storeSaturateInt32(&result);
		HK_TEST( result == f );
		HK_TEST( a.getReal() == hkReal(f) );
	}

	// setZero
	{
		hkSimdReal a; a.setZero();
		HK_TEST( a.getReal() == hkReal(0.0f) );
		hkUint16 result16; a.storeSaturateUint16(&result16);
		hkInt32 result32; a.storeSaturateInt32(&result32);
		HK_TEST( result16 == 0 );
		HK_TEST( result32 == 0 );
	}


	// Constructors
	{
		hkReal a(6.0f);
		hkSimdReal b; b.setFromFloat(a);
		HK_TEST( b.getReal() == a);
#ifndef HK_DISABLE_IMPLICIT_SIMDREAL_FLOAT_CONVERSION
		HK_TEST(b() == a);
#endif
	}

	// constants
	{
		hkSimdReal a = hkSimdReal::getConstant(HK_QUADREAL_MINUS1);
		HK_TEST(a.getReal() == -1.0f);
		HK_TEST(a == hkSimdReal::getConstant(HK_QUADREAL_MINUS1));

		a = hkSimdReal::getConstant(HK_QUADREAL_5);
		HK_TEST(a.getReal() == 5.0f);
		HK_TEST(a == hkSimdReal::getConstant(HK_QUADREAL_5));

		a = hkSimdReal::getConstant(HK_QUADREAL_INV_2);
		HK_TEST(a.getReal() == 0.5f);
		HK_TEST(a == hkSimdReal::getConstant(HK_QUADREAL_INV_2));
	}

}

static void simdreal_assign_special()
{
	// min max
	{
		hkSimdReal r1; r1.setFromFloat( 1.0f );
		hkSimdReal r2; r2.setFromFloat( 2.0f );
		hkSimdReal m1; m1.setMin( r1, r2 );
		hkSimdReal m2; m2.setMax( r1, r2 );
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_1));
		HK_TEST( m2 == hkSimdReal::getConstant(HK_QUADREAL_2));
	}
	{
		hkSimdReal r1; r1.setFromFloat( -0.0f );
		hkSimdReal r2; r2.setFromFloat( 0.0f );
		hkSimdReal m1; m1.setMin( r1, r2 );
		hkSimdReal m2; m2.setMax( r1, r2 );
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_0));
		HK_TEST( m2 == hkSimdReal::getConstant(HK_QUADREAL_0));
	}

	// clamp
	{
		hkSimdReal r1; r1.setClamped( hkSimdReal_3, hkSimdReal_1, hkSimdReal_2 ); // == 2
		hkSimdReal r2; r2.setClamped( hkSimdReal_0, hkSimdReal_1, hkSimdReal_2 ); // == 1
		hkSimdReal r3; r3.setClamped( hkSimdReal_2, hkSimdReal_0, hkSimdReal_4 ); // == 2 (no change)

		HK_TEST( r1 == hkSimdReal_2);
		HK_TEST( r2 == hkSimdReal_1);
		HK_TEST( r3 == hkSimdReal_2);

		hkSimdReal r4; r4.setClampedZeroOne( hkSimdReal_Half );
		hkSimdReal r5; r5.setClampedZeroOne( hkSimdReal_2 );
		hkSimdReal r6; r6.setClampedZeroOne( hkSimdReal_Minus1 );

		HK_TEST( r4 == hkSimdReal_Half);
		HK_TEST( r5 == hkSimdReal_1);
		HK_TEST( r6 == hkSimdReal_0);

		// Test NAN
		hkSimdReal zero = hkSimdReal_0;
		hkSimdReal simdNAN; simdNAN.setDiv<HK_ACC_FULL, HK_DIV_IGNORE>(zero,zero);

		HK_ON_CPU( HK_TEST( !simdNAN.isOk() ) );
		hkSimdReal r7; r7.setClamped( simdNAN, hkSimdReal_1, hkSimdReal_2 );
		HK_TEST( (r7 >= hkSimdReal_1) && (r7 <= hkSimdReal_2) );

		hkSimdReal r8; r8.setClampedZeroOne( simdNAN );
		HK_TEST( (r8 >= hkSimdReal_0) && (r8 <= hkSimdReal_1) );
	}

	// abs
	{
		hkSimdReal r1; r1.setFromFloat( -1.0f );
		hkSimdReal r2; r2.setFromFloat( 2.0f );
		hkSimdReal m1; m1.setAbs( r1 );
		hkSimdReal m2; m2.setAbs( r2 );
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_1));
		HK_TEST( m2 == hkSimdReal::getConstant(HK_QUADREAL_2));
	}
	{
		hkSimdReal r1; r1.setFromFloat( -0.0f );
		hkSimdReal r2; r2.setFromFloat( 0.0f );
		hkSimdReal m1; m1.setAbs( r1 );
		hkSimdReal m2; m2.setAbs( r1 );
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_0));
		HK_TEST( m2 == hkSimdReal::getConstant(HK_QUADREAL_0));
	}

	// flip sign
	{
		hkSimdReal r1; r1.setFromFloat( -1.0f );
		hkSimdReal r2; r2.setFromFloat( -2.0f );
		hkSimdReal r3; r3.setFromFloat( 3.0f );
		hkSimdReal m1;
		m1.setFlipSign( r1, r2);
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_1));
		m1.setFlipSign( r1, r3);
		HK_TEST( m1.getReal() == -1.0f);
		m1.setFlipSign( r3, r3);
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_3));
		m1.setFlipSign( r3, r1);
		HK_TEST( m1.getReal() == -3.0f);
	}
	{
		hkSimdReal r1; r1.setFromFloat( -1.0f );
		hkSimdReal r2; r2.setFromFloat( 3.0f );
		hkSimdReal m1;
		hkVector4Comparison allMasks; allMasks.set(hkVector4ComparisonMask::MASK_XYZW);
		hkVector4Comparison noMask; noMask.set(hkVector4ComparisonMask::MASK_NONE);
		m1.setFlipSign( r1, allMasks);
		HK_TEST( m1 == hkSimdReal::getConstant(HK_QUADREAL_1));

		m1.setFlipSign( r1, noMask);
		HK_TEST( m1.getReal() == -1.0f);

		m1.setFlipSign( r2, allMasks);
		HK_TEST( m1.getReal() == -3.0f);
	}

	// select
	{
		hkSimdReal r1; r1.setFromFloat( -1.0f );
		hkSimdReal r2; r2.setFromFloat( 3.0f );
		hkSimdReal m1;
		hkSimdReal m2;
		hkVector4Comparison allMasks; allMasks.set(hkVector4ComparisonMask::MASK_XYZW);
		hkVector4Comparison noMask; noMask.set(hkVector4ComparisonMask::MASK_NONE);

		m1.setSelect( allMasks, r1, r2);
		HK_TEST( m1 == r1);

		m2.setSelect( noMask, r1, r2);
		HK_TEST( m2 == r2);
	}

	// zeroIfTrue and zeroIfFalse
	{
		hkVector4Comparison trueMask = hkSimdReal_1.greater(hkSimdReal_0);
		hkVector4Comparison falseMask = hkSimdReal_0.greater(hkSimdReal_1);
		
		{
			hkSimdReal r1 = hkSimdReal_1;
			r1.zeroIfTrue(trueMask);
			HK_TEST( r1.isEqualZero() );
		}

		{
			hkSimdReal r2 = hkSimdReal_2;
			r2.zeroIfTrue(falseMask);
			HK_TEST( r2.isNotEqualZero() );
		}

		{
			hkSimdReal r3 = hkSimdReal_3;
			r3.zeroIfFalse(trueMask);
			HK_TEST( r3.isNotEqualZero() );
		}

		{
			hkSimdReal r4 = hkSimdReal_4;
			r4.zeroIfFalse(falseMask);
			HK_TEST( r4.isEqualZero() );
		}
	}
}

static void simdreal_load_store()
{
	//	store
	{
		hkSimdReal one; one.setFromFloat(1.0f);
		HK_ALIGN16( hkReal data[4] );
		data[0] = data[1] = data[2] = data[3] = 0.0f;
		one.store<1>( &data[2] );
		HK_TEST( data[0] == 0.0f );
		HK_TEST( data[1] == 0.0f );
		HK_TEST( data[2] == 1.0f );
		HK_TEST( data[3] == 0.0f );
	}
	{
		hkSimdReal one; one.setFromFloat(1.0f);
		HK_ALIGN16( hkReal data[4] );
		data[0] = data[1] = data[2] = data[3] = 0.0f;
		one.store<1>( &data[0] );
		HK_TEST( data[0] == 1.0f );
		HK_TEST( data[1] == 0.0f );
		HK_TEST( data[2] == 0.0f );
		HK_TEST( data[3] == 0.0f );

// 		HK_TEST_ASSERT(0x64211c2f, one.store<1>(&data[1]));
// 		HK_TEST_ASSERT(0x64211c2f, one.store<1>(&data[2]));
// 		HK_TEST_ASSERT(0x64211c2f, one.store<1>(&data[3]));
	}
	{
		hkSimdReal one; one.setFromFloat(1.0f);
		HK_ALIGN16(hkReal data[4]);
		data[0] = data[1] = data[2] = data[3] = 0.0f;
		one.store<1,HK_IO_NOT_CACHED>( &data[0] );
		HK_TEST( data[0] == 1.0f );
		HK_TEST( data[1] == 0.0f );
		HK_TEST( data[2] == 0.0f );
		HK_TEST( data[3] == 0.0f );

// 		HK_TEST_ASSERT(0x64211c2f, one.storeNotCached<1>(&data[1]));
// 		HK_TEST_ASSERT(0x64211c2f, one.storeNotCached<1>(&data[2]));
// 		HK_TEST_ASSERT(0x64211c2f, one.storeNotCached<1>(&data[3]));
	}

	// load
	{
		hkSimdReal r;
		hkReal data[3] = {0.0f, 1.0f, -2.0f};
		for (int i = 0; i < 3; i++)
		{
			r.load<1>( &data[i] );
			HK_TEST( r.getReal() == data[i] );
		}
	}
	{
		hkSimdReal r;
		HK_ALIGN16( hkReal data[4]) = {0.0f, 1.0f, -2.0f, 0.0f};	// these must be 4 values for XBOX
		{
			r.load<1>( &data[0] );
			HK_TEST( r.getReal() == data[0] );
// 			HK_TEST_ASSERT(0x64211c2f, r.load<1>(&data[1]));
// 			HK_TEST_ASSERT(0x64211c2f, r.load<1>(&data[2]));
// 			HK_TEST_ASSERT(0x64211c2f, r.load<1>(&data[3]));
		}
	}
	{
		hkSimdReal r;
		HK_ALIGN16( hkReal data[4]) = {0.0f, 1.0f, -2.0f, 0.0f};
		{
			r.load<1,HK_IO_NOT_CACHED>( &data[0] );
			HK_TEST( r.getReal() == data[0] );
// 			HK_TEST_ASSERT(0x64211c2f, r.loadNotCached<1>(&data[1]));
// 			HK_TEST_ASSERT(0x64211c2f, r.loadNotCached<1>(&data[2]));
// 			HK_TEST_ASSERT(0x64211c2f, r.loadNotCached<1>(&data[3]));
		}
	}
}


static void simdreal_comparisons()
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

	// isSignBitSet
	{
		hkSimdReal c;
		c.setFromFloat(1.0f);
		HK_TEST(!c.isSignBitSet());
		c.setFromFloat(-1.0f);
		HK_TEST(c.isSignBitSet());
		c.setFromFloat(0.0f);
		HK_TEST(!c.isSignBitSet());
		HK_TEST(minusZero.isSignBitSet());
	}

	// isSignBitClear
	{
		hkSimdReal c;
		c.setFromFloat(1.0f);
		HK_TEST(c.isSignBitClear());
		c.setFromFloat(-1.0f);
		HK_TEST(!c.isSignBitClear());
		c.setFromFloat(0.0f);
		HK_TEST(c.isSignBitClear());
		HK_TEST(!minusZero.isSignBitClear());
	}

	// isEqualZero
	{
		hkSimdReal c;
		c.setFromFloat(1.0f);
		HK_TEST(!c.isEqualZero());
		c.setFromFloat(-1.0f);
		HK_TEST(!c.isEqualZero());
		c.setFromFloat(0.0f);
		HK_TEST(c.isEqualZero());
		HK_TEST(minusZero.isEqualZero());
	}

	// isNotEqualZero
	{
		hkSimdReal c;
		c.setFromFloat(1.0f);
		HK_TEST(c.isNotEqualZero());
		c.setFromFloat(-1.0f);
		HK_TEST(c.isNotEqualZero());
		c.setFromFloat(0.0f);
		HK_TEST(!c.isNotEqualZero());
		HK_TEST(!minusZero.isNotEqualZero());
	}

	// isOk
	{
		hkSimdReal c;
		HK_ALIGN16(const unsigned int temp[4]) = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
		union { const unsigned int* i; hkSingleReal* r; } t2s; t2s.i = temp;
		c.m_real = *t2s.r;
		HK_TEST(!c.isOk());
		c.setFromFloat(-1.0f);
		HK_TEST(c.isOk());
		c.setFromFloat(0.0f);
		HK_TEST(c.isOk());
		HK_TEST(minusZero.isOk());
	}

	// compare
	{
		hkSimdReal a; a.setFromFloat(-1.0f);
		hkSimdReal b = minusZero;
		hkSimdReal c; c.setFromFloat(0.0f);
		hkSimdReal d; d.setFromFloat(1.0f);
		
		HK_TEST(a.less(b).allAreSet());
		HK_TEST(a.less(c).allAreSet());
		HK_TEST(a.less(d).allAreSet());
		HK_TEST(!b.less(c).allAreSet());
		HK_TEST(!c.less(b).allAreSet());
		HK_TEST(!d.less(a).allAreSet());

		HK_TEST(a.lessEqual(b).allAreSet());
		HK_TEST(a.lessEqual(c).allAreSet());
		HK_TEST(a.lessEqual(d).allAreSet());
		HK_TEST(b.lessEqual(c).allAreSet());
		HK_TEST(c.lessEqual(b).allAreSet());
		HK_TEST(!d.lessEqual(a).allAreSet());

		HK_TEST(!a.greater(b).allAreSet());
		HK_TEST(!a.greater(c).allAreSet());
		HK_TEST(!a.greater(d).allAreSet());
		HK_TEST(!b.greater(c).allAreSet());
		HK_TEST(!c.greater(b).allAreSet());
		HK_TEST(d.greater(a).allAreSet());
		HK_TEST(c.greater(a).allAreSet());
		HK_TEST(b.greater(a).allAreSet());

		HK_TEST(!a.greaterEqual(b).allAreSet());
		HK_TEST(!a.greaterEqual(c).allAreSet());
		HK_TEST(!a.greaterEqual(d).allAreSet());
		HK_TEST(b.greaterEqual(c).allAreSet());
		HK_TEST(c.greaterEqual(b).allAreSet());
		HK_TEST(d.greaterEqual(a).allAreSet());
		HK_TEST(c.greaterEqual(a).allAreSet());
		HK_TEST(b.greaterEqual(a).allAreSet());

		HK_TEST(!a.equal(b).allAreSet());
		HK_TEST(!a.equal(c).allAreSet());
		HK_TEST(!a.equal(d).allAreSet());
		HK_TEST(b.equal(c).allAreSet());
		HK_TEST(c.equal(b).allAreSet());
		HK_TEST(!d.equal(a).allAreSet());
		HK_TEST(!c.equal(a).allAreSet());
		HK_TEST(!b.equal(a).allAreSet());

		HK_TEST(a.notEqual(b).allAreSet());
		HK_TEST(a.notEqual(c).allAreSet());
		HK_TEST(a.notEqual(d).allAreSet());
		HK_TEST(!b.notEqual(c).allAreSet());
		HK_TEST(!c.notEqual(b).allAreSet());
		HK_TEST(d.notEqual(a).allAreSet());
		HK_TEST(c.notEqual(a).allAreSet());
		HK_TEST(b.notEqual(a).allAreSet());

		HK_TEST(a.lessZero().allAreSet());
		HK_TEST(!b.lessZero().allAreSet());
		HK_TEST(!c.lessZero().allAreSet());
		HK_TEST(!d.lessZero().allAreSet());

		HK_TEST(!a.greaterZero().allAreSet());
		HK_TEST(!b.greaterZero().allAreSet());
		HK_TEST(!c.greaterZero().allAreSet());
		HK_TEST(d.greaterZero().allAreSet());

		HK_TEST(!a.greaterEqualZero().allAreSet());
		HK_TEST(b.greaterEqualZero().allAreSet());
		HK_TEST(c.greaterEqualZero().allAreSet());
		HK_TEST(d.greaterEqualZero().allAreSet());
	}

	// approxEqual
	{
		hkSimdReal eps; eps.setFromFloat(0.125f);      // 1/8
		hkSimdReal eps2; eps2.setFromFloat(0.015625f); // 1/64
		hkSimdReal a; a.setFromFloat(8.0f);
		hkSimdReal b;
		b = a + eps + eps2;
		HK_TEST(!a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
		HK_TEST(!a.approxEqual(b, eps+eps2));
		b = a + eps - eps2;
		HK_TEST(a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
		b = a + eps;
		HK_TEST(!a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
		b = a + eps2;
		HK_TEST(a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));

		b = a - eps - eps2;
		HK_TEST(!a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
		HK_TEST(!a.approxEqual(b, eps+eps2));
		b = a - eps + eps2;
		HK_TEST(a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
		b = a - eps;
		HK_TEST(!a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
		b = a - eps2;
		HK_TEST(a.approxEqual(b, eps));
		HK_TEST(!a.approxEqual(b, eps2));
	}
}

static void simdreal_ops()
{
	{ // simple operators
		hkSimdReal half; half.setFromFloat(0.5f);
		hkSimdReal ofive; ofive.setFromFloat(1.5f);
		HK_TEST( hkMath::equal( (half + ofive).getReal(),  2.0f) );
		HK_TEST( hkMath::equal( (half - ofive).getReal(), -1.0f) );
		HK_TEST( hkMath::equal( (half * ofive).getReal(), 0.75f) );
		HK_TEST( hkMath::equal( (ofive / half).getReal(), 3.0f) );
		HK_TEST( hkMath::equal( (-ofive).getReal(), -1.5f) );
	}

	{ // real times simdreal
		hkSimdReal r0; r0.setFromFloat(99.0f);
		hkSimdReal r1 = hkSimdReal::getConstant<HK_QUADREAL_2>() * r0;
		HK_TEST( hkMath::equal(r1.getReal(), 2.0f*99.0f));
	}

	{
		hkSimdReal sa; sa.setFromFloat(1.0f);
		hkSimdReal sb; sb.setFromFloat(10.0f);
		hkReal fa = 5.0f;
		hkSimdReal sfa; sfa.setFromFloat(fa);
		hkSimdReal sc = sa - sb * sfa;

		HK_TEST( sc.getReal() == 1.0f - 10.0f*5.0f );
		hkReal fb = sc.getReal() * 2.0f;
		HK_TEST( hkMath::equal(fb, 2.0f*(1.0f - 10.0f*5.0f) ));
	}

	// add
	hkSimdReal ofive; ofive.setFromFloat(1.5f);
	{
		hkSimdReal a; a.setFromFloat(1.0f); a.add(ofive);
		HK_TEST(hkMath::equal(a.getReal(), 2.5f));
	}

	// sub
	{
		hkSimdReal a; a.setFromFloat(1.0f); a.sub(ofive);
		HK_TEST(hkMath::equal(a.getReal(), -0.5f));
	}

	// mul
	{
		hkSimdReal a; a.setFromFloat(2.0f); a.mul(ofive);
		HK_TEST(hkMath::equal(a.getReal(), 3.0f));
	}

	// div
	{
		hkSimdReal a; a.setFromFloat(1.0f); a.div<HK_ACC_FULL,HK_DIV_SET_ZERO>(hkSimdReal::getConstant<HK_QUADREAL_INV_2>());
		HK_TEST(hkMath::equal(a.getReal(), 2.0f));
	}

	// div_bit accurate
	{
		hkPseudoRandomGenerator random(10);
		for (int i =0; i < 100; i++)
		{
			hkSimdReal a; a.setFromFloat(random.getRandReal11());
			hkSimdReal b; b.setFromFloat(random.getRandReal11());
			hkSimdReal c(a), d(b);
			hkSimdReal e; e.setFromFloat(random.getRandReal11());
			{
				a.div<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(e); c.div<HK_ACC_FULL,HK_DIV_SET_ZERO>(e);
				b.div<HK_ACC_12_BIT,HK_DIV_SET_ZERO>(e); d.div<HK_ACC_FULL,HK_DIV_SET_ZERO>(e);
				
#ifndef HK_PLATFORM_SPU // "full" accuracy divide on SPU isn't very accurate - see COM-1656
				checkEqual23Bit(a, c);
#endif
				checkEqual12Bit(b, d);
			}
		}
	}

	// setadd
	{
		hkSimdReal c;
		hkSimdReal a; a.setFromFloat(1.0f); c.setAdd(a, ofive);
		HK_TEST(hkMath::equal(c.getReal(), 2.5f));
	}

	// setsub
	{
		hkSimdReal c;
		hkSimdReal a; a.setFromFloat(1.0f); c.setSub(a, ofive);
		HK_TEST(hkMath::equal(c.getReal(), -0.5f));
	}

	// setmul
	{
		hkSimdReal c;
		hkSimdReal a; a.setFromFloat(2.0f); c.setMul(a, ofive);
		HK_TEST(hkMath::equal(c.getReal(), 3.0f));
	}

	// setdiv
	{
		hkSimdReal c;
		hkSimdReal a; a.setFromFloat(1.0f); c.setDiv<HK_ACC_FULL,HK_DIV_SET_ZERO>(a, hkSimdReal::getConstant<HK_QUADREAL_INV_2>());
		HK_TEST(hkMath::equal(c.getReal(), 2.0f));
	}

	// setdiv_bit accurate
	{
		hkPseudoRandomGenerator random(10);
		for (int i =0; i < 100; i++)
		{
			hkSimdReal a; a.setFromFloat(random.getRandReal11());
			hkSimdReal b; b.setFromFloat(random.getRandReal11());
			hkSimdReal c(a), d(b);
			hkSimdReal e; e.setFromFloat(random.getRandReal11());
			hkSimdReal r1, r2, r3, r4;
			{
				r1.setDiv<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(a, e); r2.setDiv<HK_ACC_FULL,HK_DIV_SET_ZERO>(c, e);
				r3.setDiv<HK_ACC_12_BIT,HK_DIV_SET_ZERO>(b, e); r4.setDiv<HK_ACC_FULL,HK_DIV_SET_ZERO>(d, e);
#ifndef HK_PLATFORM_SPU // "full" accuracy divide on SPU isn't very accurate - see COM-1656
				checkEqual23Bit(r1, r2);
#endif
				checkEqual12Bit(r3, r4);
			}
		}
	}


	// Special reciprocals
	{
		const int NUM_TIMES = 100;
		{
			hkSimdReal x;
			const hkSimdReal one = hkSimdReal::getConstant<HK_QUADREAL_1>();
			hkSimdReal outFull;
			hkSimdReal out12;
			hkSimdReal out23;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.setFromFloat(hkUnitTest::rand01());
				out12.setDiv<HK_ACC_12_BIT,HK_DIV_SET_ZERO>(x,one); 
				out23.setDiv<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(x,one); 
				outFull.setDiv<HK_ACC_FULL,HK_DIV_SET_ZERO>(x,one);
				checkEqual12Bit(outFull, out12);
				checkEqual23Bit(outFull, out23);
			}
		}
	}

	// addmul
	{
		hkSimdReal a; a.setFromFloat(1.0f);
		a.addMul(hkSimdReal::getConstant<HK_QUADREAL_3>(), ofive);
		HK_TEST(hkMath::equal(a.getReal(), 5.5f));
	}

	// submul
	{
		hkSimdReal a; a.setFromFloat(1.0f);
		a.subMul(hkSimdReal::getConstant<HK_QUADREAL_3>(), ofive);
		HK_TEST(hkMath::equal(a.getReal(), -3.5f));
	}

	// setAddMul
	{
		hkSimdReal c;
		hkSimdReal a; a.setFromFloat(1.0f); c.setAddMul(a, hkSimdReal::getConstant<HK_QUADREAL_3>(), ofive);
		HK_TEST(hkMath::equal(c.getReal(), 5.5f));
	}

	// setSubMul
	{
		hkSimdReal c;
		hkSimdReal a; a.setFromFloat(1.0f); c.setSubMul(a, hkSimdReal::getConstant<HK_QUADREAL_3>(), ofive);
		HK_TEST(hkMath::equal(c.getReal(), -3.5f));
	}

	// setInterpolate
	{
		hkSimdReal eleven; eleven.setFromFloat(11.0f);
		hkSimdReal c;
		c.setInterpolate(hkSimdReal::getConstant<HK_QUADREAL_1>(), eleven, hkSimdReal::getConstant<HK_QUADREAL_0>());
		HK_TEST(hkMath::equal(c.getReal(), 1.0f));
		c.setInterpolate(hkSimdReal::getConstant<HK_QUADREAL_1>(), eleven, hkSimdReal::getConstant<HK_QUADREAL_1>());
		HK_TEST(hkMath::equal(c.getReal(), 11.0f));
		c.setInterpolate(hkSimdReal::getConstant<HK_QUADREAL_1>(), eleven, hkSimdReal::getConstant<HK_QUADREAL_INV_2>());
		HK_TEST(hkMath::equal(c.getReal(), 6.0f));
	}
}

static void simdreal_square_roots_recip ()
{
	// Note that some checks are disabled on SPU, since it never generates INF/NAN values.
	// We can check that something IS ok, but not that it's NOT ok.

	// sqrt
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkMath::equal(a.sqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 2.0f));
		HK_TEST(hkMath::equal(b.sqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 3.0f));
		HK_TEST(hkMath::equal(c.sqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 4.0f));
		HK_TEST(hkMath::equal(d.sqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
		HK_TEST(hkMath::equal(e.sqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
	}

	// sqrt_12Bit accurate
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkIsEqual12BitAccurate(a.sqrt<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f));
		HK_TEST(hkIsEqual12BitAccurate(b.sqrt<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 3.0f));
		HK_TEST(hkIsEqual12BitAccurate(c.sqrt<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 4.0f));
		HK_TEST(hkIsEqual12BitAccurate(d.sqrt<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
		HK_TEST(hkIsEqual12BitAccurate(e.sqrt<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
	}

	// sqrt_23Bit accurate
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkIsEqual23BitAccurate(a.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 2.0f));
		HK_TEST(hkIsEqual23BitAccurate(b.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 3.0f));
		HK_TEST(hkIsEqual23BitAccurate(c.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 4.0f));
		HK_TEST(hkIsEqual23BitAccurate(d.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
		HK_TEST(hkIsEqual23BitAccurate(e.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
	}

	// sqrtInverse
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkMath::equal(a.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 2.0f));
		HK_TEST(hkMath::equal(b.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 3.0f));
		HK_TEST(hkMath::equal(c.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 4.0f));
		HK_TEST(hkMath::equal(d.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
		HK_TEST(hkMath::equal(e.sqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
	}
	
	// sqrtInverse12Bit accurate
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkIsEqual12BitAccurate(a.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 2.0f));
		HK_TEST(hkIsEqual12BitAccurate(b.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 3.0f));
		HK_TEST(hkIsEqual12BitAccurate(c.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 4.0f));
		HK_TEST(hkMath::equal(d.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
		HK_TEST(hkMath::equal(e.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
	}

	// sqrtInverse23Bit accurate
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkIsEqual23BitAccurate(a.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 2.0f));
		HK_TEST(hkIsEqual23BitAccurate(b.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 3.0f));
		HK_TEST(hkIsEqual23BitAccurate(c.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 1.0f / 4.0f));
		HK_TEST(hkMath::equal(d.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
		HK_TEST(hkMath::equal(e.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>().getReal(), 0.0f));
	}

	// sqrtInverseNonZero
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkMath::equal(a.sqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>().getReal(), 1.0f / 2.0f));
		HK_TEST(hkMath::equal(b.sqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>().getReal(), 1.0f / 3.0f));
		HK_TEST(hkMath::equal(c.sqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>().getReal(), 1.0f / 4.0f));
		//HK_TEST((d.sqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>().isOk() == false));
		HK_ON_CPU( HK_TEST((e.sqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>().isOk() == hkFalse32)) );
	}

	// sqrtInverseNonZero_12BitAccurate
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkIsEqual12BitAccurate(a.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>().getReal(), 1.0f / 2.0f));
		HK_TEST(hkIsEqual12BitAccurate(b.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>().getReal(), 1.0f / 3.0f));
		HK_TEST(hkIsEqual12BitAccurate(c.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>().getReal(), 1.0f / 4.0f));
		//HK_TEST((d.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>().isOk() == false));
		//HK_ON_CPU( HK_TEST((e.sqrtInverse<HK_ACC_12_BIT,HK_SQRT_IGNORE>().isOk() == hkFalse32)) );
	}

	// sqrtInverseNonZero_23BitAccurate
	{
		hkSimdReal a; a.setFromFloat(4.0f);
		hkSimdReal b; b.setFromFloat(9.0f);
		hkSimdReal c; c.setFromFloat(16.0f);
		hkSimdReal d; d.setFromFloat(0.0f);
		hkSimdReal e; e.setFromFloat(-4.0f);
		HK_TEST(hkIsEqual23BitAccurate(a.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>().getReal(), 1.0f / 2.0f));
		HK_TEST(hkIsEqual23BitAccurate(b.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>().getReal(), 1.0f / 3.0f));
		HK_TEST(hkIsEqual23BitAccurate(c.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>().getReal(), 1.0f / 4.0f));
		//HK_TEST((d.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>().isOk() == false));
		//HK_ON_CPU( HK_TEST((e.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>().isOk() == hkFalse32)) );
	}

	// Testing functionality of setReciprocal()
	{
		const int NUM_TIMES = 100;
		{
			hkSimdReal x;
			hkSimdReal y;
			hkSimdReal out;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.setFromFloat(hkUnitTest::rand01());
				y.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(x);
				out.setMul(x,y);
				HK_TEST( hkMath::equal(out.getReal(), 1.0f) );
			}
		}
	}

	// Testing functionality of setReciprocal()
	{
		const int NUM_TIMES = 100;
		{
			hkSimdReal x;
			hkSimdReal y;
			hkSimdReal out;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.setFromFloat(hkUnitTest::rand01());
				y.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(x); 
				out.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(x);
				checkEqual23Bit(y, out);
			}
		}
	}

	// Testing functionality of setReciprocal_12BitAccurate()
	{
		const int NUM_TIMES = 100;
		{
			hkSimdReal x;
			hkSimdReal y;
			hkSimdReal out;
			for(int i = 0; i < NUM_TIMES; i++)
			{
				x.setFromFloat(hkUnitTest::rand01());
				y.setReciprocal<HK_ACC_12_BIT,HK_DIV_SET_ZERO>(x); 
				out.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(x);
				checkEqual12Bit(y, out);
			}
		}
	}

	// Special reciprocals
	{
		const hkSimdReal one = hkSimdReal::getConstant<HK_QUADREAL_1>();
		hkSimdReal oneOverOneFull;
		hkSimdReal oneOverOne12;
		hkSimdReal oneOverOne23;
		oneOverOne12.setReciprocal<HK_ACC_12_BIT,HK_DIV_SET_ZERO>(one); 
		oneOverOne23.setReciprocal<HK_ACC_23_BIT,HK_DIV_SET_ZERO>(one); 
		oneOverOneFull.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO>(one);
		checkEqual12Bit(oneOverOneFull, oneOverOne12);
		checkEqual23Bit(oneOverOneFull, oneOverOne23);
	}
	{
		const hkSimdReal one = hkSimdReal::getConstant<HK_QUADREAL_1>();
		const hkSimdReal epsSmall = hkSimdReal::getConstant<HK_QUADREAL_EPS_SQRD>(); // value smaller than machine eps
		const hkSimdReal epsLarge = hkSimdReal::getConstant<HK_QUADREAL_EPS>() + hkSimdReal::getConstant<HK_QUADREAL_EPS>(); // value larger than machine eps
		
		hkSimdReal almostOneSmall = one + epsSmall;
		hkSimdReal almostOneLarge = one + epsLarge;
		{
			// small delta should be clamped to 1
			hkSimdReal oneOverOneSmall;
			oneOverOneSmall.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO_AND_ONE>(almostOneSmall);
			HK_TEST(oneOverOneSmall.isEqual(one));

			// large delta should produce the actual reciprocal value
			hkSimdReal oneOverOneLarge;
			oneOverOneLarge.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO_AND_ONE>(almostOneLarge);
			hkSimdReal oneOverOneNoClamp;
			oneOverOneNoClamp.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(almostOneLarge);
			HK_TEST(oneOverOneLarge.isEqual(oneOverOneNoClamp));
		}

		// test abs range
		almostOneSmall = one - epsSmall;
		almostOneLarge = one - epsLarge;
		{
			// small delta should be clamped to 1
			hkSimdReal oneOverOneSmall;
			oneOverOneSmall.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO_AND_ONE>(almostOneSmall);
			HK_TEST(oneOverOneSmall.isEqual(one));

			// large delta should produce the actual reciprocal value
			hkSimdReal oneOverOneLarge;
			oneOverOneLarge.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO_AND_ONE>(almostOneLarge);
			hkSimdReal oneOverOneNoClamp;
			oneOverOneNoClamp.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(almostOneLarge);
			HK_TEST(oneOverOneLarge.isEqual(oneOverOneNoClamp));
		}

		// test zero clamp
		{
			hkSimdReal zero; zero.setZero();
			hkSimdReal oneOverOne;
			oneOverOne.setReciprocal<HK_ACC_FULL,HK_DIV_SET_ZERO_AND_ONE>(zero);
			HK_TEST(oneOverOne.isEqual(zero));
		}
	}
}

int simdreal_main()
{
#if !defined(HK_PLATFORM_LINUX) | !defined(HK_ARCH_X64)
	simdreal_test();
	simdreal_setget_basic();
	simdreal_assign_special();
	simdreal_load_store();
	simdreal_comparisons();
	simdreal_ops();
	simdreal_square_roots_recip();
#endif

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(simdreal_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/simdreal.cpp"     );

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
