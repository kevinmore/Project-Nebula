/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#ifndef HK_PLATFORM_RVL // wii can't compile large int types
#include <Common/Base/Math/LargeInt/hkLargeIntTypes.h>

//	Test the set operations

static void testSetOps()
{
	// Component set
	{
		hkInt64Vector4 v;
		v.setComponent<0>(1);
		v.setComponent<1>(2);
		v.setComponent<2>(3);
		v.setComponent<3>(4);

		HK_TEST(v.getComponent<0>() == 1);
		HK_TEST(v.getComponent<1>() == 2);
		HK_TEST(v.getComponent<2>() == 3);
		HK_TEST(v.getComponent<3>() == 4);

		v.set(4, 3, 2, 1);
		HK_TEST(v.getComponent<0>() == 4);
		HK_TEST(v.getComponent<1>() == 3);
		HK_TEST(v.getComponent<2>() == 2);
		HK_TEST(v.getComponent<3>() == 1);

		hkSimdInt<128> iv;
		iv.setFromInt32(100);
		HK_TEST(iv.getWord<0>() == 100);
		HK_TEST(iv.getWord<1>() == 0);
		HK_TEST(iv.getWord<2>() == 0);
		HK_TEST(iv.getWord<3>() == 0);

		iv.setFromInt64(100);
		HK_TEST(iv.getWord<0>() == 100);
		HK_TEST(iv.getWord<1>() == 0);
		HK_TEST(iv.getWord<2>() == 0);
		HK_TEST(iv.getWord<3>() == 0);

		iv.setFromInt32(-100);
		HK_TEST(iv.getWord<0>() == -100);
		HK_TEST(iv.getWord<1>() == -1);
		HK_TEST(iv.getWord<2>() == -1);
		HK_TEST(iv.getWord<3>() == -1);

		iv.setFromInt64(-100);
		HK_TEST(iv.getWord<0>() == -100);
		HK_TEST(iv.getWord<1>() == -1);
		HK_TEST(iv.getWord<2>() == -1);
		HK_TEST(iv.getWord<3>() == -1);
	}
}

//
//	Test the arithmetic operations

static void testArithmeticOps()
{
	// Addition
	{
		hkInt64Vector4 a;	a.set(1, 2, 3, 4);
		hkInt64Vector4 b;	b.set(-5, -6, -7, -8);

		hkInt64Vector4 c;	c.setSub(a, b);
		HK_TEST(	(c.getComponent<0>() == 6) &&
					(c.getComponent<1>() == 8) &&
					(c.getComponent<2>() == 10) &&
					(c.getComponent<3>() == 12));

		hkSimdInt<128> ia;
		hkSimdInt<128> ib;
#if defined(HK_PLATFORM_PS3) || defined(HK_PLATFORM_ANDROID)
		ia.setFromInt64(0xFFFFFFFFFFFFLL);
		ib.setFromInt64(0xFFFFFFFFFFFFLL);
#else
		ia.setFromInt64(0xFFFFFFFFFFFFL);
		ib.setFromInt64(0xFFFFFFFFFFFFL);
#endif

		hkSimdInt<128> ic;	ic.setAdd(ia, ib);
		HK_TEST(	(ic.getWord<0>() == (int)0xFFFFFFFE) &&
					(ic.getWord<1>() == 0x1FFFF) &&
					(ic.getWord<2>() == 0) &&
					(ic.getWord<3>() == 0));

		ia.setFromInt32(-100);
		ib.setFromInt32(100);
		ic.setAdd(ia, ib);
		HK_TEST(	(ic.getWord<0>() == 0) &&
					(ic.getWord<1>() == 0) &&
					(ic.getWord<2>() == 0) &&
					(ic.getWord<3>() == 0));

		ia.setFromInt32(-100);
		ib.setFromInt32(2);
		ic.setAdd(ia, ib);
		HK_TEST(	(ic.getWord<0>() == -98) &&
					(ic.getWord<1>() == -1) &&
					(ic.getWord<2>() == -1) &&
					(ic.getWord<3>() == -1));

		ia.setFromInt32(-100);
		ib.setFromInt32(102);
		ic.setAdd(ia, ib);
		HK_TEST(	(ic.getWord<0>() == 2) &&
					(ic.getWord<1>() == 0) &&
					(ic.getWord<2>() == 0) &&
					(ic.getWord<3>() == 0));

#if defined(HK_PLATFORM_PS3) || defined(HK_PLATFORM_ANDROID)
		ia.setFromInt64(0xF904DA07B303383FLL);	// -0503037531755694017	= -0x06FB25F84CFCC7C1
		ib.setFromInt64(0x13678E394909DE00LL);	// +1398242586011491840	= +0x13678E394909DE00
#else
		ia.setFromInt64(0xF904DA07B303383FL);	// -0503037531755694017	= -0x06FB25F84CFCC7C1
		ib.setFromInt64(0x13678E394909DE00L);	// +1398242586011491840	= +0x13678E394909DE00
#endif

		ic.setAdd(ia, ib);						// +0895205054255797823 = +0x0C6C6840FC0D163F
		HK_TEST(	(ic.getWord<0>() == (int)0xFC0D163F) &&
					(ic.getWord<1>() == 0x0C6C6840) &&
					(ic.getWord<2>() == 0) &&
					(ic.getWord<3>() == 0));
	}

	// Subtraction
	{
		hkInt64Vector4 a;	a.set(1, 2, 3, 4);
		hkInt64Vector4 b;	b.set(5, 6, 7, 8);

		hkInt64Vector4 c;	c.setSub(b, a);
		HK_TEST(	(c.getComponent<0>() == 4) &&
					(c.getComponent<1>() == 4) &&
					(c.getComponent<2>() == 4) &&
					(c.getComponent<3>() == 4));
	}

	// Multiplication
	{
		hkIntVector a;	a.set(1, 2, 3, 4);
		hkIntVector b;	b.set(5, 6, 7, 8);

		hkInt64Vector4 c;	c.setMul(a, b);
		HK_TEST(	(c.getComponent<0>() == 5) &&
					(c.getComponent<1>() == 12) &&
					(c.getComponent<2>() == 21) &&
					(c.getComponent<3>() == 32));

		// Overflow test
		hkInt64Vector4 la;	la.set(1, 2, 3, 4);
		hkInt64Vector4 lb;	lb.set(5, 6, 7, 8);

		c.setUnsignedMul_128<2>(la, lb);
		hkSimdInt<128> xy;	c.storeXy(xy);
		hkSimdInt<128> zw;	c.storeZw(zw);
		HK_TEST((xy.getWord<3>() == 0) && (xy.getWord<2>() == 0) && (xy.getWord<1>() == 0) && (xy.getWord<0>() == 12));
		HK_TEST((zw.getWord<3>() == 0) && (zw.getWord<2>() == 0) && (zw.getWord<1>() == 0) && (zw.getWord<0>() == 32));
	}

	// 128-bit add
	{
		hkSimdInt<128> a;	a.setFromUint64((hkUint64)-1);
		hkSimdInt<128> b;	b.setFromUint64((hkUint64)-1);
		hkSimdInt<128> ab;	ab.setAdd(a, b);
		HK_TEST(ab.getWord<3>() == 0);
		HK_TEST(ab.getWord<2>() == 1);
		HK_TEST(ab.getWord<1>() == (int)0xFFFFFFFF);
		HK_TEST(ab.getWord<0>() == (int)0xFFFFFFFE);
	}

	// CLZ
	{
		int clz;
		clz = hkMath::countLeadingZeros<hkUint32>((hkUint32)1);		HK_TEST(clz == 31);
		clz = hkMath::countLeadingZeros<hkUint32>((hkUint32)-1);	HK_TEST(clz == 0);
		clz = hkMath::countLeadingZeros<hkUint32>((hkUint32)0);		HK_TEST(clz == 32);

		clz = hkMath::countLeadingZeros<hkUint64>((hkUint64)1);		HK_TEST(clz == 63);
		clz = hkMath::countLeadingZeros<hkUint64>((hkUint64)-1);	HK_TEST(clz == 0);
		clz = hkMath::countLeadingZeros<hkUint64>((hkUint64)0);		HK_TEST(clz == 64);

		hkSimdInt<128> i;
		i.setFromInt32(1);	clz = i.countLeadingZeros();	HK_TEST(clz == 127);
		i.setFromInt32(-1);	clz = i.countLeadingZeros();	HK_TEST(clz == 0);
		i.setZero();		clz = i.countLeadingZeros();	HK_TEST(clz == 128);
	}
}

//
//	Tests comparisons

static int testComparisons()
{
	hkInt64Vector4 v1;	v1.set(1, 4, 8, 7);
	hkInt64Vector4 v2;	v2.set(2, 4, 3, 6);

	hkVector4fComparison eq = v1.equal(v2);
	HK_TEST(eq.getMask()	== hkVector4fComparison::MASK_Y);
	
	v2.setComponent<0>(1);
	eq = v1.equal(v2);
	HK_TEST(eq.getMask()	== hkVector4fComparison::MASK_XY);

	v2.setComponent<2>(8);
	eq = v1.equal(v2);
	HK_TEST(eq.getMask()	== hkVector4fComparison::MASK_XYZ);

	v2.setComponent<3>(7);
	eq = v1.equal(v2);
	HK_TEST(eq.getMask()	== hkVector4fComparison::MASK_XYZW);

	{
		hkSimdInt<128> ia;	
		hkVector4fComparison cmp;
		ia.setFromInt32(-100);	cmp.setNot(ia.lessZero());
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(-1);	cmp.setNot(ia.lessZero());
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(0);		cmp.setNot(ia.lessZero());
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(1);		cmp.setNot(ia.lessZero());
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(100);	cmp.setNot(ia.lessZero());
		HK_TEST(cmp.allAreSet());
	}

	{
		hkSimdInt<128> ia, ib;	
		hkVector4fComparison cmp;
		ia.setFromInt32(-100); ib.setFromInt32(-2);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(-100); ib.setFromInt32(-200);	ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(-100); ib.setFromInt32(-100);	ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(-100); ib.setFromInt32(100);	ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(-100); ib.setFromInt32(0);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(-2);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(-200);	ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(-100);	ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(100);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(!cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(0);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(2);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(cmp.allAreSet());
		ia.setFromInt32(100); ib.setFromInt32(200);		ia.setSub(ib, ia);	cmp = ia.lessZero();
		HK_TEST(!cmp.allAreSet());
	}

	// Success!
	return 0;
}

//////////////////////////////////////////////////////////////////////////
//	Tests dot & cross

static int testDotCross()
{
	hkInt64Vector4 vA;	vA.set(1, 2, 3, 4);
	hkInt64Vector4 vB;	vB.set(8, 6, 1, 2);

	hkSimdInt<128> lenA, lenB;

	lenA = vA.dot<3>(vA);
	lenB = vB.dot<3>(vB);
	HK_TEST(	(lenA.getWord<0>() == 14) &&
				(lenA.getWord<1>() == 0) &&
				(lenA.getWord<2>() == 0) &&
				(lenA.getWord<3>() == 0));
	HK_TEST(	(lenB.getWord<0>() == 101) &&
				(lenB.getWord<1>() == 0) &&
				(lenB.getWord<2>() == 0) &&
				(lenB.getWord<3>() == 0));

	hkIntVector ivA;	ivA.set(1, 2, 3, 4);
	hkIntVector ivB;	ivB.set(8, 6, 1, 2);
	hkInt64Vector4 vC;	vC.setCross(ivA, ivB);
	HK_TEST(	(vC.getComponent<0>() == -16) &&
				(vC.getComponent<1>() == 23) &&
				(vC.getComponent<2>() == -10) &&
				(vC.getComponent<3>() == 0));

	// Success!
	return 0;
}

int LargeInt_main()
{
	{ 
		testSetOps();
		testArithmeticOps();
		testComparisons();
		testDotCross();
	}
	return 0;
}
#else //HK_PLATFORM_RVL
int LargeInt_main() { return 0; }
#endif

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#	pragma force_active on
#endif
HK_TEST_REGISTER(LargeInt_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
