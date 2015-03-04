/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

#include <Common/Base/Math/Vector/hkFourTransposedPoints.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

// #undef HK_TEST

#ifndef HK_TEST
#	define HK_TEST(x) if ( !(x) ) { testFailed(#x, __LINE__); }
#endif

//
//	Generates a random vector

static hkVector4 genVec(hkPseudoRandomGenerator& rng)
{
	hkVector4 v;
	rng.getRandomVectorRange(20.0f, v);
	return v;
}

//
//	Generates a random transposed vector

static hkFourTransposedPoints genFourPoints(hkPseudoRandomGenerator& rng)
{
	hkFourTransposedPoints v;
	rng.getRandomVectorRange(20.0f, v.m_vertices[0]);
	rng.getRandomVectorRange(20.0f, v.m_vertices[1]);
	rng.getRandomVectorRange(20.0f, v.m_vertices[2]);
	return v;
}

//
//	Tests two transposed points for equality

static void testEqual(const hkFourTransposedPoints& ftpa, const hkFourTransposedPoints& ftpb)
{
	HK_TEST(ftpa.m_vertices[0].allExactlyEqual<3>(ftpb.m_vertices[0]));
	HK_TEST(ftpa.m_vertices[1].allExactlyEqual<3>(ftpb.m_vertices[1]));
	HK_TEST(ftpa.m_vertices[2].allExactlyEqual<3>(ftpb.m_vertices[2]));
}

//
//	Tests two transposed points for equality

static void testApproxEqual(const hkFourTransposedPoints& ftpa, const hkFourTransposedPoints& ftpb, hkSimdRealParameter tolerance)
{
	HK_TEST(ftpa.m_vertices[0].allEqual<4>(ftpb.m_vertices[0], tolerance));
	HK_TEST(ftpa.m_vertices[1].allEqual<4>(ftpb.m_vertices[1], tolerance));
	HK_TEST(ftpa.m_vertices[2].allEqual<4>(ftpb.m_vertices[2], tolerance));
}

//
//	Tests the set / extract operations

static void testSetExtract(hkPseudoRandomGenerator& rng)
{
	const hkVector4 vA = genVec(rng);
	const hkVector4 vB = genVec(rng);
	const hkVector4 vC = genVec(rng);
	const hkVector4 vD = genVec(rng);

	hkFourTransposedPoints ftp0;
	ftp0.set(vA, vB, vC, vD);

	// Test set
	{
		hkFourTransposedPoints ftp1;
		ftp1.m_vertices[0].set(vA(0), vB(0), vC(0), vD(0));
		ftp1.m_vertices[1].set(vA(1), vB(1), vC(1), vD(1));
		ftp1.m_vertices[2].set(vA(2), vB(2), vC(2), vD(2));
		testEqual(ftp0, ftp1);
	}

	// Test extract
	{
		hkVector4 v0, v1, v2, v3;
		ftp0.extract(v0, v1, v2, v3);

		hkFourTransposedPoints ftp1;
		ftp1.m_vertices[0].set(v0(0), v1(0), v2(0), v3(0));
		ftp1.m_vertices[1].set(v0(1), v1(1), v2(1), v3(1));
		ftp1.m_vertices[2].set(v0(2), v1(2), v2(2), v3(2));
		testEqual(ftp0, ftp1);
	}

	// Test extract with W
#if !defined(HK_PLATFORM_RVL)
	{
		hkVector4 v0, v1, v2, v3;
		hkIntVector indices; indices.set(4,5,6,7);
		hkVector4 h; indices.storeInto24LowerBitsOfReal( h );
		ftp0.extractWithW( h, v0, v1, v2, v3);

		hkFourTransposedPoints ftp1;
		ftp1.m_vertices[0].set(v0(0), v1(0), v2(0), v3(0));
		ftp1.m_vertices[1].set(v0(1), v1(1), v2(1), v3(1));
		ftp1.m_vertices[2].set(v0(2), v1(2), v2(2), v3(2));
		testEqual(ftp0, ftp1);
		HK_TEST( v0.getInt24W() == 4 );
		HK_TEST( v1.getInt24W() == 5 );
		HK_TEST( v2.getInt24W() == 6 );
		HK_TEST( v3.getInt24W() == 7 );
	}
#endif

	// Test extract
	{
		hkVector4 v0, v1, v2, v3;
		ftp0.extract(0, v0);
		ftp0.extract(1, v1);
		ftp0.extract(2, v2);
		ftp0.extract(3, v3);

		hkFourTransposedPoints ftp1;
		ftp1.m_vertices[0].set(v0(0), v1(0), v2(0), v3(0));
		ftp1.m_vertices[1].set(v0(1), v1(1), v2(1), v3(1));
		ftp1.m_vertices[2].set(v0(2), v1(2), v2(2), v3(2));
		testEqual(ftp0, ftp1);
	}

	// Set all
	{
		ftp0.setAll(vA);

		hkVector4 v0, v1, v2, v3;
		ftp0.extract(v0, v1, v2, v3);

		hkFourTransposedPoints ftp1;
		ftp1.m_vertices[0].set(v0(0), v1(0), v2(0), v3(0));
		ftp1.m_vertices[1].set(v0(1), v1(1), v2(1), v3(1));
		ftp1.m_vertices[2].set(v0(2), v1(2), v2(2), v3(2));
		testEqual(ftp0, ftp1);
	}
}

//
//	Test arithmetic ops (i.e. add, sub, mul)

static void testArithmeticOps(hkPseudoRandomGenerator& rng)
{
	const hkFourTransposedPoints ftp0 = genFourPoints(rng);
	const hkFourTransposedPoints ftp1 = genFourPoints(rng);
	const hkVector4 vA = genVec(rng);

	const hkSimdReal sA = vA.getComponent<0>();

	// Extract verts
	hkVector4 v00, v01, v02, v03, v10, v11, v12, v13;
	ftp0.extract(v00, v01, v02, v03);
	ftp1.extract(v10, v11, v12, v13);

	hkFourTransposedPoints ftpa, ftpb;

	hkSimdReal tol;	tol.setFromFloat(1.0e-3f);

	// setSub(const hkFourTransposedPoints& v, hkVector4Parameter a)
	{
		ftpa.setSub(ftp0, vA);

		hkVector4 t0;	t0.setSub(v00, vA);
		hkVector4 t1;	t1.setSub(v01, vA);
		hkVector4 t2;	t2.setSub(v02, vA);
		hkVector4 t3;	t3.setSub(v03, vA);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}

	// setSub(hkVector4Parameter a, const hkFourTransposedPoints& v)
	{
		ftpa.setSub(vA, ftp0);

		hkVector4 t0;	t0.setSub(vA, v00);
		hkVector4 t1;	t1.setSub(vA, v01);
		hkVector4 t2;	t2.setSub(vA, v02);
		hkVector4 t3;	t3.setSub(vA, v03);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}

	// setSub(const hkFourTransposedPoints& a, const hkFourTransposedPoints& v)
	{
		ftpa.setSub(ftp0, ftp1);

		hkVector4 t0;	t0.setSub(v00, v10);
		hkVector4 t1;	t1.setSub(v01, v11);
		hkVector4 t2;	t2.setSub(v02, v12);
		hkVector4 t3;	t3.setSub(v03, v13);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}

	//	setAdd(const hkFourTransposedPoints& v, const hkFourTransposedPoints& a);
	{
		ftpa.setAdd(ftp0, ftp1);

		hkVector4 t0;	t0.setAdd(v00, v10);
		hkVector4 t1;	t1.setAdd(v01, v11);
		hkVector4 t2;	t2.setAdd(v02, v12);
		hkVector4 t3;	t3.setAdd(v03, v13);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}

	//	setMul(const hkFourTransposedPoints& v, hkVector4Parameter a);
	{
		ftpa.setMulC(ftp0, vA);

		hkVector4 t0;	t0.setMul(vA, v00);
		hkVector4 t1;	t1.setMul(vA, v01);
		hkVector4 t2;	t2.setMul(vA, v02);
		hkVector4 t3;	t3.setMul(vA, v03);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	//	setMul(const hkFourTransposedPoints& v, hkVector4Parameter a);
	{
		ftpa.setMulT(ftp0, vA);

		hkVector4 t0;	t0.setMul(vA.getComponent<0>(), v00);
		hkVector4 t1;	t1.setMul(vA.getComponent<1>(), v01);
		hkVector4 t2;	t2.setMul(vA.getComponent<2>(), v02);
		hkVector4 t3;	t3.setMul(vA.getComponent<3>(), v03);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	// setMul(const hkFourTransposedPoints& v, hkSimdRealParameter a)
	{
		ftpa.setMul(ftp0, sA);

		hkVector4 t0;	t0.setMul(v00, sA);
		hkVector4 t1;	t1.setMul(v01, sA);
		hkVector4 t2;	t2.setMul(v02, sA);
		hkVector4 t3;	t3.setMul(v03, sA);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	//	setAddMul(const hkFourTransposedPoints& v, const hkFourTransposedPoints& u, hkVector4Parameter a);
	{
		ftpa.setAddMulT(ftp0, ftp1, vA);

		hkVector4 t0;	t0.setAddMul(v00, v10, vA.getComponent<0>());
		hkVector4 t1;	t1.setAddMul(v01, v11, vA.getComponent<1>());
		hkVector4 t2;	t2.setAddMul(v02, v12, vA.getComponent<2>());
		hkVector4 t3;	t3.setAddMul(v03, v13, vA.getComponent<3>());

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol); // fmadd possible
	}

	//	setSubMul(const hkFourTransposedPoints& v, const hkFourTransposedPoints& u, hkVector4Parameter a);
	{
		ftpa.setSubMulT(ftp0, ftp1, vA);

		hkVector4 t0;	t0.setSubMul(v00, v10, vA.getComponent<0>());
		hkVector4 t1;	t1.setSubMul(v01, v11, vA.getComponent<1>());
		hkVector4 t2;	t2.setSubMul(v02, v12, vA.getComponent<2>());
		hkVector4 t3;	t3.setSubMul(v03, v13, vA.getComponent<3>());

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol); // fmadd possible
	}
	//	add(hkVector4Parameter a);
	{
		ftpa = ftp0;
		ftpa.add(vA);

		hkVector4 t0 = v00;	t0.add(vA);
		hkVector4 t1 = v01;	t1.add(vA);
		hkVector4 t2 = v02;	t2.add(vA);
		hkVector4 t3 = v03;	t3.add(vA);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	//	sub(hkVector4Parameter a);
	{
		ftpa = ftp0;
		ftpa.sub(vA);

		hkVector4 t0 = v00;	t0.sub(vA);
		hkVector4 t1 = v01;	t1.sub(vA);
		hkVector4 t2 = v02;	t2.sub(vA);
		hkVector4 t3 = v03;	t3.sub(vA);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	//	addMul(const hkFourTransposedPoints& u, hkVector4Parameter a);
	{
		ftpa = ftp0;
		ftpa.addMulT(ftp1, vA);

		hkVector4 t0 = v00;	t0.addMul(vA.getComponent<0>(), v10);
		hkVector4 t1 = v01;	t1.addMul(vA.getComponent<1>(), v11);
		hkVector4 t2 = v02;	t2.addMul(vA.getComponent<2>(), v12);
		hkVector4 t3 = v03;	t3.addMul(vA.getComponent<3>(), v13);

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol); // fmadd possible
	}

	//	mul(hkVector4Parameter a);
	{
		ftpa = ftp0;
		ftpa.mulT(vA);

		hkVector4 t0 = v00;	t0.mul(vA.getComponent<0>());
		hkVector4 t1 = v01;	t1.mul(vA.getComponent<1>());
		hkVector4 t2 = v02;	t2.mul(vA.getComponent<2>());
		hkVector4 t3 = v03;	t3.mul(vA.getComponent<3>());

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	//	mul(hkSimdRealParameter a);
	{
		ftpa = ftp0;
		ftpa.mul(sA);

		hkVector4 t0 = v00;	t0.mul(sA);
		hkVector4 t1 = v01;	t1.mul(sA);
		hkVector4 t2 = v02;	t2.mul(sA);
		hkVector4 t3 = v03;	t3.mul(sA);

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
	//	flipSigns(hkVector4Parameter a);
	{
		ftpa = ftp0;
		ftpa.flipSigns(vA);

		hkVector4 t0;	t0.setFlipSign(v00, vA.getComponent<0>());
		hkVector4 t1;	t1.setFlipSign(v01,	vA.getComponent<1>());
		hkVector4 t2;	t2.setFlipSign(v02, vA.getComponent<2>());
		hkVector4 t3;	t3.setFlipSign(v03, vA.getComponent<3>());

		ftpb.set(t0, t1, t2, t3);
		testEqual(ftpa, ftpb);
	}
}

//
//	Test dots / crosses / normalizations

static void testDotCross(hkPseudoRandomGenerator& rng)
{
	const hkFourTransposedPoints ftp0 = genFourPoints(rng);
	const hkFourTransposedPoints ftp1 = genFourPoints(rng);
	const hkVector4 vA = genVec(rng);

	// Extract verts
	hkVector4 v00, v01, v02, v03, v10, v11, v12, v13;
	ftp0.extract(v00, v01, v02, v03);
	ftp1.extract(v10, v11, v12, v13);

	hkFourTransposedPoints ftpa, ftpb;

	hkSimdReal tol;	tol.setFromFloat(1.0e-3f);

	// dot3(hkVector4Parameter a, hkVector4& dotOut)
	{
		hkVector4 d0, d1;

		ftp0.dot3(vA, d0);
		hkVector4Util::dot3_1vs4(vA, v00, v01, v02, v03, d1);
		
		HK_TEST(d0.allEqual<4>(d1, tol));
	}

	// dot3(const hkFourTransposedPoints& a, hkVector4& dotOut)
	{
		hkVector4 d0, d1;

		ftp0.dot3(ftp1, d0);
		d1.set(v00.dot<3>(v10), v01.dot<3>(v11), v02.dot<3>(v12), v03.dot<3>(v13));

		HK_TEST(d0.allEqual<4>(d1, tol));
	}

	// dot4xyz1(hkVector4Parameter a, hkVector4& dotOut)
	{
		hkVector4 d0, d1;

		ftp0.dot4xyz1(vA, d0);
		hkVector4Util::dot4xyz1_1vs4(vA, v00, v01, v02, v03, d1);
		
		HK_TEST(d0.allEqual<4>(d1, tol));
	}

	// setCross(hkVector4Parameter n, const hkFourTransposedPoints& v)
	{
		ftpa.setCross(vA, ftp0);

		hkVector4 t0;	t0.setCross(vA, v00);
		hkVector4 t1;	t1.setCross(vA, v01);
		hkVector4 t2;	t2.setCross(vA, v02);
		hkVector4 t3;	t3.setCross(vA, v03);

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol);
	}

	// normalize()
	{
		ftpa = ftp0;
		ftpa.normalize();

		hkVector4 t0 = v00;	t0.normalize<3>();
		hkVector4 t1 = v01;	t1.normalize<3>();
		hkVector4 t2 = v02;	t2.normalize<3>();
		hkVector4 t3 = v03;	t3.normalize<3>();

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol);
	}
}

//
//	Test outer products

static void testOuterProducts(hkPseudoRandomGenerator& rng)
{
	const hkFourTransposedPoints ftp0 = genFourPoints(rng);
	const hkVector4 v0 = genVec(rng);
	const hkVector4 v1 = genVec(rng);
	hkSimdReal tol;	tol.setFromFloat(1.0e-4f);

	hkFourTransposedPoints ftpa, ftpb;

	// setOuterProduct(hkVector4Parameter b, hkVector4Parameter c)
	{
		hkVector4 ip0, ip1, ip2, ip3;
		ip0.setMul(v0, v1.getComponent<0>());
		ip1.setMul(v0, v1.getComponent<1>());
		ip2.setMul(v0, v1.getComponent<2>());
		ip3.setMul(v0, v1.getComponent<3>());

		ftpa.setOuterProduct(v0, v1);

		ftpb.set(ip0, ip1, ip2, ip3);

		testApproxEqual(ftpa, ftpb, tol);
	}

	// addOuterProduct(hkVector4Parameter b, hkVector4Parameter c)
	{
		ftpa = ftp0;
		ftpa.addOuterProduct(v0, v1);

		ftpb.setOuterProduct(v0, v1);
		ftpb.setAdd(ftpb, ftp0);

		testApproxEqual(ftpa, ftpb, tol);
	}

	// subOuterProduct(hkVector4Parameter b, hkVector4Parameter c)
	{
		ftpa = ftp0;
		ftpa.subOuterProduct(v0, v1);

		ftpb.setOuterProduct(v0, v1);
		ftpb.mulT(hkVector4::getConstant<HK_QUADREAL_MINUS1>());
		ftpb.setAdd(ftpb, ftp0);

		testApproxEqual(ftpa, ftpb, tol);
	}
}


//
//	Tests rotations / transforms

static void testTransforms(hkPseudoRandomGenerator& rng)
{
	const hkFourTransposedPoints ftp0 = genFourPoints(rng);

	// Extract verts
	hkVector4 v00, v01, v02, v03;
	ftp0.extract(v00, v01, v02, v03);

	// Generate a random transform
	hkTransform tm;
	{
		hkQuaternion q;	rng.getRandomRotation(q);
		tm.setRotation(q);
		tm.setTranslation(genVec(rng));
	}

	hkSimdReal tol;	tol.setFromFloat(1.0e-4f);
	
	hkFourTransposedPoints ftpa, ftpb;

	// setTransformedInverseDir(const hkMatrix3& m, const hkFourTransposedPoints& v)
	{
		ftpa.setTransformedInverseDir(tm.getRotation(), ftp0);

		hkVector4 t0;	t0.setRotatedInverseDir(tm.getRotation(), v00);
		hkVector4 t1;	t1.setRotatedInverseDir(tm.getRotation(), v01);
		hkVector4 t2;	t2.setRotatedInverseDir(tm.getRotation(), v02);
		hkVector4 t3;	t3.setRotatedInverseDir(tm.getRotation(), v03);

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol);
	}

	// setTransformedInversePos(const hkTransform& m, const hkFourTransposedPoints& v)
	{
		ftpa.setTransformedInversePos(tm, ftp0);

		hkVector4 t0;	t0.setTransformedInversePos(tm, v00);
		hkVector4 t1;	t1.setTransformedInversePos(tm, v01);
		hkVector4 t2;	t2.setTransformedInversePos(tm, v02);
		hkVector4 t3;	t3.setTransformedInversePos(tm, v03);

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol);
	}

	// setTransformedInverseDir(const hkMatrix3& m, const hkFourTransposedPoints& v)
	{
		ftpa.setRotatedDir(tm.getRotation(), ftp0);

		hkVector4 t0;	t0.setRotatedDir(tm.getRotation(), v00);
		hkVector4 t1;	t1.setRotatedDir(tm.getRotation(), v01);
		hkVector4 t2;	t2.setRotatedDir(tm.getRotation(), v02);
		hkVector4 t3;	t3.setRotatedDir(tm.getRotation(), v03);

		ftpb.set(t0, t1, t2, t3);
		testApproxEqual(ftpa, ftpb, tol);
	}
}

//
//	Miscellaneous tests

static void testMisc(hkPseudoRandomGenerator& rng)
{
	const hkFourTransposedPoints ftp0 = genFourPoints(rng);
	const hkFourTransposedPoints ftp1 = genFourPoints(rng);
	//const hkVector4 vA = genVec(rng);

	// Extract verts
	hkVector4 v00, v01, v02, v03, v10, v11, v12, v13;
	ftp0.extract(v00, v01, v02, v03);
	ftp1.extract(v10, v11, v12, v13);

	hkFourTransposedPoints ftpa, ftpb;

	hkVector4Comparison mask;
	mask.set((hkVector4Comparison::Mask)rng.getRandChar(16));

	//	setSelect(hkVector4ComparisonParameter mask, const hkFourTransposedPoints& trueVecs, const hkFourTransposedPoints& falseVecs)
	{
		ftpa.setSelect(mask, ftp0, ftp1);

		hkVector4 t0 = mask.anyIsSet<hkVector4ComparisonMask::MASK_X>() ? v00 : v10;
		hkVector4 t1 = mask.anyIsSet<hkVector4ComparisonMask::MASK_Y>() ? v01 : v11;
		hkVector4 t2 = mask.anyIsSet<hkVector4ComparisonMask::MASK_Z>() ? v02 : v12;
		hkVector4 t3 = mask.anyIsSet<hkVector4ComparisonMask::MASK_W>() ? v03 : v13;
		ftpb.set(t0, t1, t2, t3);

		testEqual(ftpa, ftpb);
	}
}

//
//	Main entry point

int fourTransposedPoints_main()
{
	const int numTests = 1000;
	hkPseudoRandomGenerator rng(13);

	for (int i = 0; i < numTests; i++)
	{
		testSetExtract(rng);
		testArithmeticOps(rng);
		testDotCross(rng);
		testOuterProducts(rng);
		testTransforms(rng);
		testMisc(rng);
	}

	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(fourTransposedPoints_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/FourTransposedPoints.cpp"    );

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
