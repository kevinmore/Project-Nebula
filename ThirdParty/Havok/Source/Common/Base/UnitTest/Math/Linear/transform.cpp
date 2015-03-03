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
#include <Common/Base/Math/Vector/hkVector4Util.h>

static void is_identity()
{
	//make up a space

	hkVector4 rand_first_row; rand_first_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
	rand_first_row.normalize<3>();
	hkVector4 rand_second_row;
	hkVector4Util::calculatePerpendicularVector( rand_first_row, rand_second_row);
	rand_second_row.normalize<3>();

	hkVector4 rand_third_row;
	rand_third_row.setCross( rand_first_row, rand_second_row );

	hkRotation rand_rotation;
	rand_rotation.setRows( rand_first_row, rand_second_row, rand_third_row );

	hkVector4 rand_translation; rand_translation.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );

	hkTransform rand_transform; rand_transform.set( rand_rotation, rand_translation );
	hkTransform rand_inverse;
	rand_inverse.setInverse( rand_transform );

	hkTransform should_be_identity;
	should_be_identity.setMul( rand_transform, rand_inverse );
	hkTransform idT; idT.setIdentity();
	HK_TEST( should_be_identity.isApproximatelyEqual( idT ) );

	should_be_identity.setMulMulInverse( rand_transform, rand_transform );
	HK_TEST( should_be_identity.isApproximatelyEqual( idT ) );

	should_be_identity.setMulInverseMul( rand_transform, rand_transform );
	HK_TEST( should_be_identity.isApproximatelyEqual( idT ) );
}

static void transform_test()
{
	hkVector4 r1;
	r1.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
	r1.normalize<3>();
	hkVector4 r2;
	r2.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
	r2.normalize<3>();
	hkVector4 r3;
	r3.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
	r3.normalize<3>();
	hkRotation rot;
	rot.setRows( r1, r2, r3 );
	hkQuadReal vtrVal = HK_QUADREAL_CONSTANT(1, 2, 3, 0.0f);
	hkVector4 vtr; vtr.m_quad = vtrVal;

	// verification of constructor
	hkTransform tran; tran.set(rot,vtr);
	HK_TEST(tran.getTranslation().isOk<3>());
	hkMatrix3& rotMat_tr = tran.getRotation(); // cannot test as hkRotation because it is probably not orthogonal
	HK_TEST(rotMat_tr.isOk());

	// Verification test of copy constructor, getTranslation(), getRotation().
	{
		hkTransform tran_copy(tran);
		HK_TEST( tran_copy.isApproximatelyEqual(tran) );
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( tran_copy.getTranslation().allEqual<4>(tran.getTranslation(),eps) );
		HK_TEST( tran_copy.getRotation().isApproximatelyEqual(tran.getRotation()) );

		// Verification test of getColumn().
		for(int i = 0; i < 2; i++)
		{
			HK_TEST( tran_copy.getColumn(i).allEqual<4>(tran.getColumn(i),eps) );
		}
	}

	// Verification of set4x4ColumnMajor() and get4x4ColumnMajor()
	{
		hkTransform tran_copy(tran);
		{
			HK_ALIGN_REAL(hkReal columns[16]);
			tran.get4x4ColumnMajor(columns);
			HK_ALIGN_REAL(hkReal copy_columns[16]);
			tran_copy.get4x4ColumnMajor(copy_columns);
			for( int i = 0; i < 15; i++)
			{
				HK_TEST( columns[i] == copy_columns[i] );
			}
		}

		// Verification of set4x4ColumnMajor().
		{
			HK_ALIGN_REAL(hkReal columns[16]);
			for(int i = 0; i < 16; i++)
			{
				if ( i==3 || i==7 ||i==11 || i==15)
				{
					columns[i] = 0;
				}
				else
				{
					columns[i] = hkUnitTest::rand01();
				}
			}
			tran_copy.set4x4ColumnMajor(columns);
			HK_ALIGN_REAL(hkReal cols[16]);
			tran_copy.get4x4ColumnMajor(cols);
			for(int i = 1; i < 15; i++)
			{
				HK_TEST(cols[i] == columns[i]);

			}
		}
	}

	// Verification of setMulEq ().
	{
		hkTransform transform(tran);
		hkVector4 v0;
		v0.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v0.normalize<3>();
		hkVector4 v1;
		v1.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v1.normalize<3>();
		hkVector4 v2;
		v2.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v2.normalize<3>();
		hkRotation r;
		r.setRows( v0, v1, v2 );
		hkQuadReal tVal = HK_QUADREAL_CONSTANT(1, 2, 3, 0);
		hkVector4 t; t.m_quad = tVal;

		hkTransform tr1; tr1.set(r,t);
		HK_TEST(tr1.getTranslation().isOk<3>());
		hkMatrix3& rotMat = tr1.getRotation(); // cannot test as hkRotation because it is probably not orthogonal
		HK_TEST(rotMat.isOk());

		hkTransform trout;
		trout.setIdentity();
		trout.setMul(transform,tr1);

		transform.setMulEq(tr1);
		HK_TEST(transform.getTranslation().isOk<3>());
		hkMatrix3& rotMat2 = transform.getRotation(); // cannot test as hkRotation because it is probably not orthogonal
		HK_TEST(rotMat2.isOk());
		HK_TEST(trout.isApproximatelyEqual(transform));
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(trout.getTranslation().allEqual<3>(transform.getTranslation(),eps));
	}

	// Checking setTranslation()  & getTranslation() Non version..
	{
		hkVector4 v0;
		v0.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v0.normalize<3>();
		hkVector4 v1;
		v1.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v1.normalize<3>();
		hkVector4 v2;
		v2.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v2.normalize<3>();
		hkRotation r;
		r.setRows( v0, v1, v2 );
		hkQuadReal trnsVal = HK_QUADREAL_CONSTANT(1,2,3,0);
		hkVector4 trns; trns.m_quad = trnsVal;

		hkTransform tf; tf.set(r,trns);
		hkVector4 ov = tf.getTranslation();
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(ov.allEqual<3>(trns,eps));

		hkQuadReal trns1Val = HK_QUADREAL_CONSTANT(2,3,4,0);
		hkVector4 trns1; trns1.m_quad = trns1Val;
		tf.setTranslation(trns1);
		ov = tf.getTranslation();
		HK_TEST(ov.allEqual<3>(trns1,eps));

	}

	// Checking getTranslation() const version.
	{
		hkVector4 v0;
		v0.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v0.normalize<3>();

		hkVector4 v1;
		v1.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v1.normalize<3>();

		hkVector4 v2;
		v2.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
		v2.normalize<3>();
		hkRotation r;
		r.setRows( v0, v1, v2 );
		hkQuadReal trnsVal = HK_QUADREAL_CONSTANT(1,2,3,0);
		hkVector4 trns; trns.m_quad = trnsVal;

		hkTransform tf; tf.set(r,trns);
		hkVector4 ov = tf.getTranslation();
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(ov.allEqual<3>(trns,eps));
	}

	// Checking setRotation() using a hkRotation.
	{
		hkVector4 a;
		a.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		a.normalize<3>();

		hkVector4 b;
		b.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		b.normalize<3>();

		hkVector4 c;
		c.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		c.normalize<3>();

		hkRotation r;
		r.setCols(a,b,c);

		hkTransform t(tran);
		t.setRotation(r);

		hkRotation ro = t.getRotation();
		HK_TEST( ro.isApproximatelyEqual(r) );

	}

	// Checking setRotation() using a hkQuaternion.
	{
		hkVector4 axis0;
		axis0.set(1,2,3);
		axis0.normalize<3>();

		hkQuaternion q0; q0.setAxisAngle(axis0, 0.7f);

		hkTransform t(tran);
		t.setRotation(q0);

		hkRotation ro = t.getRotation();

		HK_TEST(ro.isOk());
		HK_TEST(! ro.isApproximatelyEqual(tran.getRotation()));
	}

	// Checking setRows4().
	{
		hkVector4 axis0;
		axis0.set(1,2,3,4);

		hkVector4 axis1;
		axis1.set(5,6,7,8);

		hkVector4 axis2;
		axis2.set(9,10,11,12);

		hkVector4 axis3;
		axis3.set(13,14,15,16);

		hkTransform t;
		t.setRows4(axis0,axis1,axis2,axis3);

		hkQuadReal vVal = HK_QUADREAL_CONSTANT(1,5,9,13);
		hkVector4 v; v.m_quad = vVal;
		hkVector4 v1 = t.getColumn(0);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(v.allEqual<4>(v1,eps));

		v.set(2,6,10,14);
		v1 = t.getColumn(1);
		HK_TEST(v.allEqual<4>(v1,eps));

		v.set(3,7,11,15);
		v1 = t.getColumn(2);
		HK_TEST(v.allEqual<4>(v1,eps));

		v.set(4,8,12,16);
		v1 = t.getColumn(3);
		HK_TEST(v.allEqual<4>(v1,eps));
	}

	// Checking  of setInverse().
	{
		hkTransform transform(tran);
		hkTransform tran_inv;
		tran_inv.setIdentity();
		tran_inv.setInverse(transform);
		HK_TEST(tran_inv.getTranslation().isOk<3>());
		hkMatrix3& rotMat = tran_inv.getRotation(); // cannot test as hkRotation because it is probably not orthogonal
		HK_TEST(rotMat.isOk());

		// Verification of translation component after Inverse operation
		hkVector4 translation_out = tran_inv.getTranslation();
		hkVector4 tr;
		tr.setNeg<4>(transform.getTranslation());
		hkVector4 translation_calc;
		hkRotation rot2;
		rot2.setTranspose(transform.getRotation());
		translation_calc._setRotatedDir(rot2,tr);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( translation_out.allEqual<3>(translation_calc,eps) );

		// Verification of rotation component after double Inverse operation
		hkTransform tran_out;
		tran_out.setIdentity();
		tran_out.setInverse(tran_inv);
		HK_TEST(tran_out.getTranslation().isOk<3>());
		hkMatrix3& rotMat2 = tran_out.getRotation(); // cannot test as hkRotation because it is probably not orthogonal
		HK_TEST(rotMat2.isOk());
		HK_TEST( tran_out.getRotation().isApproximatelyEqual(tran.getRotation()) );
	}

}


void qs_mul_transform()
{

	hkQsTransform qstrans;
	hkTransform tmp, tmpb, tmpc;
	hkVector4 tb;
	hkQuaternion qb;
	
	hkPseudoRandomGenerator random(15);

	random.getRandomRotation(qb);
	hkQuaternion qq; random.getRandomRotation(qq); qstrans.setRotation(qq);
	random.getRandomVector01(tb);
	tb(3) = 0;
	hkVector4 tt; random.getRandomVector01(tt);
	tt(3) = 0;
	qstrans.setTranslation(tt);

	qstrans.setScale(hkVector4::getConstant<HK_QUADREAL_1>());

	tmpb.set(qb, tb);

	hkTransform trans;
	qstrans.copyToTransform(trans);
	tmp.setMul(trans, tmpb);
	tmpc.setMul(qstrans, tmpb);

	hkSimdReal eps; eps.setFromFloat(1e-3f);
	for(int i=0; i<3; i++)
	{
		HK_TEST( tmp.getColumn(i).allEqual<3>( tmpc.getColumn(i),eps ) );
	}

	HK_TEST( tmp.getTranslation().allEqual<3>( tmpc.getTranslation(),eps ) );

}


int transform_main()
{
	is_identity();
	transform_test();
	qs_mul_transform();
	return 0;
}

//void ___1() { }
#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(transform_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/transform.cpp"     );

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
