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

static int NUM_TEST = 100;

// Generates random qs transforms with no scale

static hkPseudoRandomGenerator rGen(12343);
static void buildRandomQsTransformNoScale (hkQsTransform& transformOut)
{
	hkQuaternion q; rGen.getRandomRotation(q);
	transformOut.setRotation(q);

	hkVector4 t; rGen.getRandomVector11(t);
	hkSimdReal rr; rr.setFromFloat(rGen.getRandRange(1.0f, 10.0f));
	t.mul(rr);
	transformOut.setTranslation(t);

	hkVector4 s; s.setAll(1.0f);
	transformOut.setScale(s);
}

static void buildRandomQsTransform (hkQsTransform& transformOut)
{
	hkQuaternion q; rGen.getRandomRotation(q); transformOut.setRotation(q);

	hkVector4 t; rGen.getRandomVector11(t);
	hkSimdReal rr; rr.setFromFloat(rGen.getRandRange(1.0f, 10.0f));
	t.mul(rr);
	transformOut.setTranslation(t);

	// Make sure we don't have any <=0 scale
	hkVector4 s;
	s(0) = rGen.getRandRange(0.01f,5.0f);
	s(1) = rGen.getRandRange(0.01f,5.0f);
	s(2) = rGen.getRandRange(0.01f,5.0f);
	s(3) = 1;
	transformOut.setScale(s);
}

// TEST ZERO : Approxequal
static void check_zero()
{

	const hkReal tol = 0.01f;
	const hkReal eps = tol * 0.9f;

	
	for (int i=0; i < NUM_TEST; ++i)
	{
		hkQsTransform transform1;
		buildRandomQsTransform(transform1);

		hkQsTransform transform2 = transform1;
		{
			hkVector4 t2 = transform2.getTranslation();
			t2(0) += eps;
			t2(1) -= eps;
			t2(2) += eps;
			transform2.setTranslation(t2);
		}

		{
			hkQuaternion q2 = transform2.getRotation();
			q2.m_vec(0) = hkMath::clamp(q2.m_vec(0) + eps, hkReal(-1.0f), hkReal(1.0f));
			q2.m_vec(1) = hkMath::clamp(q2.m_vec(1) + eps, hkReal(-1.0f), hkReal(1.0f));
			q2.m_vec(2) = hkMath::clamp(q2.m_vec(2) + eps, hkReal(-1.0f), hkReal(1.0f));
			q2.m_vec(3) = hkMath::clamp(q2.m_vec(3) + eps, hkReal(-1.0f), hkReal(1.0f));
			transform2.setRotation(q2);
		}

		{
			hkVector4 qs = transform2.getScale();
			qs(0) += eps;
			qs(1) -= eps;
			qs(2) += eps;
			transform2.setScale(qs);
		}

		HK_TEST(transform1.isApproximatelyEqual(transform2, tol));
	}
	

	// And also explicitly check case which broke for COM-394
	hkQsTransform transform1; transform1.setIdentity();
	hkQuaternion q1;
	q1.m_vec.set(1.0f, 0.0f, 0.0f, 0.0001f);
	q1.normalize();
	transform1.setRotation(q1);

	hkQsTransform transform2 = transform1;
	hkQuaternion q2;
	q2.m_vec.set(1.0f, 0.0f, 0.0f, -0.0001f);
	q2.normalize();
	transform2.setRotation(q2);

	HK_TEST(transform1.isApproximatelyEqual(transform2, tol));
}

// TEST ONE : Inverses
static void check_one()
{

	hkQsTransform QS_IDENTITY; QS_IDENTITY.setIdentity();

	for (int i=0; i < NUM_TEST; ++i)
	{
		hkQsTransform transform;
		buildRandomQsTransform(transform);

		hkQsTransform inverse;
		inverse.setInverse(transform);

		HK_TEST( inverse.isOk() );
		
		// T * Inv(T) = I
		{
			hkQsTransform shouldBeIdentity;
			shouldBeIdentity.setMul(transform, inverse);
			HK_TEST(shouldBeIdentity.isApproximatelyEqual(QS_IDENTITY));
		}
	}

	for (int i=0; i < NUM_TEST; ++i)
	{
		hkQsTransform transform;
		buildRandomQsTransform(transform);

		hkQsTransform inverse;
		inverse.setInverse(transform);

		HK_TEST( inverse.isOk() );

		// Inv(T) * T = I
		{
			hkQsTransform shouldBeIdentity;
			shouldBeIdentity.setMul(inverse, transform);
			HK_TEST(shouldBeIdentity.isApproximatelyEqual(QS_IDENTITY));
		}
	}

}


// TEST TWO : Equivalence of operations with hkTransform (NO SCALE)
static void check_two()
{

	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		// Check symmetry of conversions
		{
			// T2QS (QS2T (qs)) = qs
			hkQsTransform test;
			test.setFromTransform(transform1);
			HK_TEST(test.isApproximatelyEqual(qsTransform1));
		}
		// Check symmetry of conversions without scale
		{
			// T2QS (QS2T (qs)) = qs
			hkTransform transformNoScale;
			qsTransform1.copyToTransformNoScale(transformNoScale);
			hkQsTransform test;
			test.setFromTransformNoScale(transformNoScale);
			HK_TEST(test.isApproximatelyEqual(qsTransform1));
		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		// Check inverses
		{
			hkTransform inverseT;
			inverseT.setInverse(transform1);
			hkQsTransform inverseQs;
			inverseQs.setInverse(qsTransform1);

			// T2QS ( Inv ( QS2T (qs) ) ) = Inv (qs)
			{
				hkQsTransform test;
				test.setFromTransform(inverseT);
				HK_TEST(test.isApproximatelyEqual(inverseQs));
			}

			// QS2T ( Inv (qs) ) = Inv ( QS2T (qs) )
			// (guaranteed by symmetry)
			{
				hkTransform test;
				inverseQs.copyToTransform(test);
				HK_TEST(test.isApproximatelyEqual(inverseT));
			}
		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		hkQsTransform qsTransform2;
		hkTransform transform2;
		buildRandomQsTransformNoScale(qsTransform2);
		qsTransform2.copyToTransform(transform2);

		// setMul
		{
			hkTransform mulmulT;
			hkQsTransform mulmulQs;

			mulmulT.setMul(transform1, transform2);
			mulmulQs.setMul(qsTransform1, qsTransform2);

			// T2QS ( QS2T(qs1) * QS2T (qs2) ) = qs1 * qs2
			{
				hkQsTransform test;
				test.setFromTransform(mulmulT);
				HK_TEST(test.isApproximatelyEqual(mulmulQs));
			}

			// QS2T ( qs1 * qs2 ) = QS2T (qs1) * QS2T(qs2)
			// (guaranteed by symmetry)
			{
				hkTransform test;
				mulmulQs.copyToTransform(test);
				HK_TEST(test.isApproximatelyEqual(mulmulT));
			}

		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		hkQsTransform qsTransform2;
		hkTransform transform2;
		buildRandomQsTransformNoScale(qsTransform2);
		qsTransform2.copyToTransform(transform2);

		// setMulEq
		{
			// t2 := t2 * t1 
			transform2.setMulEq(transform1);
			// qs2 := qs2 * qs1
			qsTransform2.setMulEq(qsTransform1);

			// QS2T (qs2) = t2
			{
				hkTransform test;
				qsTransform2.copyToTransform(test);
				HK_TEST(test.isApproximatelyEqual(transform2));
			}

			// T2QS (t2) = qs2
			{
				hkQsTransform test;
				test.setFromTransform(transform2);
				HK_TEST(test.isApproximatelyEqual(qsTransform2));
			}
		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		hkQsTransform qsTransform2;
		hkTransform transform2;
		buildRandomQsTransformNoScale(qsTransform2);
		qsTransform2.copyToTransform(transform2);

		// setMulInvMul 
		{
			hkTransform mulinvmulT;
			hkQsTransform mulinvmulQs;

			mulinvmulT.setMulInverseMul(transform1, transform2);
			mulinvmulQs.setMulInverseMul(qsTransform1, qsTransform2);

			// T2QS ( Inv(QS2T(qs1)) * QS2T (qs2) ) = Inv(qs1) * qs2
			{
				hkQsTransform test;
				test.setFromTransform(mulinvmulT);
				HK_TEST(test.isApproximatelyEqual(mulinvmulQs));
			}

			// QS2T ( Inv(qs1) * qs2 ) = Inv(QS2T (qs1)) * QS2T(qs2)
			// (guaranteed by symmetry)
			{
				hkTransform test;
				mulinvmulQs.copyToTransform(test);
				HK_TEST(test.isApproximatelyEqual(mulinvmulT));
			}

		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		hkQsTransform qsTransform2;
		hkTransform transform2;
		buildRandomQsTransformNoScale(qsTransform2);
		qsTransform2.copyToTransform(transform2);

		// setMulMulInv 
		{
			hkTransform mulmulinvT;
			hkQsTransform mulmulinvQs;

			mulmulinvT.setMulMulInverse(transform1, transform2);
			mulmulinvQs.setMulMulInverse(qsTransform1, qsTransform2);

			// T2QS ( QS2T(qs1) * Inv(QS2T (qs2)) ) = qs1 * Inv(qs2)
			{
				hkQsTransform test;
				test.setFromTransform(mulmulinvT);
				HK_TEST(test.isApproximatelyEqual(mulmulinvQs));
			}

			// QS2T ( qs1 * Inv(qs2) ) = QS2T (qs1) * Inv(QS2T(qs2))
			// (guaranteed by symmetry)
			{
				hkTransform test;
				mulmulinvQs.copyToTransform(test);
				HK_TEST(test.isApproximatelyEqual(mulmulinvT));
			}

		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		hkTransform transform1;
		buildRandomQsTransformNoScale(qsTransform1);
		qsTransform1.copyToTransform(transform1);

		hkVector4 position;
		rGen.getRandomVector11(position);

		// hkVector4 setTransformedPos / setTransformedInversePost
		{
			hkVector4 transformedQs;
			transformedQs.setTransformedPos(qsTransform1, position);
			hkVector4 transformedInvQs;
			transformedInvQs.setTransformedInversePos(qsTransform1, position);
			hkVector4 transformedT;
			transformedT.setTransformedPos(transform1, position);
			hkVector4 transformedInvT;
			transformedInvT.setTransformedInversePos(transform1, position);

			// TRANS (qs1, P) = TRANS (t1, P)   [ = TRANS (QS2T(qs1) , P) ]
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST(transformedQs.allEqual<3>(transformedT,eps));

			// TRANSINV (qs1, P) = TRANSINV (t1, P)   [ = TRANSINV (QS2T(qs1), P) ]
			HK_TEST(transformedInvQs.allEqual<3>(transformedInvT,eps));

		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform qsTransform1;
		buildRandomQsTransformNoScale(qsTransform1);

		hkVector4 position;
		rGen.getRandomVector11(position);

		{
			// TRANSINV (qs1, P) = TRANS (Inv(qs1), P)  -> WORKS ONLY WHEN THERE IS NO SCALE (LIKE HERE)
			
			hkVector4 mulInv_qs;
			mulInv_qs.setTransformedInversePos(qsTransform1, position);

			hkQsTransform invQs;
			invQs.setInverse(qsTransform1);
			hkVector4 mul_invQs;
			mul_invQs.setTransformedPos(invQs, position);

			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST(mulInv_qs.allEqual<3>(mul_invQs,eps));
		}

	}
}

// TEST THREE : Other properties (scale included)
static void check_three()
{
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform a;
		hkQsTransform b;
		buildRandomQsTransform (a);
		buildRandomQsTransform (b);

		hkQsTransform left;
		{
			hkQsTransform ab;
			ab.setMul(a,b);
			left.setInverse(ab);
			HK_TEST( left.isOk() );
		}

		hkQsTransform right;
		{
			hkQsTransform inv_b;
			inv_b.setInverse(b);
			HK_TEST( inv_b.isOk() );
			hkQsTransform inv_a;
			inv_a.setInverse(a);
			HK_TEST( inv_a.isOk() );
			right.setMul(inv_b, inv_a);
		}

		// Inv (a*b) = Inv (b) * Inv (a)
		{
			HK_TEST(left.isApproximatelyEqual(right));
		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform a;
		hkQsTransform b;
		hkQsTransform c;
		buildRandomQsTransform (a);
		buildRandomQsTransform (b);
		buildRandomQsTransform (c);

		hkQsTransform right;
		{
			hkQsTransform ab;
			ab.setMul(a,b);
			right.setMul(ab,c);
		}

		hkQsTransform left;
		{
			hkQsTransform bc;
			bc.setMul(b,c);
			left.setMul(a,bc);
		}

		// Associative : a*(b*c) = (a*b)*c
		{
			HK_TEST(left.isApproximatelyEqual(right));
		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform a;
		buildRandomQsTransform (a);
		hkVector4 position;
		rGen.getRandomVector11(position);

		// TRANSINV( a, TRANS ( a, P) ) = P
		// Also tests the TRANSINV operation is alias-safe
		{
			hkVector4 left;
			left.setTransformedPos(a, position);
			left.setTransformedInversePos(a, left); 
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST(left.allEqual<3>(position,eps));
		}
	}
	for (int i=0; i < NUM_TEST; i++)
	{
		hkQsTransform a;
		buildRandomQsTransform (a);
		hkVector4 position;
		rGen.getRandomVector11(position);

		// TRANS( a, TRANSINV ( a, P) ) = P
		// Also tests the TRANS operation is alias-safe
		{
			hkVector4 left;
			left.setTransformedInversePos(a, position);
			left.setTransformedPos(a, left);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST(left.allEqual<3>(position,eps));
		}

	}

}

// TEST FOUR : Decomposition
static void check_four()
{
	static HK_ALIGN_REAL(hkReal bufferOne[16]);
	
	for (int i=0; i<NUM_TEST; i++)
	{
		hkQsTransform qstOne;
		buildRandomQsTransform(qstOne);

		qstOne.get4x4ColumnMajor(bufferOne);

		// round trip through buffer
		{
			hkQsTransform test;
			test.set4x4ColumnMajor(bufferOne);
			HK_TEST(test.isApproximatelyEqual(qstOne));
		}

		// round trip through hkMatrix4
		{
			hkMatrix4 testMatrix;
			testMatrix.set(qstOne);
			hkQsTransform test;
			test.set(testMatrix);
			HK_TEST( test.isApproximatelyEqual(qstOne) );
		}
		
	}

}

// Test setTranslation(),setScale(),setRotation(),set(),
// Test getTranslation(),getScale(),getRotation()
static void check_five()
{
	hkQsTransform qst;
	hkQsTransform qst_test;
	for (int i = 0; i < NUM_TEST; ++i)
	{
		qst.setIdentity();
		HK_TEST( qst.isApproximatelyEqual(hkQsTransform::getIdentity()) );

		hkQuadReal transVal = HK_QUADREAL_CONSTANT(hkUnitTest::rand01(),hkUnitTest::rand01(),hkUnitTest::rand01(), 0.0f);
		hkVector4 trans; trans.m_quad = transVal;
		qst.setTranslation(trans);
		
		hkQuadReal scaleVal = HK_QUADREAL_CONSTANT(hkUnitTest::rand01(),hkUnitTest::rand01(),hkUnitTest::rand01(), 0.0f);
		hkVector4 scale; scale.m_quad = scaleVal;
		qst.setScale(scale);
		
		hkQuadReal row1Val = HK_QUADREAL_CONSTANT(hkUnitTest::rand01(),hkUnitTest::rand01(),hkUnitTest::rand01(), 0.0f);
		hkVector4 first_row; first_row.m_quad = row1Val;
		hkQuadReal row2Val = HK_QUADREAL_CONSTANT(hkUnitTest::rand01(),hkUnitTest::rand01(),hkUnitTest::rand01(), 0.0f);
		hkVector4 second_row; second_row.m_quad = row2Val;
		hkQuadReal row3Val = HK_QUADREAL_CONSTANT(hkUnitTest::rand01(),hkUnitTest::rand01(),hkUnitTest::rand01(), 0.0f);
		hkVector4 third_row; third_row.m_quad = row3Val;
		hkRotation rot;
		rot.setRows(first_row,second_row,third_row);
		if (!rot.isOrthonormal()) continue;
		rot.renormalize();
		qst.setRotation(rot);

		hkQuaternion qt; qt.set(rot);
		qst_test.set(trans,qt,scale);

		HK_TEST( qst_test.isApproximatelyEqual(qst) );
		hkVector4 trans_test; trans_test = qst_test.getTranslation();
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( trans_test.allEqual<3>(trans,eps) );
		hkVector4 scale_test; scale_test = qst_test.getScale();
		HK_TEST( scale_test.allEqual<3>(scale,eps) );
		hkQuaternion qt_test = qst_test.getRotation();
		HK_TEST( qt_test.getImag().allEqual<3>(qt.getImag(),eps) );
		HK_TEST( hkMath::equal(qt_test.getReal(),qt.getReal()) );
	}
}

// TEST SIX : Blend / Normalize operations
static void check_six()
{
	hkQsTransform qst;
	// Verify setZero()
	{
		qst.setZero();
		hkQuaternion rot = qst.getRotation();
		HK_TEST(rot.getReal() == 0);
		hkVector4 vscale = qst.getScale();
		HK_TEST( vscale(0) == 0 && vscale(1) == 0 && vscale(2) == 0);
		hkVector4 vtrans = qst.getTranslation();
		HK_TEST( vtrans(0) == 0 && vtrans(1) == 0 && vtrans(2) == 0);
	}

	// Verify blending operations and isOk()
	for (int i = 0; i < NUM_TEST; ++i)
	{
		qst.setZero();
		hkQsTransform qst_other;
		buildRandomQsTransform(qst_other);
		hkSimdReal epsa; epsa.setFromFloat(0.1f);
		qst.blendAddMul(qst_other,epsa);
		qst.isOk();

		hkVector4 vtrans_test; vtrans_test = hkVector4::getConstant<HK_QUADREAL_0>();
		vtrans_test.addMul(epsa,qst_other.getTranslation());
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( vtrans_test.allEqual<3>(qst.getTranslation(),eps) );
		hkVector4 vscale_test; vscale_test = hkVector4::getConstant<HK_QUADREAL_0>();
		vscale_test.addMul(epsa,qst_other.getScale());
		HK_TEST( vscale_test.allEqual<3>(qst.getScale(),eps) );

		// Verify blendNormalize()
		hkQsTransform qsidentity(qst_other);
		hkVector4 tran_test; tran_test = qsidentity.getTranslation();

		qsidentity.blendNormalize(hkSimdReal::getConstant<HK_QUADREAL_2>());
		qsidentity.isOk();

		tran_test.mul(hkSimdReal::getConstant(HK_QUADREAL_INV_2));
		HK_TEST( tran_test.allEqual<3>(qsidentity.getTranslation(),eps) );

		// verify fastRenormalize()
		hkQsTransform qstone(qst_other);
		tran_test = qstone.getTranslation();

		qstone.fastRenormalize(hkSimdReal::getConstant<HK_QUADREAL_4>());
		qstone.isOk();

		tran_test.mul(hkSimdReal::getConstant<HK_QUADREAL_INV_4>());
		HK_TEST( tran_test.allEqual<3>(qstone.getTranslation(),eps) );
	}

	// Verify setInterpolate4()
	{
		qst.setZero();
		hkQsTransform qst_one;
		qst_one.setIdentity();
		hkQsTransform qst_two;
		qst_two.setIdentity();
		hkSimdReal weight = hkSimdReal::getConstant<HK_QUADREAL_1>();
		qst.setInterpolate4(qst_one,qst_two,weight);
		qst.isOk();
		HK_TEST(qst.isApproximatelyEqual(hkQsTransform::getIdentity()));

		buildRandomQsTransform(qst_one);
		buildRandomQsTransform(qst_two);
		weight = hkSimdReal::getConstant<HK_QUADREAL_2>();
		qst.setZero();
		qst.setInterpolate4(qst_one,qst_two,weight);
		hkVector4 transtest; transtest.setZero();
		transtest.setInterpolate(qst_one.getTranslation(),qst_two.getTranslation(),weight);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( transtest.allEqual<3>(qst.getTranslation(),eps) );

		hkVector4 scaletest; scaletest.setZero();
		scaletest.setInterpolate(qst_one.getScale(),qst_two.getScale(),weight);
		HK_TEST( scaletest.allEqual<3>(qst.getScale(),eps) );
	}

	// verify fastRenormalizeBatch()
	{
		hkQsTransform qst4[4];
		hkReal weight[4];
		for (int i = 0; i < 4; i++)
		{
			qst4[i].setIdentity();
			weight[i] = 4.0f;
		}
		hkQsTransform::fastRenormalizeBatch(qst4,weight,4);

		hkQsTransform test;
		test.setIdentity();
		hkVector4 vtrans = test.getTranslation();
		hkSimdReal q; q.setFromFloat(0.25f);
		vtrans.mul(q);

		hkSimdReal eps; eps.setFromFloat(1e-3f);
		for (int i = 0; i < 4; i++)
		{
			HK_TEST( qst4[i].getTranslation().allEqual<3>(vtrans,eps) );
		}
	}
}

int qstransform_main()
{
	check_zero();  // Approxequal
	check_one();   // Inverses
	check_two();   // Equivalence with hkTransfrom (no scale)
	check_three(); // Other properties (scale included)
	check_four();  // Decomposition
	check_five();  // Get set operations
	check_six();   // Blend / Normalize operations
	return 0;
}

//void ___1() { }
#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(qstransform_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/qstransform.cpp"     );

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
