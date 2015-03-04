/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>

//sort of checks inverse too
static void mat4_mul_inverse_equals()
{
	//make up a space
	hkMatrix4 the_identity;
	the_identity.setIdentity();

	hkVector4 rand_first_row; rand_first_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
	rand_first_row.normalize<3>();
	hkVector4 rand_second_row;
	hkVector4Util::calculatePerpendicularVector( rand_first_row, rand_second_row);
	rand_second_row.normalize<3>();

	hkVector4 rand_third_row;
	rand_third_row.setCross( rand_first_row, rand_second_row );

	hkVector4 rand_forth_row; rand_forth_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01(), 1.0f );

	hkMatrix4 rand_mat4;
	rand_mat4.setRows( rand_first_row, rand_second_row, rand_third_row, rand_forth_row );

	hkMatrix4 rand_inverse( rand_mat4 );
	hkMatrix4Util::invert( rand_inverse, hkSimdReal::fromFloat(hkReal(0.001f)) );

	hkSimdReal eps = hkSimdReal::fromFloat(1e-3f);
	hkMatrix4 should_be_identity;
	should_be_identity.setMul( rand_mat4, rand_inverse );
	HK_TEST( should_be_identity.isApproximatelyEqual( the_identity, eps ) );

	hkMatrix4Util::_setMulInverse( rand_mat4, rand_mat4, should_be_identity );
	HK_TEST( should_be_identity.isApproximatelyEqual( the_identity, eps ) );
}

static void mat4_mul()
{
	// make up a result matrix
	hkMatrix4 the_result;
	hkVector4 c0; c0.set(2.25f, 2.25f, 2.25f); 
	hkVector4 c1; c1.set(2.625f, 2.625f, 2.625f); 
	hkVector4 c2; c2.set(2.85f, 2.85f, 2.85f);
	hkVector4 c3; c3.set(1.5f, 3, 4.5f, 6);
	the_result.setCols(c0, c1, c2, c3);

	// make up a test matrix for ::mul
	hkMatrix4 test_matrix;
	hkVector4 cc0; cc0.set(1.5f, 1.5f, 1.5f);
	hkVector4 cc1; cc1.set(1.75f, 1.75f, 1.75f);
	hkVector4 cc2; cc2.set(1.9f, 1.9f, 1.9f);
	hkVector4 cc3; cc3.set(1,2,3,4);
	test_matrix.setCols(cc0, cc1, cc2, cc3);

	hkSimdReal o; o.setFromFloat(1.5f);
	test_matrix.mul(o);
	HK_TEST( test_matrix.isApproximatelyEqual(the_result, hkSimdReal::fromFloat(0.0000001f)));
}

static void mat4_set_mul()
{
	// make up a result matrix
	hkMatrix4 the_result;
	hkVector4 c0; c0.set(2.25f, 2.25f, 2.25f); 
	hkVector4 c1; c1.set(2.625f, 2.625f, 2.625f); 
	hkVector4 c2; c2.set(2.85f, 2.85f, 2.85f);
	hkVector4 c3; c3.set(1.5f, 3, 4.5f, 6);
	the_result.setCols(c0, c1, c2, c3);

	// make up a test matrix for ::setmul
	hkMatrix4 the_source;
	hkVector4 cc0; cc0.set(1.5f, 1.5f, 1.5f);
	hkVector4 cc1; cc1.set(1.75f, 1.75f, 1.75f);
	hkVector4 cc2; cc2.set(1.9f, 1.9f, 1.9f);
	hkVector4 cc3; cc3.set(1,2,3,4);
	the_source.setCols(cc0, cc1, cc2, cc3);

	hkMatrix4 test_matrix;
	hkSimdReal o; o.setFromFloat(1.5f);
	test_matrix.setMul( o, the_source );

	HK_TEST( test_matrix.isApproximatelyEqual(the_result, hkSimdReal::fromFloat(0.0000001f)));
}

static void mat4_bugs_fixed()
{
	hkMatrix4 randomMatrix;
	hkVector4 c0; c0.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01()); 
	hkVector4 c1; c1.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01()); 
	hkVector4 c2; c2.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01()); 
	hkVector4 c3; c3.set(hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01()); 
	randomMatrix.setCols(c0, c1, c2, c3);

	hkSimdReal zero; zero.setZero();

	// HVK-3599 : setTranspose didn't really transpose
	{
		hkMatrix4 copy = randomMatrix;
		copy.transpose();
		copy.transpose();
		HK_TEST(copy.isApproximatelyEqual(randomMatrix,zero));

		copy.setTranspose(randomMatrix);
		copy.transpose();
		HK_TEST(copy.isApproximatelyEqual(randomMatrix,zero));

		copy = randomMatrix;
		copy.transpose();
		hkMatrix4 copy2; copy2.setTranspose(copy);
		HK_TEST(copy2.isApproximatelyEqual(randomMatrix,zero));
	}	
}

static void mat4_vector_ops()
{
	// vector multiplication - arbitrary values (not a transformation)
	{
		hkMatrix4 arbitraryMatrix;
		hkVector4 r0; r0.set( 1, 2, 3, 4);
		hkVector4 r1; r1.set( 5, 6, 7, 8);
		hkVector4 r2; r2.set( 9,10,11,12);
		hkVector4 r3; r3.set(13,14,15,16);
		arbitraryMatrix.setRows(r0,r1,r2,r3);

		HK_TEST(!arbitraryMatrix.isAffineTransformation());

		hkVector4 arbitraryVector;
		arbitraryVector.set(10,20,30,40);

		hkVector4 result;
		arbitraryMatrix.multiplyVector(arbitraryVector, result);

		hkQuadReal testVal = HK_QUADREAL_CONSTANT(300,700,1100,1500);
		hkVector4 test_vec; test_vec.m_quad = testVal;
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(result.allEqual<4>(test_vec,eps));
	}

	// transformation of positions and directions
	{
		hkMatrix4 aTransform;
		hkVector4 r0; r0.set(-2, 0, 0, 1);
		hkVector4 r1; r1.set( 0, 0, 1, 2);
		hkVector4 r2; r2.set( 0, 3, 0, 3);
		hkVector4 r3; r3.set( 0, 0, 0, 1);
		aTransform.setRows(r0,r1,r2,r3);

		HK_TEST(aTransform.isAffineTransformation());

		hkVector4 aVector;
		aVector.set(10,20,30,40); // the 40 should be ignored

		hkVector4 result;
		aTransform.transformPosition(aVector, result);

		hkQuadReal testVal = HK_QUADREAL_CONSTANT(-19,32,63,1);
		hkVector4 test_vec; test_vec.m_quad = testVal;
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST(result.allEqual<4>(test_vec,eps));

		aTransform.transformDirection(aVector, result);

		hkQuadReal testVal2 = HK_QUADREAL_CONSTANT(-20,30,60,0);
		test_vec.m_quad = testVal2;
		HK_TEST(result.allEqual<4>(test_vec,eps));
	}
}
static void mat4_matrix_getset_ops()
{
	// Verification of getIdentity() and setIdentity() functions
	{
		hkMatrix4 the_identity;
		the_identity.setIdentity();
		
		for( int i = 0; i < 4; i++)
		{
			for( int j = 0; j < 4; j++)
			{
				HK_TEST( the_identity(i,i) == 1 );
				if( i != j)
				{
					HK_TEST( the_identity(i,j) == 0 );
				}
			}
		}
		
		HK_TEST( the_identity.isApproximatelyIdentity(hkSimdReal::fromFloat(1e-3f)) );
	}
	// Verification of setZero()
	{
		hkMatrix4 the_zero;
		the_zero.setZero();
		for( int i = 0; i < 4; i++)
		{
			for( int j = 0; j < 4; j++)
			{
				HK_TEST( the_zero(i,j) == 0 );
			}
		}
	}
	// Verification of set4x4ColumnMajor(),get4x4ColumnMajor()
	{
		hkMatrix4 mat;
		HK_ALIGN_REAL(hkReal arr_in[16]);
		HK_ALIGN_REAL(hkReal arr_out[16]);
		for(int i = 0; i < 16; i++)
		{
			arr_in[i] = static_cast<hkReal>(i);
			arr_out[i] = 0;
		}
		mat.set4x4ColumnMajor(arr_in);
		mat.get4x4ColumnMajor(arr_out);
		
		for(int i = 0; i < 16; i++)
		{
			HK_TEST( hkMath::equal(arr_in[i],arr_out[i]) );			
		}
		
		// Verification of set4x4RowMajor() and get4x4RowMajor()
		{
			hkMatrix4 matrow;
			matrow.set4x4RowMajor(arr_in);
			HK_TEST( matrow.isOk() );
			HK_ALIGN_REAL(hkReal arr[16]);
			matrow.get4x4RowMajor(arr);
			for(int i = 0; i < 16; i++)
			{
				HK_TEST( hkMath::equal(arr_in[i],arr[i]) );			
			}
		}

		// Verification of getRows(),getRow() 
		{
			hkVector4 r0;
			hkVector4 r1;
			hkVector4 r2;
			hkVector4 r3;
			mat.getRows(r0,r1,r2,r3);
			hkVector4 out;
			mat.getRow(0,out);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( r0.allEqual<4>(out,eps) );
			mat.getRow(1,out);
			HK_TEST( r1.allEqual<4>(out,eps) );
			mat.getRow(2,out);
			HK_TEST( r2.allEqual<4>(out,eps) );
			mat.getRow(3,out);
			HK_TEST( r3.allEqual<4>(out,eps) );
		}
		
		// Verification of getCols(),getColumn() const , getColumn()
		{
			hkVector4 c0;
			hkVector4 c1;
			hkVector4 c2;
			hkVector4 c3;
			mat.getCols(c0,c1,c2,c3);
			hkVector4 out;
			out = mat.getColumn(0);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( c0.allEqual<4>(out,eps) );
			out = mat.getColumn(1);
			HK_TEST( c1.allEqual<4>(out,eps) );
			out = mat.getColumn(2);
			HK_TEST( c2.allEqual<4>(out,eps) );
			out = mat.getColumn(3);
			HK_TEST( c3.allEqual<4>(out,eps) );
			
			out.set(10,20,30,40);
			mat.setColumn<3>(out);
			hkVector4 temp;
			temp = mat.getColumn(3);
			HK_TEST ( temp.allEqual<4>(out,eps) );
		}
		// Verification of setDiagonal() and operator(), operator() const
		{
			hkMatrix4 mat1;
			mat1.setIdentity();
			mat1.setDiagonal(0,5,10,15);
			HK_TEST( mat1(0,0) == 0 );
			HK_TEST( mat1(1,1) == 5 );
			HK_TEST( mat1(2,2) == 10 );
			HK_TEST( mat1(3,3) == 15 );
			mat1(1,0) = 4;
			mat1(2,0) = 8;
			mat1(3,0) = 12;
			hkVector4 out = mat1.getColumn(0);
			HK_TEST( out(0) == 0);
			HK_TEST( out(1) == 4);
			HK_TEST( out(2) == 8);
			HK_TEST( out(3) == 12);
		}
		// Verification of = , resetFourthRow() and set() 
		{
			hkSimdReal eps = hkSimdReal::fromFloat(1e-3f);
			hkMatrix4 mat2 = mat;
			HK_TEST( mat2.isApproximatelyEqual(mat,eps) );
			
			mat2.resetFourthRow();
			
			hkVector4 r;
			mat2.getRow(3,r);
			HK_TEST( r(0) == 0);
			HK_TEST( r(1) == 0);
			HK_TEST( r(2) == 0);
			HK_TEST( r(3) == 1);

			hkMatrix4 mat1;
			hkRotation rot;
			rot.setCols(mat2.getColumn(0),mat2.getColumn(1),mat2.getColumn(2));
			hkTransform t; t.set(rot,mat2.getColumn(3));
			mat1.set(t);
			HK_TEST( mat1.isOk() );
			HK_TEST( mat1.isApproximatelyEqual(mat2,eps) );
		}
	}
}
static void mat4_matrix_ops()
{
	hkVector4 r0; 
	hkVector4 r1; 
	hkVector4 r2; 
	hkVector4 r3; 
	
	r0.set( 1, 2, 3, 4);
	r1.set( 5, 6, 7, 8);
	r2.set( 9,10,11,12);
	r3.set(13,14,15,16);

	hkSimdReal eps = hkSimdReal::fromFloat(1e-3f);

	// Verification of add() 
	{
		hkMatrix4 mat1;
		mat1.setRows(r0,r1,r2,r3);
		hkMatrix4 mat2;	
		mat2.setRows(r0,r1,r2,r3);
		mat1.add(mat2);

		hkMatrix4 test_matrix;
		test_matrix.setMul( hkSimdReal::getConstant<HK_QUADREAL_2>(), mat2 );

		HK_TEST ( test_matrix.isApproximatelyEqual(mat1,eps) );
	}

	// Verifictaion of sub() 
	{
		hkMatrix4 mat1;
		mat1.setRows(r0,r1,r2,r3);
		hkMatrix4 mat2 = mat1;	
		mat1.sub(mat2);
		hkMatrix4 test_matrix;
		test_matrix.setZero();
		HK_TEST ( test_matrix.isApproximatelyEqual(mat1,eps) );
	}
	// Verification of mul()
	{
		hkMatrix4 mat1;
		mat1.setRows(r0,r1,r2,r3);
		hkMatrix4 mat2 = mat1;	
		mat1.mul(mat2);
		
		hkMatrix4 test_matrix;
		test_matrix.setMul(mat2,mat2);
		HK_TEST ( test_matrix.isApproximatelyEqual(mat1,eps) );
	}
}

int matrix4_main()
{
	mat4_mul_inverse_equals();
	mat4_mul();
	mat4_set_mul();
	mat4_bugs_fixed();
	mat4_vector_ops();
	mat4_matrix_getset_ops();
	mat4_matrix_ops();
	return 0;
}

//void ___1() { }
#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(matrix4_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/matrix4.cpp"    );

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
