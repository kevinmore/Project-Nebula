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
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

// static void do_test(bool cond, const char* x)
// {
// 	if ( !cond )
// 	{
// 		HK_REPORT("Test " << x << " failed");
//		HK_BREAKPOINT(0);
// 	}
// }
// 
// #undef HK_TEST
// #define HK_TEST(x)	do_test(x, #x)

static void mat3_mul_inverse_equals()
{
	//make up a space
	hkMatrix3 the_identity;
	the_identity.setIdentity();

	hkVector4 rand_first_row; rand_first_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
	rand_first_row.normalize<3>();
	hkVector4 rand_second_row;
	hkVector4Util::calculatePerpendicularVector( rand_first_row, rand_second_row);
	rand_second_row.normalize<3>();

	hkVector4 rand_third_row;
	rand_third_row.setCross( rand_first_row, rand_second_row );

	hkMatrix3 rand_mat3;
	rand_mat3.setRows( rand_first_row, rand_second_row, rand_third_row );

	hkMatrix3 rand_inverse( rand_mat3);
	rand_inverse.invert( 0.001f );

	hkMatrix3 should_be_identity;
	should_be_identity.setMul( rand_mat3, rand_inverse );
	HK_TEST( should_be_identity.isApproximatelyEqual( the_identity ) );

	hkRotation rand_rot;
	hkVector4 rows[3];
	rand_mat3.getRows( rows[0], rows[1], rows[2] );
	rand_rot.setRows( rows[0], rows[1], rows[2] );
	should_be_identity.setMulInverse( rand_mat3, rand_rot );
	HK_TEST( should_be_identity.isApproximatelyEqual( the_identity ) );

	rand_mat3.mul( rand_inverse );
	HK_TEST( rand_mat3.isApproximatelyEqual( the_identity ) );
}

static void mat3_mul()
{
	// make up a result matrix
	hkMatrix3 the_result;
	hkVector4 c0; c0.set(2.25f, 2.25f, 2.25f); 
	hkVector4 c1; c1.set(2.625f, 2.625f, 2.625f); 
	hkVector4 c2; c2.set(2.85f, 2.85f, 2.85f);
	the_result.setCols(c0, c1, c2);

	// make up a test matrix for ::mul
	hkMatrix3 test_matrix;
	hkVector4 cc0; cc0.set(1.5f, 1.5f, 1.5f);
	hkVector4 cc1; cc1.set(1.75f, 1.75f, 1.75f);
	hkVector4 cc2; cc2.set(1.9f, 1.9f, 1.9f);
	test_matrix.setCols(cc0, cc1, cc2);

	hkSimdReal o; o.setFromFloat(1.5f);
	test_matrix.mul(o);
	HK_TEST( test_matrix.isApproximatelyEqual(the_result, 0.0000001f));
}

static void mat3_set_mul()
{
	// make up a result matrix
	hkMatrix3 the_result;
	hkVector4 c0; c0.set(2.25f, 2.25f, 2.25f); 
	hkVector4 c1; c1.set(2.625f, 2.625f, 2.625f); 
	hkVector4 c2; c2.set(2.85f, 2.85f, 2.85f);
	the_result.setCols(c0, c1, c2);
	
	// make up a test matrix for ::setmul
	hkMatrix3 the_source;
	hkVector4 cc0; cc0.set(1.5f, 1.5f, 1.5f);
	hkVector4 cc1; cc1.set(1.75f, 1.75f, 1.75f);
	hkVector4 cc2; cc2.set(1.9f, 1.9f, 1.9f);
	the_source.setCols(cc0, cc1, cc2);
	
	hkMatrix3 test_matrix;
	hkSimdReal o; o.setFromFloat(1.5f);
	test_matrix.setMul( o, the_source );
	
	HK_TEST( test_matrix.isApproximatelyEqual(the_result, 0.0000001f));
}

static void mat3_misc()
{
	// Verification of setIdentity() and getIdentity() 
	{
		hkMatrix3 the_identity;
		the_identity = hkMatrix3::getIdentity();
		
		hkMatrix3 should_be_identity;
		should_be_identity.setIdentity();

		HK_TEST(should_be_identity.isApproximatelyEqual(the_identity));
	}
	// Verification of setZero()
	{
		hkMatrix3 the_zero;
		the_zero.setZero();
		for( int i = 0; i < 3; i++)
		{
			for( int j = 0; j < 3; j++)
			{
				HK_TEST( the_zero(i,j) == 0 );
			}
		}
	}
	// Verification of getCols() and getColumn()const , getColumn()
	{
		hkMatrix3 the_result;
		hkVector4 c0; 
		hkVector4 c1; 
		hkVector4 c2; 
		c0.set(2.25f, 2.25f, 2.25f); 
		c1.set(2.625f, 2.625f, 2.625f); 
		c2.set(2.85f, 2.85f, 2.85f);
		the_result.setCols(c0, c1, c2);
		
		hkVector4 cout0; 
		hkVector4 cout1; 
		hkVector4 cout2; 
		//Verification of getCols()
		the_result.getCols(cout0, cout1, cout2);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( c0.allEqual<3>(cout0,eps) );
		HK_TEST( c1.allEqual<3>(cout1,eps) );
		HK_TEST( c2.allEqual<3>(cout2,eps) );
		
		// Verification of getColumn() const		
		hkVector4 test = the_result.getColumn(0);
		HK_TEST( cout0.allEqual<3>(test,eps) );
		test = the_result.getColumn(1);
		HK_TEST( cout1.allEqual<3>(test,eps) );
		test = the_result.getColumn(2);
		HK_TEST( cout2.allEqual<3>(test,eps) );

		// Verification of getColumn()		
		the_result.getColumn(0) = cout2;
		test = the_result.getColumn(0);
		HK_TEST( test.allEqual<3>(cout2,eps) );
	}
	// Verification of setDiagonal() and operator(), operator() const
	{
		hkMatrix3 mat1;
		mat1.setIdentity();
		hkMatrix3Util::_setDiagonal(0,5,10, mat1);
		HK_TEST( mat1(0,0) == 0 );
		HK_TEST( mat1(1,1) == 5 );
		HK_TEST( mat1(2,2) == 10 );
		
		mat1(0,0) = 2;
		mat1(1,0) = 4;
		mat1(2,0) = 8;
		hkVector4 out = mat1.getColumn(0);
		HK_TEST( out(0) == 2);
		HK_TEST( out(1) == 4);
		HK_TEST( out(2) == 8);
	}
	// Verification of setCrossSkewSymmetric()
	{
		hkMatrix3 mat;
		hkVector4 c0; 
		hkVector4 c1; 
		hkVector4 c2; 
		
		c0.set(0, 3, -2); 
		c1.set(-3, 0, 1); 
		c2.set(2,-1, 0);
		mat.setCols(c0, c1, c2);
		
		hkQuadReal cVal = HK_QUADREAL_CONSTANT(1,2,3,0);
		hkVector4 c; c.m_quad=cVal;
		hkMatrix3 the_result;
		the_result.setIdentity();
		the_result.setCrossSkewSymmetric(c);
		HK_TEST( the_result.isOk() );
		HK_TEST( the_result.isApproximatelyEqual(mat) );
	}
	// Verification of _invertSymmetric() and invertSymmetric()
	{
		hkMatrix3 mat;
		
		hkVector4 c0; 
		hkVector4 c1; 
		hkVector4 c2; 
		
		c0.set(1.0f, 1.0f, 1.0f); 
		c1.set(0.0f, 1.0f, 1.0f); 
		c2.set(1.0f, 0.0f, 1.0f);
		mat.setRows(c0, c1, c2);
		mat._invertSymmetric();
		HK_TEST( mat.isOk() );
		hkMatrix3 mat1;
		mat1.setRows(c0, c1, c2);

		mat1.invertSymmetric();
		HK_TEST( mat1.isOk() );
		HK_TEST( mat.isApproximatelyEqual(mat1) );
		
		hkMatrix3 mat2;
		mat2.setRows(c0, c1, c2);
		// Invert operation first
		if(mat2.invert(0.5) == HK_SUCCESS)
		{
			// Symmetric operation
			HK_TEST( mat2.isOk() );
			hkAlgorithm::swap( mat2(0,1), mat2(1,0) );
			hkAlgorithm::swap( mat2(0,2), mat2(2,0) );
			hkAlgorithm::swap( mat2(1,2), mat2(2,1) );
			
			HK_TEST( mat.isApproximatelyEqual(mat2) );
		}
	}
}

// Testing transpose() and setTranspose(), _setTranspose() functionality.
static void mat3_Transpose()
{
	hkMatrix3 trans_mat;
	hkMatrix3 trans_mat1;
	hkMatrix3 trans_mat2;

	hkVector4 first_row;  
	hkVector4 second_row; 
	hkVector4 third_row;  
	first_row.set(1.0, 2.0, 3.0);
	second_row.set(4.0, 5.0, 6.0);
	third_row.set(7.0, 8.0, 9.0);
	trans_mat.setRows(first_row, second_row, third_row);
	
	trans_mat2 = trans_mat;
	trans_mat.transpose();
	trans_mat.transpose();
	HK_TEST( trans_mat2.isApproximatelyEqual(trans_mat) );
	
	trans_mat1 = trans_mat;
	trans_mat1.transpose();

	trans_mat2.setTranspose(trans_mat);
	HK_TEST( trans_mat1.isApproximatelyEqual(trans_mat2) );
	
	trans_mat1._setTranspose(trans_mat2);
	HK_TEST( trans_mat1.isApproximatelyEqual(trans_mat) );

}

// Testing _setMul(),add(), _add()and sub() functionality.
static void mat3_add_mul()
{
	hkMatrix3 mat;
	hkVector4 first_row;  
	hkVector4 second_row; 
	hkVector4 third_row;  
	first_row.set(1.0f, 2.0f, 3.0f);
	second_row.set(4.0f, 5.0f, 6.0f);
	third_row.set(7.0f, 8.0f, 9.0f);
	mat.setRows(first_row, second_row, third_row);
	
	// Verification of add(), _add() and _setMul and addMul()
	{
		hkMatrix3 mat2 = mat;
		mat2.add(mat);

		hkMatrix3 test_mat;
		test_mat._setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(),mat);
		
		HK_TEST( mat2.isApproximatelyEqual(test_mat) );

		hkMatrix3 out_mat = mat2;
		out_mat._add(test_mat);
		
		hkMatrix3 test;
		test.setZero();
		test.addMul(hkSimdReal::getConstant<HK_QUADREAL_2>(),test_mat);
		HK_TEST( test.isApproximatelyEqual(out_mat) );
	}
	
	// Verification of sub() and _sub()
	{
		hkMatrix3 mat2 = mat;
		mat2.sub(mat);

		hkMatrix3 test_mat;
		test_mat.setZero();
		HK_TEST( mat2.isApproximatelyEqual(test_mat) );
		
		test_mat._setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(),mat);
		test_mat._sub(mat);
		HK_TEST( test_mat.isApproximatelyEqual(mat) );
	}

	// Verification of _setMul()
	{
		hkMatrix3 mat2 = mat;
		hkMatrix3 test_mat;
		hkMatrix3 test_out = mat;
		test_mat._setMul(mat,mat2);
		test_out.mul(mat2);
		HK_TEST( test_out.isApproximatelyEqual(test_mat) );
	}
}

// Testing changeBasis() functionality.
static void mat3_change_basis()
{
	hkMatrix3 rand_mat;
	hkVector4 first_row;  
	hkVector4 second_row; 
	hkVector4 third_row;  
	first_row.set(1.0, 3.0, 3.0);
	second_row.set(4.0, 3.0, 3.0);
	third_row.set(1.0, 3.0, 4.0);
	rand_mat.setRows(first_row, second_row, third_row);
	
	hkMatrix3 org_mat = rand_mat;
	hkRotation rot ;
	rot.setRows(first_row, second_row, third_row);

	hkRotation rot_temp ;
	rot_temp.setMulInverse(org_mat,rot);
	org_mat.setMul(rand_mat,rot_temp);
	HK_TEST( org_mat.isOk() );
	
	rand_mat.changeBasis(rot);

	HK_TEST( rand_mat.isOk() );
	HK_TEST( rand_mat.isApproximatelyEqual(org_mat) );
}


// Compute x = M v
static hkVector4 _mulMatrixVector( const hkMatrix3& M, const hkVector4& v )
{
	hkReal x[3]; x[0] = x[1] = x[2] = 0.0f;

	for (int i=0; i<3; ++i)
	{
		for (int j=0; j<3; ++j)
		{
			x[i] += M(i,j) * v(j);		
		}
	}

	hkVector4 X;
	X.set(x[0], x[1], x[2]);
	return X;
}


static void _checkEigenvalueEquation( const hkVector4& eigenVec, hkReal eigenVal, const hkMatrix3& M, hkReal fracTol, hkReal absTol = 2e-4f )
{
	hkVector4 lhs = _mulMatrixVector(M, eigenVec);
	
	hkVector4 rhs = eigenVec;
	hkSimdReal se; se.setFromFloat(eigenVal);
	rhs.mul(se);

	hkReal len = rhs.length<3>().getReal();
	rhs.sub(lhs);

	if (len<1.0e-5f) return;
	
	hkReal absErr = rhs.length<3>().getReal();
	hkReal fracErr = absErr / len;
	
	HK_TEST( fracErr < fracTol || absErr < absTol);
}

// Testing diagonalizeSymmetric() functionality
static void mat3_diagonalize()
{
	int seed = 'm'+'a'+'t'+'3'+'d'+'i'+'a'+'g';
	hkPseudoRandomGenerator rand(seed);

	for (int n=1; n<100; ++n)
	{
		hkReal s = (hkReal)n;

		// make random symmetric matrix
		hkVector4 c0; rand.getRandomVector01( c0 ); 
		hkVector4 c1; rand.getRandomVector01( c1 );
		hkVector4 c2; rand.getRandomVector01( c2 );
		
		hkMatrix3 A;
		A.setCols(c0, c1, c2);
		hkSimdReal srs; srs.setFromFloat(s);
		A.mul(srs);

		hkMatrix3 M(A); M.transpose();
		M.add(A);

		HK_TEST( M.isOk() );
		HK_TEST( M.isSymmetric() );

		// diagonalize
		hkRotation eigenVec;
		hkVector4 eigenVal;

		int niter = 64;
		hkReal epsilon = 1.0e-15f;

		M.diagonalizeSymmetric(eigenVec, eigenVal, niter, epsilon);

		hkVector4 e0, e1, e2; 
		eigenVec.getCols(e0, e1, e2);

		const hkReal allowFracError = 1.0e-3f; // check eigenvalue equation is satisfied to 0.1%
		_checkEigenvalueEquation(e0, eigenVal(0), M, allowFracError);
		_checkEigenvalueEquation(e1, eigenVal(1), M, allowFracError);
		_checkEigenvalueEquation(e2, eigenVal(2), M, allowFracError);
	}
}

int matrix3_main()
{
	mat3_mul_inverse_equals();
	mat3_mul();
	mat3_set_mul();
	mat3_misc();
	mat3_add_mul();
	mat3_Transpose();
	mat3_change_basis();
	mat3_diagonalize();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(matrix3_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/matrix3.cpp"     );

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
