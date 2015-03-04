/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

static void rotation_test()
{
	// Testing set().
	{
		hkReal angle = 0.7f;
		hkVector4 axis;
		axis.set(1,2,3);
		axis.normalize<3>();

		hkQuaternion q0;
		q0.setAxisAngle(axis,angle);	

		hkRotation r;
		r.set(q0);
		
		hkRotation rtest;
		rtest.setAxisAngle(axis,angle);

		HK_TEST( rtest.isApproximatelyEqual(r) );
		HK_TEST(r.isOrthonormal() );
	}

	// Testing setAxisAngle().
	{
		hkReal angle = 0.7f;
		hkVector4 axis;
		axis.set(1,2,3);
		axis.normalize<3>();

		hkRotation r;
		r.setAxisAngle(axis,angle);
		
		hkQuaternion q0;
		q0.setAxisAngle(axis,angle);	
		hkRotation rtest;
		rtest.set(q0);
		
		HK_TEST( rtest.isApproximatelyEqual(r) );
		HK_TEST(r.isOrthonormal() );
	}

	// Testing isOrthonormal().
	{
		hkVector4 axis;
        axis.set(1,2,3);
        axis.normalize<3>();
        hkReal angle = 0.7f;
        hkRotation r;
        r.setAxisAngle(axis,angle);
        hkReal epsilon = 1e-3f ;
        hkReal R = r.getColumn(0).dot<3>(r.getColumn(0)).getReal();
		if (hkMath::fabs( R - 1.0f) > epsilon)
		{
	        HK_TEST(!r.isOrthonormal() );
		}
	    
		R = r.getColumn(1).dot<3>(r.getColumn(1)).getReal();
        if (hkMath::fabs( R - 1.0f) > epsilon)
		{
            HK_TEST(!r.isOrthonormal() );
		}
        
		R = r.getColumn(2).dot<3>(r.getColumn(2)).getReal();
        if (hkMath::fabs( R - 1.0f) > epsilon)
        {
			HK_TEST(!r.isOrthonormal() );
		}
	}

	// Testing renormalize().
	{
		hkRotation r;
		
		hkVector4 c0;
		c0.set(2.25f, 2.25f, 2.25f); 
		hkVector4 c1;
		c1.set(2.625f, 2.625f, 2.625f); 
		hkVector4 c2;
		c2.set(2.85f, 2.85f, 2.85f);
		r.setCols(c0, c1, c2);
		
		hkRotation r1 = r;
		
		r.renormalize();
				
		hkQuaternion q;
		q.setAndNormalize(r1);
		r1.set(q);

		HK_TEST ( r.isApproximatelyEqual(r1) );
	}

	// Testing setMul().
	{
		hkRotation the_result;
		hkVector4 c0; c0.set(2.25f, 2.25f, 2.25f); 
		hkVector4 c1; c1.set(2.625f, 2.625f, 2.625f); 
		hkVector4 c2; c2.set(2.85f, 2.85f, 2.85f);
		the_result.setCols(c0, c1, c2);
		
		hkRotation the_source;
		hkVector4 cc0;
		cc0.set(1.5f, 1.5f, 1.5f);
		hkVector4 cc1; cc1.set(1.75f, 1.75f, 1.75f);
		hkVector4 cc2; cc2.set(1.9f, 1.9f, 1.9f);
		the_source.setCols(cc0, cc1, cc2);
	
		hkRotation test_matrix;
		hkSimdReal o; o.setFromFloat(1.5f);
		test_matrix.setMul( o, the_source );
	
		HK_TEST( test_matrix.isApproximatelyEqual(the_result, 0.0000001f));
	}

	// Testing transpose()& _setTranspose().
	{
		hkRotation trans_mat;
		hkRotation trans_mat1;
		
		hkVector4 first_row;
		first_row.set(1.0, 2.0, 3.0);
		hkVector4 second_row;
		second_row.set(4.0, 5.0, 6.0);
		hkVector4 third_row;  third_row.set(7.0, 8.0, 9.0);
		trans_mat.setRows(first_row, second_row, third_row);

		trans_mat1 = trans_mat;
			
		trans_mat.transpose();
		trans_mat.transpose();
				
		HK_TEST( trans_mat1.isApproximatelyEqual(trans_mat) );

		hkRotation trans_mat2 = trans_mat;
		trans_mat1._setTranspose(trans_mat2);
		trans_mat2.transpose();
		HK_TEST( trans_mat1.isApproximatelyEqual(trans_mat2) );
	}

	// Testing addMul(),add() and sub() .
	{
		hkRotation rand_mat;
		hkVector4 first_row;  first_row.set(1.0, 2.0, 3.0);
		hkVector4 second_row; second_row.set(4.0, 5.0, 6.0);
		hkVector4 third_row;  third_row.set(7.0, 8.0, 9.0);
		rand_mat.setRows(first_row, second_row, third_row);

		hkRotation rand_mat1;
		hkVector4 first_row1;  first_row1.set(1.0, 1.0, 1.0);
		hkVector4 second_row1; second_row1.set(1.0, 1.0, 1.0);
		hkVector4 third_row1;  third_row1.set(1.0, 1.0, 1.0);
		rand_mat1.setRows(first_row1, second_row1, third_row1);

		hkRotation rand_mat2;
		hkVector4 first_row2;  first_row2.set(2.0, 3.0, 4.0);
		hkVector4 second_row2; second_row2.set(5.0, 6.0, 7.0);
		hkVector4 third_row2;  third_row2.set(8.0, 9.0, 10.0);
		rand_mat2.setRows(first_row2, second_row2, third_row2);
	
		hkRotation rand_mat3;
		hkVector4 first_row3;  first_row3.set(0.0, 1.0, 2.0);
		hkVector4 second_row3; second_row3.set(3.0, 4.0, 5.0);
		hkVector4 third_row3;  third_row3.set(6.0, 7.0, 8.0);
		rand_mat3.setRows(first_row3, second_row3, third_row3);
		// Verify addMul()
		{
			rand_mat.addMul(hkSimdReal::getConstant<HK_QUADREAL_1>(),rand_mat1);
			HK_TEST(rand_mat.isApproximatelyEqual(rand_mat2));
		}
		// Verify add()
		{
			rand_mat.setRows(first_row, second_row, third_row);
			rand_mat.add(rand_mat1);
			HK_TEST(rand_mat.isApproximatelyEqual(rand_mat2));
		}
		// Verify sub()
		{
			rand_mat.setRows(first_row, second_row, third_row);
			rand_mat.sub(rand_mat1);
			HK_TEST(rand_mat.isApproximatelyEqual(rand_mat3));
		}
  }

	// Verification of setIdentity() and getIdentity().
	{
		hkMatrix3 the_identity;
		the_identity = hkMatrix3::getIdentity();
		
		hkRotation should_be_identity;
		should_be_identity.setIdentity();

		HK_TEST(should_be_identity.isApproximatelyEqual(the_identity));
	}
	
	// Testing setZero().
	{
		hkRotation the_zero;
		the_zero.setZero();
		for( int i = 0; i < 3; i++)
		{
			for( int j = 0; j < 3; j++)
			{
				HK_TEST( the_zero(i,j) == 0 );
			}
		}
	}
	
	// Verification of getCols() and getColumn()const , getColumn().
	{
		hkRotation the_result;
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

		// Verification of getCols()
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
	
	// Testing  of setDiagonal() and operator(), operator() const.
	{
		hkRotation mat1;
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
	
	// Testing misc operations
	{
		hkRotation mat;
		hkVector4 first_row_init;  
		hkVector4 second_row_init; 
		hkVector4 third_row_init;  
		first_row_init.set(1.0, 2.0, 3.0);
		second_row_init.set(4.0, 5.0, 6.0);
		third_row_init.set(7.0, 8.0, 9.0);
		mat.setRows(first_row_init, second_row_init, third_row_init);
	
		// Verification of add(), _add() and _setMul and addMul()
		{
			hkRotation mat2 = mat;
			mat2.add(mat);

			hkRotation test_mat;
			test_mat._setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(),mat);
			
			HK_TEST( mat2.isApproximatelyEqual(test_mat) );

			hkRotation out_mat = mat2;
			out_mat._add(test_mat);
			
			hkMatrix3 test;
			test.setZero();
			test.addMul(hkSimdReal::getConstant<HK_QUADREAL_2>(),test_mat);
			HK_TEST( test.isApproximatelyEqual(out_mat) );
		}
		
		// Verification of sub() and _sub().
		{
			hkRotation mat2 = mat;
			mat2.sub(mat);

			hkRotation test_mat;
			test_mat.setZero();
			HK_TEST( mat2.isApproximatelyEqual(test_mat) );
			
			test_mat._setMul(hkSimdReal::getConstant<HK_QUADREAL_2>(),mat);
			test_mat._sub(mat);
			HK_TEST( test_mat.isApproximatelyEqual(mat) );
		}

		// Verification of _setMul()
		{
			hkRotation mat2 = mat;
			hkRotation test_mat;
			hkRotation test_out = mat;
			test_mat._setMul(mat,mat2);
			test_out.mul(mat2);
			HK_TEST( test_out.isApproximatelyEqual(test_mat) );
		}
		

		// Testing changeBasis() .
		{
			hkRotation r1;
			hkVector4 first_row;  
			hkVector4 second_row; 
			hkVector4 third_row;  

			first_row.set(1.0, 2.0, 3.0);
			first_row.normalize<3>();

			second_row.set(4.0, 5.0, 6.0);
			second_row.normalize<3>();

			third_row.set(7.0, 8.0, 9.0);
			third_row.normalize<3>();
			
			r1.setRows(first_row, second_row, third_row);
			
			hkMatrix3 r2_temp = r1;
		
			hkRotation r2;
			r2.setCols(second_row,third_row,first_row);

			hkRotation rot_temp ;
			rot_temp.setMulInverse(r1,r2);

			r1.setMul(r2,rot_temp); 
			
			r2_temp.changeBasis(r2);

			HK_TEST( r2_temp.isOk() );
			HK_TEST( r2_temp.isApproximatelyEqual(r1) );
		
		}
	}
}

int rotation_main()
{
	rotation_test();
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(rotation_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Rotation.cpp"     );

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
