/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

static void matrix6()
{
	hkMatrix6 matrix_1;
	hkVector4 first_row_init;
	hkVector4 second_row_init;
	hkVector4 third_row_init;

	first_row_init.setAll(1);
	second_row_init.setAll(2);
	third_row_init.setAll(3);
	
	for(int i = 0; i < 2; i++)
	{
		for(int j = 0; j < 2; j++)
		{
			matrix_1.m_m[i][j].setRows(first_row_init, second_row_init, third_row_init);
		}
	}

	// testing setZero() functionality.
	{
		hkMatrix6 matrix_2;
		hkVector4 vec1;
		hkVector4 vec2;
		hkVector4 vec3;
		
		matrix_2.setZero();
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2 ; j++)
			{
				hkMatrix3 mat = matrix_2.m_m[i][j];
				mat.getCols(vec1, vec2, vec3);
				for (int k = 0; k < 3; k++)
				{
					HK_TEST(vec1(k) == 0);
					HK_TEST(vec2(k) == 0);
					HK_TEST(vec3(k) == 0);
				}
			}
		}
	}

	// testing setIdentity() functionality.
	{
		hkMatrix6 matrix_2 ;
		hkMatrix3 mat;
		hkVector4 vec1;
		hkVector4 vec2;
		hkVector4 vec3;

		matrix_2.setIdentity();

		hkVector4 vec4; vec4 = hkVector4::getConstant<HK_QUADREAL_1000>();
		hkVector4 vec5; vec5 = hkVector4::getConstant<HK_QUADREAL_0100>();
		hkVector4 vec6; vec6 = hkVector4::getConstant<HK_QUADREAL_0010>();

		for(int i = 0; i < 2; i++)
		{
			mat = matrix_2.m_m[i][i];
			mat.getCols(vec1,vec2,vec3);
			for (int k = 0; k < 3; k++)
			{
				HK_TEST(vec1(k) == vec4(k));
				HK_TEST(vec2(k) == vec5(k));
				HK_TEST(vec3(k) == vec6(k));
			}
		}
	}

	// testing setTranspose() functionality.
	{
		hkMatrix6 matrix_3,matrix_4;

		matrix_3.setTranspose(matrix_1);
		matrix_4.setTranspose(matrix_3);
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(matrix_4.m_m[i][j].isApproximatelyEqual(matrix_1.m_m[i][j], HK_REAL_EPSILON));
			}
		}
	}

	// Testing setMul() functionality.
	{
		hkMatrix6 matrix_2 = matrix_1;
		hkMatrix6 matrix_3;
		hkMatrix3 mat;
		hkVector4 first_row;
		hkVector4 second_row;
		hkVector4 third_row;

		matrix_3.setMul(matrix_1,matrix_2);
		
		first_row.setAll(12);
		second_row.setAll(24);
		third_row.setAll(36);

		mat.setRows(first_row, second_row, third_row);
		
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(matrix_3.m_m[i][j].isApproximatelyEqual(mat, HK_REAL_EPSILON));
			}
		}
	}

	// Testing add() functionality.
	{
		hkMatrix6 matrix_2 = matrix_1;
		hkMatrix3 mat;
		hkVector4 first_row;
		hkVector4 second_row;
		hkVector4 third_row;

		matrix_2.add(matrix_1);
		
		first_row.setAll(2);
		second_row.setAll(4);
		third_row.setAll(6);
		mat.setRows(first_row, second_row, third_row);

		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(mat.isApproximatelyEqual(matrix_2.m_m[i][j], HK_REAL_EPSILON));
			}
		}
	}

	// Testing sub() functionality.
	{
		hkMatrix6 matrix_2 = matrix_1;
		hkMatrix3 mat;
		hkVector4 first_row;
		hkVector4 second_row;
		hkVector4 third_row;

		matrix_2.sub(matrix_1);
		
		first_row.setAll(0);
		second_row.setAll(0);
		third_row.setAll(0);
		mat.setRows(first_row, second_row, third_row);

		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(mat.isApproximatelyEqual(matrix_2.m_m[i][j], HK_REAL_EPSILON));
			}
		}
	}

	// Testing mul(hkMatrix6) functionality
	{
		hkMatrix6 matrix_2 = matrix_1;
		hkMatrix3 mat;
		hkVector4 first_row;
		hkVector4 second_row;
		hkVector4 third_row;

		matrix_2.mul(matrix_1);
		
		first_row.setAll(12);
		second_row.setAll(24);
		third_row.setAll(36);
		mat.setRows(first_row, second_row, third_row);
		
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(matrix_2.m_m[i][j].isApproximatelyEqual(mat, HK_REAL_EPSILON));
			}
		}
	}

	// Testing mul(scale) functionality.
	{
		hkMatrix6 matrix_2 = matrix_1;
		hkMatrix3 mat;
		hkVector4 first_row;
		hkVector4 second_row;
		hkVector4 third_row;

		matrix_2.mul(hkSimdReal::getConstant<HK_QUADREAL_2>());
		
		first_row.setAll(2);
		second_row.setAll(4);
		third_row.setAll(6);
		mat.setRows(first_row, second_row, third_row);

		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(matrix_2.m_m[i][j].isApproximatelyEqual(mat, 0.001000f));
			}
		}
	}
	
	// Testing setInvert() functionality.
	{
		hkMatrix6 matrix_org;
		hkMatrix6 matrix_inv1;
		hkMatrix6 matrix_out;
		hkMatrix6 mat;
		hkVector4 first_row;
		hkVector4 second_row;
		hkVector4 third_row;

		first_row.set(1,2,3);
		second_row.set(0,4,5);
		third_row.set(1,0,6);

		matrix_org.m_m[0][0].setRows(first_row, second_row, third_row);
		matrix_org.m_m[1][1].setRows(first_row, second_row, third_row);

		first_row.set(2,3,5);
		second_row.set(5,1,5);
		third_row.set(6,7,8);
		
		matrix_org.m_m[0][1].setRows(first_row, second_row, third_row);
		matrix_org.m_m[1][0].setCols(first_row, second_row, third_row);
		//matrix_org.m_m[1][0].setRows(first_row, second_row, third_row);	// test fails if you enable this line as setInvert does require symmetric outer diagonal matrices

		matrix_inv1.setInvert(matrix_org);
		
		// Verification of setInvert by A * A^-1 = Identity 
		// Test fails ?
		matrix_out.setMul(matrix_org,matrix_inv1);
		hkMatrix6 mat_identity;
		mat_identity.setIdentity();
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				HK_TEST(matrix_out.m_m[i][j].isApproximatelyEqual(mat_identity.m_m[i][j]));
			}
		}
	}

	// Verifying functionality of hkVector8
	{
		hkVector8 vector;
		vector.setAll(2);
		hkVector8 vector1;
		vector1._setMul6(matrix_1,vector);
		hkVector4 vec1 = vector1.m_ang;
		hkVector4 vec2 = vector1.m_lin;
		for(int i = 0; i < 3; i++)
		{
			HK_TEST(vec1(i) == vec2(i));
		}
	}
}
int matrix6_main()
{
	matrix6();
	return 0;
}

//void ___1() { }
#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(matrix6_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
