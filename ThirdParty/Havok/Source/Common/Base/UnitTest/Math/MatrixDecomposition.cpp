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

static void mathutil_test()
{
	hkVector4 dummy;
	hkPseudoRandomGenerator rand(10);

	// Testing decompose functionality
	{
		hkVector4 rand_first_row; 
		rand.getRandomVector01(rand_first_row);
			
		hkVector4 rand_second_row; 
		rand.getRandomVector01(rand_second_row);
			
		hkVector4 rand_third_row; 
		rand.getRandomVector01(rand_third_row);
		
		// Testing decomposeMatrix for 3 * 3 Matrix
		{	
			hkMatrix3 rand_mat3;
			rand_mat3.setRows( rand_first_row, rand_second_row, rand_third_row );
		
			hkMatrixDecomposition::Decomposition decompose;
			hkMatrixDecomposition::decomposeMatrix(rand_mat3,decompose);
			{
				HK_TEST(decompose.m_hasScale);
				HK_TEST(decompose.m_hasSkew);
			}
			
			// Testing Translation which should be zero for 3 * 3 matrix
			{
				HK_TEST(decompose.m_translation(0) == 0);
				HK_TEST(decompose.m_translation(1) == 0);
				HK_TEST(decompose.m_translation(2) == 0);
			}
			
			// Testing M = R x S as T = 0
			{
				hkVector4 ssr0;
				hkVector4 ssr1;
				hkVector4 ssr2;
				decompose.m_scaleAndSkew.getRows(ssr0, ssr1, ssr2, dummy);
				
				// Testing ScaleSkew = Scale * Skew
				{
					hkVector4 sr0;
					hkVector4 sr1;
					hkVector4 sr2;
					decompose.m_skew.getRows(sr0, sr1, sr2, dummy);
					hkVector4 scale = decompose.m_scale;
					hkReal firstnode = scale(0) * hkMath::fabs(sr0(0));
					HK_TEST(hkMath::equal(ssr0(0), firstnode));
					hkReal lastnode = scale(2) * hkMath::fabs(sr2(2));
					HK_TEST(hkMath::equal(ssr2(2), lastnode));
				}

				hkVector4 r0;
				hkVector4 r1;
				hkVector4 r2;
				decompose.m_basis.getRows(r0, r1, r2);
				
				hkReal first = (hkMath::fabs(ssr0(0))*r0(0))+(hkMath::fabs(ssr1(0))*r0(1))+(hkMath::fabs(ssr2(0))*r0(2));
				HK_TEST(hkMath::equal(rand_first_row(0), first));
				
				hkReal mid = (hkMath::fabs(ssr0(1))*r1(0))+(hkMath::fabs(ssr1(1))*r1(1))+(hkMath::fabs(ssr2(1))*r1(2));
				HK_TEST(hkMath::equal(rand_second_row(1), mid));

				hkReal last = (hkMath::fabs(ssr0(2))*r2(0))+(hkMath::fabs(ssr1(2))*r2(1))+(hkMath::fabs(ssr2(2))*r2(2));
				HK_TEST(hkMath::equal(rand_third_row(2), last));		
			}
		}

		// Testing decomposeMatrix for 4 x 4 Matrix
		{
			hkVector4 rand_fourth_row; 
			rand.getRandomVector01(rand_fourth_row);
			
			hkMatrix4 rand_mat4;
			rand_mat4.setRows( rand_first_row, rand_second_row, rand_third_row, rand_fourth_row );
			
			hkMatrixDecomposition::Decomposition decompose;
			hkMatrixDecomposition::decomposeMatrix(rand_mat4, decompose);
			
			{
				HK_TEST(decompose.m_hasScale);
				HK_TEST(decompose.m_hasSkew);
			}
			
			// Testing M = T * R * S  
			{
				hkVector4 ssr0;
				hkVector4 ssr1;
				hkVector4 ssr2;
				decompose.m_scaleAndSkew.getRows(ssr0, ssr1, ssr2, dummy);
				
				hkVector4 r0;
				hkVector4 r1;
				hkVector4 r2;
				decompose.m_basis.getRows(r0, r1, r2);
				
				hkReal first = (hkMath::fabs(ssr0(0))*r0(0))+(hkMath::fabs(ssr1(0))*r0(1))+(hkMath::fabs(ssr2(0))*r0(2));
				HK_TEST(hkMath::equal(rand_first_row(0), first));
				
				hkReal mid = (hkMath::fabs(ssr0(1))*r1(0))+(hkMath::fabs(ssr1(1))*r1(1))+(hkMath::fabs(ssr2(1))*r1(2));
				HK_TEST(hkMath::equal(rand_second_row(1), mid));
 
				hkReal last = (hkMath::fabs(ssr0(2))*r2(0))+(hkMath::fabs(ssr1(2))*r2(1))+(hkMath::fabs(ssr2(2))*r2(2));
				HK_TEST(hkMath::equal(rand_third_row(2), last));		
			}
		}
		
		// Testing decompose4x4ColTransform() functionality
		if (0) // See COM-771
		{
			hkVector4 rand_fourth_row; 
			rand.getRandomVector01(rand_fourth_row);
			
			hkMatrix4 rand_mat4;
			rand_mat4.setRows( rand_first_row, rand_second_row, rand_third_row, rand_fourth_row );
			
			HK_ALIGN_REAL(hkReal mat[16]);
			rand_mat4.get4x4RowMajor(mat);

			hkMatrixDecomposition::Decomposition decompose;
			hkMatrixDecomposition::decompose4x4ColTransform(mat, decompose);
			
			{
				HK_TEST(decompose.m_hasScale);
				HK_TEST(decompose.m_hasSkew);
			}
			
			// Testing M = T * R * S  
			{
				hkVector4 ssr0;
				hkVector4 ssr1;
				hkVector4 ssr2;
				decompose.m_scaleAndSkew.getRows(ssr0, ssr1, ssr2, dummy);
				
				hkVector4 r0;
				hkVector4 r1;
				hkVector4 r2;
				decompose.m_basis.getRows(r0, r1, r2);
				
				hkReal first = (hkMath::fabs(ssr0(0))*r0(0))+(hkMath::fabs(ssr1(0))*r0(1))+(hkMath::fabs(ssr2(0))*r0(2));
				HK_TEST(hkMath::equal(rand_first_row(0), first));
				
				hkReal mid = (hkMath::fabs(ssr0(1))*r1(0))+(hkMath::fabs(ssr1(1))*r1(1))+(hkMath::fabs(ssr2(1))*r1(2));
				HK_TEST(hkMath::equal(rand_second_row(1), mid));

				hkReal last = (hkMath::fabs(ssr0(2))*r2(0))+(hkMath::fabs(ssr1(2))*r2(1))+(hkMath::fabs(ssr2(2))*r2(2));
				HK_TEST(hkMath::equal(rand_third_row(2), last));		
			}
		}
	}
}


static hkReal HK_CALL atan2fApproximation2( hkReal x, hkReal y )
// todo check precision differences
{
	static const hkReal pi_quarter = hkReal(3.14159265358979f * 0.25f);

	hkReal abs_y = hkMath::fabs(y);

	// terms
	hkReal xPlusAbsY = x + abs_y;
	hkReal absYMinusX = abs_y - x;
	hkReal xMinusAbsY = x - abs_y;

	//const hkReal c2 = hkReal(-0.121079f);
	//const hkReal c3 = HK_REAL_PI * hkReal(0.25f) - hkReal(1) - c2;

	// case x<0
	hkReal angle0;
	{
		hkReal r = xPlusAbsY / absYMinusX;
		//r += c2 * r*r + c3 * r*r*r;
		angle0 = (hkReal(3) * pi_quarter) - (pi_quarter * r);
	}

	// case x>=0
	hkReal angle1;
	{
		hkReal r = xMinusAbsY / xPlusAbsY;
		//r += c2 * r*r + c3 * r*r*r;
		angle1 = pi_quarter - (pi_quarter * r);
	}

	// select case
	hkReal result = hkMath::fselectLessZero(x, angle0, angle1);

	// invert symmetry
	hkReal negResult = -result;
	return hkMath::fselectLessZero(y, negResult, result);
}

static hkReal HK_CALL atan2fApproximation3( hkReal x, hkReal y )
{
	hkReal fx = hkMath::fabs(x);
	hkReal fy = hkMath::fabs(y);

	hkReal result;
	const hkReal c2 = hkReal(-0.121079f);
	const hkReal c3 = HK_REAL_PI * hkReal(0.25f) - hkReal(1) - c2;

	{
		if ( fx <= fy )
		{
			fy += HK_REAL_EPSILON;
			hkReal a = fx / fy;
			result = a;
			result += c2 * a*a;
			result += c3 * a*a*a;
		}
		else
		{
			fx += HK_REAL_EPSILON;
			hkReal a = fy / fx;
			result = a;
			result += c2 * a*a;
			result += c3 * a*a*a;
			result = HK_REAL_PI * hkReal(0.5f) - result;
		}
	}

	if ( y < hkReal(0))
	{
		result = HK_REAL_PI - result;
	}

	if ( x < hkReal(0) )
	{
		result = -result;
	}
	return result;
}

void atan2_test()
{
	hkReal maxDiff2 = 0.0f;
	hkReal maxDiff3 = 0.0f;
	for (hkReal t = -HK_REAL_PI + HK_REAL_EPSILON; t < HK_REAL_PI; t+= 0.1f )
	{
		hkReal x = hkMath::sin( t );
		hkReal y = hkMath::cos( t );
		hkReal a1 = HK_STD_NAMESPACE::atan2(y,x);
		hkReal a2 = atan2fApproximation2(x,y);
		hkReal a3 = atan2fApproximation3(y,x);
		maxDiff2 = hkMath::max2( maxDiff2, hkMath::fabs(a2-a1));
		maxDiff3 = hkMath::max2( maxDiff3, hkMath::fabs(a3-a1));
		HK_REPORT( "Atan "<< a1 << " " << a2 << " " << a3 );
	}
	HK_REPORT(" MaxDiff2 " << maxDiff2);
	HK_REPORT(" MaxDiff3 " << maxDiff3);
}



int mathutil_main()
{
	atan2_test();
	mathutil_test();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(mathutil_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
