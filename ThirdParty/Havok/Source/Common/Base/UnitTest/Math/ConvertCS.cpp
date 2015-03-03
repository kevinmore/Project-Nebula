/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

static void ConvertCS()
{
	//Testing convertQuaternion().
	{
		hkVector4 axis;
		axis.set(1,0,0);
		axis.normalize<3>();
		hkReal angle = 0.7f;
		hkQuaternion q0; q0.setAxisAngle(axis, angle);
		hkVector4 vimg_org = q0.getImag();

		hkConvertCS cs;
		cs.convertQuaternion(q0);

		// Axis verification test
		{
			HK_TEST(q0.isOk());
			HK_TEST(hkMath::equal(q0.getAngle(), angle, 1e-3f));

			hkVector4 axisTest;
			q0.getAxis(axisTest);
			HK_TEST( axisTest.isNormalized<3>() );
			HK_TEST( hkMath::fabs(axisTest(0) - axis(0)) < 1e-3f);
			HK_TEST( hkMath::fabs(axisTest(1) - axis(1)) < 1e-3f);
			HK_TEST( hkMath::fabs(axisTest(2) - axis(2)) < 1e-3f);
		}

		// convertQuaternion imaginary component verification test
		{
			hkVector4 vimg_cur = q0.getImag();

			hkRotation rot;
			rot.setIdentity();

			hkVector4 vout;
			vout._setRotatedDir(rot,vimg_org);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( vimg_cur.allEqual<3>(vout,eps) );
		}
	}

	// Testing convertVector() & setConversionRotation().
    {
		hkVector4 axis;
		axis.set(5,2,-4);
		axis.normalize<3>();

		hkRotation r;
		r.setAxisAngle( axis, 0.62f);

		hkVector4 v0;
		v0.set(2,3,4);
		hkVector4 vorg = v0;

		hkConvertCS cs;
		cs.setConversionRotation(r);
		cs.convertVector(v0);

		// verification test for convertVector()
		hkVector4 vout ;
		vout._setRotatedDir(r,vorg);
		hkSimdReal eps; eps.setFromFloat(1e-3f);
		HK_TEST( v0.allEqual<3>(vout,eps) );
	}

	//Testing convertRotation().
	{
		hkVector4 rand_first_row;
		rand_first_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		rand_first_row.normalize<3>();
		hkVector4 rand_second_row;
		rand_second_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		hkVector4 rand_third_row;
		rand_third_row.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );

		hkMatrix3 rand_mat3;
		rand_mat3.setRows( rand_first_row, rand_second_row, rand_third_row );

		hkVector4 rows[3];
		rand_mat3.getRows( rows[0], rows[1], rows[2] );

		hkRotation rand_rot;
		rand_rot.setRows( rows[0], rows[1], rows[2] );

		hkRotation org_rot_inout;
		org_rot_inout = rand_rot;

		hkConvertCS cs;
		cs.convertRotation(rand_rot);

		// Verification test for convertRotation operation
		hkRotation rot;
		rot.setIdentity();

		hkRotation temp_rot;
		temp_rot.setMulInverse(org_rot_inout,rot);
		org_rot_inout.setMul(rot, temp_rot);

		HK_TEST(rand_rot.isApproximatelyEqual(org_rot_inout));
	}

	//Testing convertRotationAngle().
	{
		hkVector4 axis;
		axis.set(5,2,-4);
		hkReal angle = 0.3f;
		hkReal angle1 = angle;
		axis.normalize<4>();
		hkConvertCS cs;
		cs.convertRotationAngle(angle);
		HK_TEST( hkMath::fabs(angle - angle1) < 1e-3f);
	}

	//Testing convertMinMaxAngles().
	{
		hkVector4 axis;
		axis.set(5,2,-4);
		hkReal max = 0.3f;
		hkReal min = 0.7f;
		hkConvertCS cs;
		cs.convertMinMaxAngles(min,max);
		HK_TEST(max > min);
		HK_TEST(max == 0.7f);
		HK_TEST(min == 0.3f);
	}

	//Testing convertTransform().
	{
		hkVector4 v0;
		v0.set(4,1,7,2);
		hkVector4 vorg = v0;

		hkVector4 axis;
		axis.set(5,2,-4);
		axis.normalize<4>();

		hkRotation r;
		r.setAxisAngle( axis, 0.62f);
		hkRotation org = r;

		hkTransform t;
		t.setRotation(org);
		t.setTranslation(vorg);

		hkConvertCS cs;
		cs.convertRotation(r);
		cs.convertVector(v0);
		cs.convertTransform(t);

		// Verification test of translation of transformed t
		{
			hkVector4 vt = t.getTranslation();

			hkVector4 vout;
			hkRotation identity_rot;
			identity_rot.setIdentity();
			vout._setRotatedDir(identity_rot,vorg);

			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( vt.allEqual<3>(vout,eps) );
		}

		// Verification test of rotation of transformed t
		{
			hkRotation vrot = t.getRotation();

			hkRotation rot;
			rot.setIdentity();

			hkRotation temp_rot;
			temp_rot.setMulInverse(org,rot);
			org.setMul(rot, temp_rot);

			HK_TEST(vrot.isApproximatelyEqual(org));
		}
	}

	// Testing setConversionType()
	{
		// Verification test for conversion type CT_FLIP_X
		{
			hkConvertCS cs;
			cs.setConversionType(hkConvertCS::CT_FLIP_X);
			hkVector4 v0;
			v0.set(2,3,4);
			hkVector4 vorg = v0;
			cs.convertVector(v0);
			hkRotation rot;
			rot.setIdentity();
			rot(0,0) = -1;
			hkVector4 vout ;
			vout._setRotatedDir(rot,vorg);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( v0.allEqual<3>(vout,eps) );
		}

		// Verification test for conversion type CT_FLIP_Y
		{
			hkConvertCS cs;
			cs.setConversionType(hkConvertCS::CT_FLIP_Y);
			hkVector4 v0;
			v0.set(2,3,4);
			hkVector4 vorg = v0;
			cs.convertVector(v0);
			hkRotation rot;
			rot.setIdentity();
			rot(1,1) = -1;
			hkVector4 vout ;
			vout._setRotatedDir(rot,vorg);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( v0.allEqual<3>(vout,eps) );
		}

		// Verification test for conversion type CT_FLIP_Z
		{
			hkConvertCS cs;
			cs.setConversionType(hkConvertCS::CT_FLIP_Z);
			hkVector4 v0;
			v0.set(2,3,4);
			hkVector4 vorg = v0;
			cs.convertVector(v0);
			hkRotation rot;
			rot.setIdentity();
			rot(2,2) = -1;
			hkVector4 vout ;
			vout._setRotatedDir(rot,vorg);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( v0.allEqual<3>(vout,eps) );
		}

		// Verification test for conversion type CT_SWITCH_YZ
		{
			hkConvertCS cs;
			cs.setConversionType(hkConvertCS::CT_SWITCH_YZ);
			hkVector4 v0;
			v0.set(2,3,4);
			hkVector4 vorg = v0;
			cs.convertVector(v0);
			hkRotation rot;
			hkVector4 c0;
			c0.set(1,0,0,0);
			hkVector4 c1;
			c1.set(0,0,1,0);
			hkVector4 c2;
			c2.set(0,1,0,0);
			rot.setCols(c0,c1,c2);
			hkVector4 vout ;
			vout._setRotatedDir(rot,vorg);
			hkSimdReal eps; eps.setFromFloat(1e-3f);
			HK_TEST( v0.allEqual<3>(vout,eps) );
		}

	}
}

int convertCS_main()
{
	ConvertCS();
    return 0;
}

#if defined(HK_COMPILER_MWERKS)
#   pragma fullpath_file on
#endif
HK_TEST_REGISTER(convertCS_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
