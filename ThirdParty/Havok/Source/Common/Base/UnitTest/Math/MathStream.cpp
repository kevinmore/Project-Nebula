/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

static void mathStream_test()
{
	// Testing of hkVector4
	{
		hkArray<char> cbuf;
		hkOstream os(cbuf);
		HK_TEST( cbuf.getCapacity() >= 1 );
		HK_TEST( cbuf.begin()[0] == 0 );

		hkQuadReal vectVal = HK_QUADREAL_CONSTANT(1,2,3,4);
		hkVector4 vect; vect.m_quad = vectVal;
		os << vect;
		HK_TEST(os.isOk());
		const char* str = "[1,2,3,4]";
		
		HK_TEST( hkString::strCmp( cbuf.begin(), str) == 0 );
		HK_TEST( hkString::strLen( cbuf.begin() ) == hkString::strLen( str ) );		
	}
	// Testing of hkQuaternion
	{
		hkArray<char> cbuf;
		hkOstream os(cbuf);
		HK_TEST( cbuf.getCapacity() >= 1 );
		HK_TEST( cbuf.begin()[0] == 0 );

		hkQuaternion q;
		q.setIdentity();
		os << q;
		HK_TEST(os.isOk());
		const char* str = "[0.000000,0.000000,0.000000,(1.000000)]";
		HK_TEST( hkString::strCmp( cbuf.begin(), str ) == 0 );
		HK_TEST( hkString::strLen( cbuf.begin() ) == hkString::strLen( str ) );		
	}
	// Testing of hkMatrix3
	{
		hkArray<char> cbuf;
		hkOstream os(cbuf);
		HK_TEST( cbuf.getCapacity() >= 1 );
		HK_TEST( cbuf.begin()[0] == 0 );

		hkMatrix3 mat;
		hkQuadReal vect0Val = HK_QUADREAL_CONSTANT(1,2,3,0);
		hkVector4 vect0; vect0.m_quad=vect0Val;
		hkQuadReal vect1Val = HK_QUADREAL_CONSTANT(4,5,6,0);
		hkVector4 vect1; vect1.m_quad=vect1Val;
		hkQuadReal vect2Val = HK_QUADREAL_CONSTANT(7,8,9,0);
		hkVector4 vect2; vect2.m_quad=vect2Val;
		mat.setCols(vect0,vect1,vect2);
		os << mat;
		const char* str = "|1.000000,4.000000,7.000000|\n|2.000000,5.000000,8.000000|\n|3.000000,6.000000,9.000000|\n";
		HK_TEST( hkString::strCmp( cbuf.begin(), str ) == 0 );
		HK_TEST( hkString::strLen( cbuf.begin() ) == hkString::strLen( str ) );		
	}
	// Testing hkTransform
	{
		hkArray<char> cbuf;
		hkOstream os(cbuf);
		HK_TEST( cbuf.getCapacity() >= 1 );
		HK_TEST( cbuf.begin()[0] == 0 );

		hkVector4 vaxis; vaxis = hkVector4::getConstant<HK_QUADREAL_1000>();
		hkQuadReal vtransVal = HK_QUADREAL_CONSTANT(2,3,4,0);
		hkVector4 vtrans; vtrans.m_quad=vtransVal;
		hkRotation rot;
		rot.setIdentity();
		hkTransform t; t.set(rot,vtrans);
		os << t;
		
		const char* str = "|1.000000,0.000000,0.000000|\n|0.000000,1.000000,0.000000|\n|0.000000,0.000000,1.000000|\n[2,3,4,0]";
		HK_TEST( hkString::strCmp( cbuf.begin(), str) == 0 );
		HK_TEST( hkString::strLen( cbuf.begin() ) == hkString::strLen( str ) );		
	
	}
	// Testing of hkQuadReal 
	// It is giving build error on win32_SIMD platform but working  on win32
	{
		/*
		hkArray<char> cbuf;
		hkOstream os(cbuf);
		HK_TEST( cbuf.getCapacity() >= 1 );
		HK_TEST( cbuf.begin()[0] == 0 );

		hkQuadReal qreal;
		
		qreal.x = 4.5f;
		qreal.y = 10.3f;
		qreal.z = 11.1f;
		qreal.w = 1.5f;
		os << qreal;
		HK_TEST(os.isOk());
		const char* str = "[4.500000,10.300000,11.100000,1.500000]";
		HK_TEST( hkString::strCmp( cbuf.begin(), str) == 0 );
		HK_TEST( hkString::strLen( cbuf.begin() ) == hkString::strLen( str ) );
		*/
	}
}

int mathStream_main()
{
	mathStream_test();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(mathStream_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
