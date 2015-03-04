/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>


namespace
{
	struct TestConvert
	{
		template<typename Tf, typename Td> 
		static void test1()
		{
			{
				Tf vf; vf.setZero();
				Td vd = vf;
			}
			{
				Td vd; vd.setZero();
				Tf vf = vd;
			}
		}

		template<typename Tf, typename Td> 
		static void test2()
		{
			{
				Tf vf; vf.setIdentity();
				Td vd = vf;
			}
			{
				Td vd; vd.setIdentity();
				Tf vf = vd;
			}
		}
	};
}

static void floatDouble_test()
{
#define HK_DISABLE_MIXED_FLOAT_DOUBLE_CONVERSIONS

//	Currently we have no way of requiring compilation failure of a piece of code in the test framework.
//	This code rightfully fails to compile. 
//	It is therefore commented out for later use in combination with HK_DISABLE_MIXED_FLOAT_DOUBLE_CONVERSIONS
#ifndef	HK_DISABLE_MIXED_FLOAT_DOUBLE_CONVERSIONS

	TestConvert::test1<hkVector4f, hkVector4d>();
	TestConvert::test1<hkSimdFloat32, hkSimdDouble64>();
	TestConvert::test1<hkMatrix3f, hkMatrix3d>();
	TestConvert::test2<hkTransformf, hkTransformd>();

#endif
}

int floatDouble_main()
{
	floatDouble_test();
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(floatDouble_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/FloatDouble.cpp"     );

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
