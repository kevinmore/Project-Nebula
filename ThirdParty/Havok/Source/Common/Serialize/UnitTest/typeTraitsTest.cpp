/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

struct Test_Non_Reflected_Forwarded;
HK_DEFINE_AS_POD_TYPE(Test_Non_Reflected_Forwarded);

struct Test_Non_Reflected_Forwarded {};

struct Test_Non_ReflectedBase
{
	HK_DECLARE_POD_TYPE();
};
struct Test_Non_Reflected : public Test_Non_ReflectedBase
{
	HK_DECLARE_POD_TYPE();
	hkVector4 mydata;
};
struct Test_Non_ReflectedBase_Non_Pod
{
	int data;
};
struct Test_Non_Reflected_Non_Pod : public Test_Non_ReflectedBase_Non_Pod
{
	hkVector4 moredata;
};

template <typename T>
struct Test_Template
{
	HK_DECLARE_POD_TYPE_IF_POD(T);

	T data;
};

// we may overrule definition of Test_Template and define it as pod, disregarding a non pod type T
HK_DEFINE_AS_POD_TYPE( Test_Template<Test_Non_ReflectedBase_Non_Pod> );

struct Test_Reflected_Pod_Forwarded;
HK_DEFINE_AS_POD_TYPE(Test_Reflected_Pod_Forwarded);

struct Test_Reflected_Pod_Forwarded
{
	HK_DECLARE_REFLECTION();

	int m_data;
};

struct Test_Reflected_Pod
{
	HK_DECLARE_REFLECTION();
	HK_DECLARE_POD_TYPE();

	int m_data;
};

static void typeTraitsTestPod()
{
	// integral Havok types
	HK_TEST(hkTrait::IsPodType<bool>::result );
	HK_TEST(hkTrait::IsPodType<hkBool>::result );
	HK_TEST(hkTrait::IsPodType<char>::result );
	HK_TEST(hkTrait::IsPodType<hkInt8>::result );
	HK_TEST(hkTrait::IsPodType<hkUint8>::result );
	HK_TEST(hkTrait::IsPodType<hkInt16>::result );
	HK_TEST(hkTrait::IsPodType<hkUint16>::result );
	HK_TEST(hkTrait::IsPodType<hkInt32>::result );
	HK_TEST(hkTrait::IsPodType<hkUint32>::result );
	HK_TEST(hkTrait::IsPodType<hkInt64>::result );
	HK_TEST(hkTrait::IsPodType<hkUint64>::result );
	HK_TEST(hkTrait::IsPodType<hkLong>::result );
	HK_TEST(hkTrait::IsPodType<hkUlong>::result );
	HK_TEST(hkTrait::IsPodType<hkUFloat8>::result );
	HK_TEST(hkTrait::IsPodType<hkHalf>::result );
	HK_TEST(hkTrait::IsPodType<hkReal>::result );
	HK_TEST(hkTrait::IsPodType<hkDouble64>::result );

	HK_TEST(hkTrait::IsPodType<hkVector2>::result );
	HK_TEST(hkTrait::IsPodType<hkVector4>::result );
	HK_TEST(hkTrait::IsPodType<hkQuaternion>::result );
	HK_TEST(hkTrait::IsPodType<hkMatrix3>::result );
	HK_TEST(hkTrait::IsPodType<hkRotation>::result );
	HK_TEST(hkTrait::IsPodType<hkQsTransform>::result );
	HK_TEST(hkTrait::IsPodType<hkMatrix4>::result );
	HK_TEST(hkTrait::IsPodType<hkTransform>::result );

	HK_TEST(hkTrait::IsPodType< Test_Template<hkInt32> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkVector4> >::result );

	// non reflected struct tests
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected_Forwarded>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_ReflectedBase>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected>::result );
	HK_TEST(hkTrait::IsPodType<Test_Reflected_Pod_Forwarded>::result );
	HK_TEST(hkTrait::IsPodType<Test_Reflected_Pod>::result );

	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected_Forwarded> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Reflected_Pod_Forwarded> >::result );

	// array tests
	HK_TEST(hkTrait::IsPodType<hkInt32[3]>::result );
	HK_TEST(hkTrait::IsPodType<hkVector4[3]>::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkInt32[3]> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkVector4[3]> >::result );

	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected_Forwarded[3]>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_ReflectedBase[3]>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected[3]>::result );
	HK_TEST(hkTrait::IsPodType<Test_Reflected_Pod_Forwarded[3]>::result );
	HK_TEST(hkTrait::IsPodType<Test_Reflected_Pod[3]>::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected_Forwarded[3]> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_ReflectedBase[3]> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected[3]> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Reflected_Pod_Forwarded[3]> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Reflected_Pod[3]> >::result );

	// pointer tests
	HK_TEST(hkTrait::IsPodType<hkInt32*>::result );
	HK_TEST(hkTrait::IsPodType<hkVariant*>::result );
	HK_TEST(hkTrait::IsPodType<hkVector4*>::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkInt32*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkVariant*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkVector4*> >::result );

	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected_Forwarded*>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_ReflectedBase*>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected*>::result );
	HK_TEST(hkTrait::IsPodType<Test_Reflected_Pod_Forwarded*>::result );
	HK_TEST(hkTrait::IsPodType<Test_Reflected_Pod*>::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected_Forwarded*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_ReflectedBase*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Reflected_Pod_Forwarded*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Reflected_Pod*> >::result );

	// pointers to non-pod are pods
	HK_TEST(hkTrait::IsPodType<hkReferencedObject*>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_ReflectedBase_Non_Pod*>::result );
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected_Non_Pod*>::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<hkReferencedObject*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_ReflectedBase_Non_Pod*> >::result );
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected_Non_Pod*> >::result );
}

static void typeTraitsTestNonPod()
{
	HK_TEST(hkTrait::IsPodType<hkReferencedObject>::result == false);
	HK_TEST(hkTrait::IsPodType< Test_Template<hkReferencedObject> >::result == false);

	HK_TEST(hkTrait::IsPodType<Test_Non_ReflectedBase_Non_Pod>::result == false);
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected_Non_Pod>::result == false);

	// hkIsPodType<Test_Non_ReflectedBase_Non_Pod> is defined as pod type, skip it
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected_Non_Pod> >::result == false);

	// array tests
	HK_TEST(hkTrait::IsPodType<hkReferencedObject[3]>::result == false);

	HK_TEST(hkTrait::IsPodType<Test_Non_ReflectedBase_Non_Pod[3]>::result == false);
	HK_TEST(hkTrait::IsPodType<Test_Non_Reflected_Non_Pod[3]>::result == false);
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_ReflectedBase_Non_Pod[3]> >::result == false);
	HK_TEST(hkTrait::IsPodType< Test_Template<Test_Non_Reflected_Non_Pod[3]> >::result == false);
}

static int typeTraitsTest()
{
	typeTraitsTestPod();
	typeTraitsTestNonPod();

	// test overruled definition of Test_Template as pod, disregarding a non pod type Test_Non_ReflectedBase_Non_Pod
	HK_TEST((bool)hkTrait::IsPodType< Test_Template<Test_Non_ReflectedBase_Non_Pod> >::result != (bool)hkTrait::IsPodType<Test_Non_ReflectedBase_Non_Pod>::result);
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER( typeTraitsTest, "Fast", "Common/Test/UnitTest/Serialize/", "typeTraitsTest" );

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
