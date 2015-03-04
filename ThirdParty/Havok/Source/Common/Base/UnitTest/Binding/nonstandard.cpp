/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

// The binding project relies on "undefined behavior". Namely, using method pointers
// with the wrong object. This unit test determines whether that the behavior is 
// nevertheless valid on all our supported platform/compiler combinations.

class ActualClass
{
public:
	int m(int x, int y)
	{
		return x * y;
	}

	virtual int vm(int x, int y)
	{
		return x * y * 2;
	}

	virtual ~ActualClass() {}

public:
	int m_testVal;
};

class ActualClass2 : public ActualClass
{
public:
	virtual int vm(int x, int y)
	{
		return x * y * 3;
	}
};

class DummyParent1 {};
class DummyParent2 {};

// We have to ensure that we obtain the method pointer on a class with
// more than one parent.

class ActualClass_SubClass : public ActualClass, public DummyParent2
{
};

class ActualClass2_SubClass : public ActualClass2, public DummyParent2
{
};

// This is the class we'll be pretending the methods belong to when we
// call them.
class SurrogateClass : public DummyParent1, public DummyParent2
{
public:
	int dummy( int x, int y)
	{
		// Should never be called.
		HK_TEST(false);
		return 0;
	}
};

int nonstandard_main()
{
	typedef int (SurrogateClass::*faketype)(int, int);
	typedef int (ActualClass_SubClass::*mptype)(int, int);
	typedef int (ActualClass2_SubClass::*mptype2)(int, int);

	HK_COMPILE_TIME_ASSERT(sizeof(faketype) == sizeof(mptype));
	HK_COMPILE_TIME_ASSERT(sizeof(faketype) == sizeof(mptype2));

	ActualClass a;
	ActualClass2 a2;
	SurrogateClass* b = (SurrogateClass*) &a;
	SurrogateClass* b2 = (SurrogateClass*) &a2;

	// Call the method mp directly.
	{
		mptype mp = &ActualClass_SubClass::m;

		union {
			mptype m_actual;
			faketype m_fake;
		} u = { mp };

		int r = (b->*(u.m_fake))(3, 5);
		HK_TEST(r == 15);
	}

	// Check that it still works for virtual methods.
	{
		mptype mp = &ActualClass_SubClass::vm;

		union {
			mptype m_actual;
			faketype m_fake;
		} u = { mp };
		
		int r = (b->*(u.m_fake))(4, 6);

		HK_TEST(r == 48);
	}

	// Check it calls the over-ridding implementation when called on a subclass.
	{
		mptype2 mp = &ActualClass2_SubClass::vm;

		union {
			mptype2 m_actual;
			faketype m_fake;
		} u = { mp };
		
		int r = (b2->*(u.m_fake))(5, 7);
		HK_TEST(r == 105);
	}

	// Check it calls the over-ridding implementation even when called on a
	// superclass.
	{
		mptype mp = &ActualClass_SubClass::vm;

		union {
			mptype m_actual;
			faketype m_fake;
		} u = { mp };
	
		int r = (b2->*(u.m_fake))(8, 9);
		HK_TEST(r == 216);
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(nonstandard_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

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
