/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

static void inplace_array()
{
	{
		typedef hkInplaceArrayAligned16<hkReal,1> Array;
		Array a;
		a.pushBack(4);
		HK_TEST( a.wasReallocated() == false );
		a.pushBack(10);
		HK_TEST( a.wasReallocated() );
	}
}

class NonPodArrayElement
{
	public:
		NonPodArrayElement() { m_someObjectData = s_numLivingObjects % 3; ++s_numLivingObjects; }
		~NonPodArrayElement() { --s_numLivingObjects; }
		bool operator==( const NonPodArrayElement& other ) { return other.m_someObjectData == m_someObjectData; }
		bool operator!=( const NonPodArrayElement& other ) { return other.m_someObjectData != m_someObjectData; }

		static int s_numLivingObjects;
		int m_someObjectData;
};

int NonPodArrayElement::s_numLivingObjects = 0;

int array_main()
{
	NonPodArrayElement::s_numLivingObjects = 0; // Otherwise test will fail on second pass...

	inplace_array();
	{
		int a12345[] = {1,2,3,4,5};
		hkArray<int> a( a12345, HK_COUNT_OF(a12345), HK_COUNT_OF(a12345) ); a.reserve(10); // ensure copied
		hkArray<int> b( a12345, HK_COUNT_OF(a12345), HK_COUNT_OF(a12345) ); b.reserve(10);
		hkArray<int> c( a12345, HK_COUNT_OF(a12345), HK_COUNT_OF(a12345) ); c.reserve(10);

		const int a678[] = {6,7,8};
		a.spliceInto( 1, 1, a678, HK_COUNT_OF(a678) );
		b.spliceInto( 1, 4, a678, HK_COUNT_OF(a678) );
		c.spliceInto( 1, 3, a678, HK_COUNT_OF(a678) );

		const int aout[] = { 1,6,7,8,3,4,5 };
		const int bout[] = { 1,6,7,8 };
		const int cout[] = { 1,6,7,8,5 };

		HK_TEST( HK_COUNT_OF(aout) == a.getSize() );
		HK_TEST( hkString::memCmp( a.begin(), aout, a.getSize() * sizeof(int) ) == 0 );
		HK_TEST( HK_COUNT_OF(bout) == b.getSize() );
		HK_TEST( hkString::memCmp( b.begin(), bout, b.getSize() * sizeof(int) ) == 0 );
		HK_TEST( HK_COUNT_OF(cout) == c.getSize() );
		HK_TEST( hkString::memCmp( c.begin(), cout, c.getSize() * sizeof(int) ) == 0 );
	}
	{
		hkArray<int> a; 
		const int a12[] = {1,2};
		a.append(a12, HK_COUNT_OF(a12) );
		HK_TEST( a.getSize() == HK_COUNT_OF(a12) );
		const int a3456789[] = {3,4,5,6,7,8,9};
		a.append(a3456789, HK_COUNT_OF(a3456789) );
		HK_TEST( a.getSize() == HK_COUNT_OF(a12)+HK_COUNT_OF(a3456789) );
		for( int i = 0; i < a.getSize(); ++i )
		{
			HK_TEST(a[i]==i+1);
		}
	}
	{
		char ctest[] = {'H','e','l','l','o'};
		hkArray<char> c( ctest, HK_COUNT_OF(ctest), HK_COUNT_OF(ctest) ); c.reserve(10);
		c.spliceInto( 2, 0, ctest, HK_COUNT_OF(ctest) );
		HK_TEST( c.getSize() == 10 );
		HK_TEST( hkString::memCmp( &c[2], ctest, HK_COUNT_OF(ctest) ) == 0 );
		HK_TEST( hkString::memCmp( &c[7], &ctest[2], 3 ) == 0 );
		c.spliceInto( 0, 2, ctest, HK_COUNT_OF(ctest) );
		HK_TEST( c.getSize() == 13 );
		HK_TEST( hkString::memCmp( c.begin(), ctest, HK_COUNT_OF(ctest) ) == 0 );
		HK_TEST( hkString::memCmp( &c[5], ctest, HK_COUNT_OF(ctest) ) == 0 );
		HK_TEST( hkString::memCmp( &c[10], &ctest[2], 3 ) == 0 );
	}
	{
		// Test removeAll on nonPod objects.
		hkArray<NonPodArrayElement> a( 9 );
		NonPodArrayElement zero;
		NonPodArrayElement one;
		NonPodArrayElement two;
		a.removeAllAndCopy( zero );
		HK_TEST( NonPodArrayElement::s_numLivingObjects == 9 );
		HK_TEST( a.getSize() == 6 );
		a.removeAllAndCopy( one );
		HK_TEST( NonPodArrayElement::s_numLivingObjects == 6 );
		HK_TEST( a.getSize() == 3 );
		a.removeAllAndCopy( two );
		HK_TEST( NonPodArrayElement::s_numLivingObjects == 3 );
		HK_TEST( a.getSize() == 0 );
	}
	{
		// Similar for removeAll on smallArray.
		hkSmallArray<int> a;
		for ( int i = 0; i < 9; ++i )
		{
			a.pushBack( i % 3 );
		}
		a.removeAllAndCopy( 0 );
		HK_TEST( a.getSize() == 6 );
		a.removeAllAndCopy( 1 );
		HK_TEST( a.getSize() == 3 );
		a.removeAllAndCopy( 2 );
		HK_TEST( a.getSize() == 0 );
	}
	{
		int tempData[5] = {1,2,3,4,5};
		hkArrayBase<int> tempArray(tempData, 5, 5);
		HK_TEST( tempArray.getSize() == 5 );
		// hkArrayBase destructor shouldn't assert, even though the size is non-zero, because we're using a POD type.
	}
	{
		NonPodArrayElement tempData[5];
		hkArrayBase<NonPodArrayElement> tempArray(tempData, 5, 5);
		HK_TEST( tempArray.getSize() == 5 );
		HK_TEST( NonPodArrayElement::s_numLivingObjects == 5 );

		tempArray.clear();		// hkArrayBase destructor will assert if we don't clear
		HK_TEST( NonPodArrayElement::s_numLivingObjects == 0 );
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(array_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
