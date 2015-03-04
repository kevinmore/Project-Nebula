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
#include <Common/Base/Types/hkRefPtr.h>

static void baseTypes_halfTest()
{
	hkPseudoRandomGenerator random(1);
	{
		for (int i =0; i < 1000; i++)
		{
			hkReal x = random.getRandRange( -FLT_MAX*0.5f, FLT_MAX*0.5f );
			hkReal maxError = hkMath::fabs(x * 0.01f);
			hkHalf half; half.setReal<true>(x);
			hkReal uncompressed = half.getReal();
			HK_TEST( hkMath::equal(uncompressed, x, maxError));
		}
	}
	{
		for (int i =0; i < 1000; i++)
		{
			hkReal x = random.getRandRange( -100.0f, 100.0f );
			hkReal maxError = hkMath::fabs(x * 0.01f);
			hkHalf half; half.setReal<false>(x);
			hkReal uncompressed = half.getReal();
			HK_TEST( hkMath::equal(uncompressed, x, maxError));
		}
	}
	{
		for (int i =0; i < 1000; i++)
		{
			hkReal x = random.getRandRange( -0.001f, 0.001f );
			hkReal maxError = hkMath::fabs(x * 0.01f);
			hkHalf half; half.setReal<true>(x);
			hkReal uncompressed = half.getReal();
			HK_TEST( hkMath::equal(uncompressed, x, maxError));
		}
	}
}

#ifndef HK_PLATFORM_SPU
static hkRefLoan<hkReferencedObject> borrowedRef(hkReferencedObject& r)
{
	return &r;
}

// simulate returning a new ref on an existing object
static hkRefNew<hkReferencedObject> newRef(hkReferencedObject& r)
{
	r.addReference();
	return &r;
}

static void baseTypes_refptrTest()
{
	hkReferencedObject r0; 
	r0.m_memSizeAndFlags = sizeof(hkReferencedObject); // Explicitly initialize to non zero value - zero has special meaning for packfiles.
	{
		HK_TEST( r0.getReferenceCount() == 1 );
		borrowedRef(r0); // ok, ignore return
		hkReferencedObject* r = borrowedRef(r0); // ok, use return
		HK_TEST(r->getReferenceCount() == 1);
		hkRefPtr<hkReferencedObject> rr = borrowedRef(r0); // take another ref
		hkRefPtr<hkReferencedObject> r2 = HK_NULL;
		HK_TEST( r0.getReferenceCount() == 2 );
		r2 = borrowedRef(r0);
		HK_TEST( r0.getReferenceCount() == 3 );
	}
	HK_TEST( r0.getReferenceCount() == 1 );
	{
		HK_TEST( r0.getReferenceCount() == 1 );
		hkReferencedObject* r = newRef(r0).stealOwnership();
		HK_TEST( r0.getReferenceCount() == 2 );
		r->removeReference();
		HK_TEST( r0.getReferenceCount() == 1 );

		//hkReferencedObject* s = newRef(r0); // shouldn't compile
		hkRefPtr<hkReferencedObject> rr = newRef(r0); // take another ref
		hkRefPtr<hkReferencedObject> r2 = HK_NULL;
		HK_TEST( r0.getReferenceCount() == 2 );
		r2 = newRef(r0);
		HK_TEST( r0.getReferenceCount() == 3 );
	}
	HK_TEST( r0.getReferenceCount() == 1 );
}
#endif

static void baseTypes_ufloat8Test()
{
	hkPseudoRandomGenerator random(1);
	{
		for (int i =0; i < 1000; i++)
		{
			hkReal x = random.getRandRange( 0, hkUFloat8_maxValue );
			hkReal maxError = hkMath::max2( hkReal(hkUFloat8_eps), hkMath::fabs(x * 0.1f) );
			hkUFloat8 half = float(x);
			hkReal uncompressed = half;
			HK_TEST( hkMath::equal(uncompressed, x, maxError));
		}
	}
	{
		for (int i =0; i < 1000; i++)
		{
			hkReal x = random.getRandRange( 0.0f, 100.0f );
			hkReal maxError = hkMath::max2(  hkReal(hkUFloat8_eps), hkMath::fabs(x * 0.1f) );
			hkUFloat8 half = float(x);
			hkReal uncompressed = half;
			HK_TEST( hkMath::equal(uncompressed, x, maxError));
		}
	}
	{
		for (int i =0; i < 1000; i++)
		{
			hkReal x = random.getRandRange( 0.0, 0.1f );
			hkReal maxError = hkMath::max2(  hkReal(hkUFloat8_eps), hkMath::fabs(x * 0.1f) );
			hkUFloat8 half = float(x);
			hkReal uncompressed = half;
			HK_TEST( hkMath::equal(uncompressed, x, maxError));
		}
	}
}

int baseTypes_main()
{
	baseTypes_halfTest();
	baseTypes_ufloat8Test();
#ifndef HK_PLATFORM_SPU
	baseTypes_refptrTest();
#endif
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(baseTypes_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
