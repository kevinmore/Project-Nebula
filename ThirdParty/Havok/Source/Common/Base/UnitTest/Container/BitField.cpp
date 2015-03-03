/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/BitField/hkBitField.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

static void BitField()
{
	// Testing get() & getSize().
	{
		hkBitField bf(8, hkBitFieldValue::UNINITIALIZED);
		for(int i = 0; i < 8; i++)
		{
			HK_TEST((bf.get(i) == 0) || (bf.get(i) == 1));
		}

		HK_TEST(bf.getSize() == 8);
	}
   
    // Testing get() & getSize() with value '1'.
	{
		hkBitField bf(16, hkBitFieldValue::ONE);
		for(int i = 0; i < 16; i++)
		{
			HK_TEST(bf.get(i) != 0);
		}

		HK_TEST(bf.getSize() == 16);
	}

	// Testing clear(). 
	{
		hkBitField bf(32, hkBitFieldValue::ONE);
		for(int i = 0; i < 32; i++)
		{
			HK_TEST(bf.get(i) == 1);
		}

		for(int i = 0; i < 32; i++)
		{
			bf.clear(i);
		}

		for(int i = 0; i < 32; i++)
		{
			HK_TEST(bf.get(i) == 0);
		}

		HK_TEST(bf.getSize() == 32);
	}

	// Testing set(). 
	{
		hkBitField bf(32, hkBitFieldValue::ZERO);
		for(int i = 0; i < 32; i++)
		{
			HK_TEST(bf.get(i) == 0);
		}

		for(int i = 0; i < 32; i++)
		{
			bf.set(i);
		}

		for(int i = 0; i < 32; i++)
		{
			HK_TEST(bf.get(i) == 1);
		}
	}

	// Testing assign(). 
	{
		hkBitField bf(32, hkBitFieldValue::ZERO);
		for(int i = 0; i < 32; i++)
		{
			HK_TEST(bf.get(17) == 0);
		}

		for(int i = 0; i < 32; i++)
		{
			bf.assign(i,1);
		}

		for(int i = 0; i < 32; i++)
		{
			HK_TEST(bf.get(17) == 1);
		}
	}

	// Testing assignAll().
	{
		hkBitField bf(16, hkBitFieldValue::ZERO);
		for(int i = 0; i < 16; i++)
		{
			HK_TEST(bf.get(i) == 0);
		}

		bf.assignAll(1);

		for(int i = 0; i < bf.getSize(); i++)
		{
			HK_TEST(bf.get(i) == 1);
		}
	}

	// Testing setSize().
	{
		hkBitField bf(32, hkBitFieldValue::ZERO);
		bf.setSizeAndFill(0, 12, 0);  
		HK_TEST(bf.getSize() == 12);
		for(int i = 0; i < bf.getSize(); i++)
		{
			HK_TEST(bf.get(i)==0);
		}
	}

	// Testing orWith().
	{
		hkBitField bf1;
		bf1.setSizeAndFill(0, 100, 0);
		bf1.set(0);
		bf1.set(25);

		hkBitField bf2;
		bf2.setSizeAndFill(0, 100, 0);
		bf2.set(25);
		bf2.set(50);

		bf1.orWith(bf2);
		HK_TEST(bf1.get(0) == 1);
		HK_TEST(bf1.get(25) == 1);
		HK_TEST(bf1.get(50) == 1);
		HK_TEST(bf1.get(75) == 0);
	}

	// Testing andWith().
	{
		hkBitField bf1;
		bf1.setSizeAndFill(0, 100, 0);
		bf1.set(0);
		bf1.set(25);

		hkBitField bf2;
		bf2.setSizeAndFill(0, 100, 0);
		bf2.set(25);
		bf2.set(50);

		bf1.andWith(bf2);
		HK_TEST(bf1.get(0) == 0);
		HK_TEST(bf1.get(25) == 1);
		HK_TEST(bf1.get(50) == 0);
		HK_TEST(bf1.get(75) == 0);
	}
}


static bool compareBitfields( const hkBitField& bf1, const hkLocalBitField& bf2, int startIndex = 0)
{
	if (bf1.getSize() != bf2.getSize())
	{
		return false;
	}

	for (int i=startIndex; i<bf1.getSize(); i++)
	{
		if ( bf1.get(i) != bf2.get(i) )
		{
			return false;
		}
	}

	return true;
}

static void testIterator(const hkBitField& bf, hkPseudoRandomGenerator& rand)
{
	hkLocalBitField copied(bf.getSize(), hkBitFieldValue::ZERO);

	{
		hkBitField::Iterator iter( bf );
		for ( int idx=0; iter.isValid(bf); iter.getNext(bf), idx++)
		{
			copied.assign(iter.getCurrentBit(), iter.isCurrentBitSet() );
		}

		HK_TEST( compareBitfields(bf, copied) );
		copied.assignAll(0);
	}

	// try restarting from a random (or not so random) spot a few times.
	for (int i=0; i<5; i++)
	{
		int startIndex = bf.getSize() ? rand.getRandInt16( bf.getSize() ) : 0;
		if (i == 0)
		{
			startIndex = 0;
		}
		else if (i == 1)
		{
			startIndex = bf.getSize();
		}

		hkBitField::Iterator iter( bf );
		iter.setPosition( bf, startIndex );

		for ( int idx=0; iter.isValid(bf); iter.getNext(bf), idx++)
		{
			copied.assign(iter.getCurrentBit(), iter.isCurrentBitSet() );
		}

		HK_TEST( compareBitfields(bf, copied, startIndex) );
		copied.assignAll(0);
	}

}

static void BitFieldIteratorTest()
{
	hkPseudoRandomGenerator rand(1234);

	for (int i=0; i<32*4; i++)
	{
		// All empty
		{
			hkBitField bf; bf.setSizeAndFill( 0, i, 0);
			testIterator(bf, rand);
		}

		// All set
		{
			hkBitField bf; bf.setSizeAndFill( 0, i, 1);
			testIterator(bf, rand);
		}

		// Random
		{
			hkBitField bf; bf.setSizeAndFill( 0, i, 0 );
			for (int j=0; j<i; j++)
			{
				bf.assign( j, rand.getRandChar(2) );
			}
			testIterator(bf, rand);
		}

	}
}

int bitfield_main()
{
	BitField();
	BitFieldIteratorTest();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(bitfield_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
