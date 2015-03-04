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
#include <Common/Base/Container/PointerMultiMap/hkPointerMultiMap.h>

static void pointerMultiMap_selfTest()
{
    // General checking
    {
		hkPointerMultiMap<int, char> map;

		map.insert(10, 'a');
		map.insert(10, 'b');
		map.insert(11, 'b');
		map.insert(10, 'b');

        HK_TEST(map.findNumEntries(10) == 3);
        HK_TEST(map.findNumEntries(10, 'a') == 1);

		int numRemoved = map.removeAll(10);
        HK_TEST(numRemoved == 3);

		map.insert(10, 'a');
		map.insert(10, 'b');
		map.insert(10, 'c');

		int bits = 0;
		hkPointerMultiMap<int, char>::Iterator iter = map.findKey(10);
		for (; map.isValid(iter); iter = map.getNext(iter, 10))
		{
			char value = map.getValue(iter);
			HK_TEST(value >= 'a' && value <= 'c');
			int bit = 1 << ( value - 'a' );
			HK_TEST((bits & bit) == 0);

			bits |= bit;
		}

        HK_TEST( bits == 0x7);
    }


    // Checking if removals correctly handles case of intermixed as, and bs
    {
        hkPseudoRandomGenerator rand(0x1321);

        hkMultiMap<hkUlong, hkUlong, hkMultiMapIntegralOperations> map;

        for (int i = 0; i < 100; i++)
        {
            // We don't want to clash with invalid
            int a = int(rand.getRand32()) & 0x7fffffff;
            int b = a + 1;
            /*
            do
            {
                b = int(rand.getRand32()) & 0x7fffffff;
            } while (b == a); */

            int numAs = 0;
            int numBs = 0;

            for (int j = 0; j < 100; j++)
            {
                if (rand.getRand32() & 8)
                {
                    map.insert(a, 1);
                    numAs ++;
                }
                else
                {
                    map.insert(b, 1);
                    numBs ++;
                }
            }

            while (map.getSize() > 0)
            {
                HK_TEST(map.findNumEntries(a) == numAs);
                HK_TEST(map.findNumEntries(b) == numBs);

                if (rand.getRand32() & 16)
                {
                    if (numAs > 0)
                    {
                        map.remove(a);
                        numAs--;
                    }
                }
                else
                {
                    if (numBs > 0)
                    {
                        map.remove(b);
                        numBs--;
                    }
                }
            }
        }
    }
}

int pointerMultiMap_main()
{
    pointerMultiMap_selfTest();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(pointerMultiMap_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
