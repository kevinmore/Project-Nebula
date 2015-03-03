/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

static void local_array()
{
	// Testing hkLocalArray functionality
	{
		hkLocalArray<int> arr(5);
		HK_TEST(arr.getSize() == 0);
		HK_TEST(arr.getCapacity() != 0);
		HK_TEST(arr.isEmpty());
	}

	// Testing Local array's setSize functionality
	{
		hkLocalArray<int> arr(5);
		HK_TEST(arr.getCapacity() == 5);
		arr.setSize(4);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}
		HK_TEST(arr.getSize() == 4);
	}

	//Testing basic array's capacity and size functionality
	{
		hkLocalArray<int> arr(5);
		HK_TEST(arr.getCapacity() == 5);
		arr.setSize(3);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}
		HK_TEST(arr.getSize() == 3);
	}

	// Testing pushback and insertAt with 2 parameter.
	{
		hkLocalArray<int> arr(5);
		arr.setSize(3);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}
		arr.pushBack(3);
		HK_TEST(arr.getSize() == 4);

		arr.insertAt(0,5);
		HK_TEST(arr.getSize() == 5);
		HK_TEST(arr[0] == 5);

		for(int i = 0; i < 4; ++i)
		{
			HK_TEST(arr[i+1] == i);
		}
	}

	// Testing insertAt with 3 parameter.
	{
		hkLocalArray<int> arr(20);
		arr.setSize(10);
		int t[] = {7,9,2};
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}

		arr.insertAt(1,t,2);
		HK_TEST(arr.getSize() == 12);
		HK_TEST(arr[0] ==  0);
		HK_TEST(arr[1] ==  7);
		HK_TEST(arr[2] ==  9);
		HK_TEST(arr[3] ==  1);

	}

	// Testing back() functionality
	{
		hkLocalArray<int> arr(5);
		arr.setSize(5);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}
		HK_TEST(arr.back() == 4);
		arr.back()=7;
		HK_TEST(arr[4] == 7);
	}

	// Testing remove functions .
	{
		hkLocalArray<int> arr(10);
		arr.reserve(10);
		for(int i = 0; i < 10; ++i)
		{
			arr.pushBack(i);
		}
		// Testing popback functionality
		{
			arr.popBack();
			HK_TEST(arr.getSize() == 9);
			for(int i = 0; i < arr.getSize();++i)
			{
				HK_TEST(arr[i] == i);
			}
		}
		// Testing removeAt functionality
		{
			arr.removeAt(5);
			HK_TEST(arr.getSize() == 8);
			HK_TEST(arr[5]!=5);
			HK_TEST(arr[4] == 4);
			HK_TEST(arr[6] == 6);
		}
		// Testing removeAtandCopy functionality.
		{
			arr.clear();
			HK_TEST(arr.getSize() == 0);
			arr.reserveExactly(10);
			for(int i = 0; i < 10; ++i)
			{
				arr.pushBackUnchecked(i);
			}
			HK_TEST(arr.getSize() == 10);
			arr.removeAtAndCopy(4);
			HK_TEST(arr.getSize() == 9);
			HK_TEST(arr[3] == 3);
			HK_TEST(arr[4] == 5);
			HK_TEST(arr[5] == 6);
			HK_TEST(arr[8] == 9);
			HK_TEST(arr.indexOf(3) == 3);
			HK_TEST(arr.indexOf(4) == -1);
			HK_TEST(arr.indexOf(5) == 4);
		}
	}

	// Testing copy  functionality.
	 {
		const int size = 3;
		int srcarr[]={7,8,9};
		int dstarr[5];

		hkLocalArray<int>::copy(dstarr,srcarr,size);
		HK_TEST(dstarr[0]  ==  7);
		HK_TEST(dstarr[1]  ==  8);
		HK_TEST(dstarr[2]  ==  9);
	}

	// Testing  expandBy().
	{
		hkLocalArray<int> arr(5);
		arr.setSize(5);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}

		int* ref = arr.expandBy(5);

		HK_TEST(arr.wasReallocated());
		HK_TEST(arr.getCapacity() >= 10);
		HK_TEST(arr.getSize() == 10);
		HK_TEST(ref != HK_NULL);

		// Here only size get expanded. There is no shifting of values.
		int j=0;
		for(int i = 0; i < 5; ++i)
		{
			HK_TEST( arr[i] == j);
			++j;
		}
	}

	// Testing expandat().
	{
		hkLocalArray<int> arr(5);
		HK_TEST(arr.getCapacity() == 5);
		arr.setSize(5);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}

		int* ref = arr.expandAt(3,9);

		HK_TEST(arr.getSize() == 14);
		HK_TEST(arr.getCapacity() >= 14);
		HK_TEST(ref != HK_NULL);

	 }

	// Testing expandByUnchecked().
	{
		hkLocalArray<int> arr(20);
		arr.setSize(5);
		for(int i = 1; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}

		int* ref = arr.expandByUnchecked(10);
		HK_TEST(arr.getSize() == 15);
		HK_TEST(arr.getCapacity() == 20);
		HK_TEST(ref != HK_NULL);
	 }

	// Testing expandOne().
	{
		hkLocalArray<int> arr(20);
		arr.setSize(7);
		for(int i = 0; i < arr.getSize(); ++i)
		{
			arr[i] = i;
		}

		int& ref = arr.expandOne();
		ref = 44;
		HK_TEST(arr.getSize() == 8);
		HK_TEST(arr.getCapacity() == 20);
		HK_TEST(arr[7] == 44);
	 }

	//Testing setDataUserFree() Test Case 1 with Reduced Capacity & size.
	{
		int dstarr[]={7,8,9};
		hkLocalArray<int> arr(5);
		arr.setSize(5);
		for(int i = 0; i < arr.getSize();++i)
		{
				arr[i] = i;
		}
		arr.setDataUserFree( dstarr, 3, 3);

		HK_TEST(arr.getSize() == 3);
		HK_TEST(arr.getCapacity() == 3);

		for(int i = 0; i < arr.getSize(); ++i)
		{
			HK_TEST(arr[i] == dstarr[i]);
		}
	 }

	//Testing setDataUserFree() Test Case 2 with Extended Capacity & size.
	{
		int dstarr[]={7,8,9,10,11,12,13};
		hkLocalArray<int> arr(5);
		arr.setSize(5);
		for(int i = 0; i < arr.getSize();++i)
		{
			arr[i] = i;
		}
		arr.setDataUserFree( dstarr, 7, 7);

		HK_TEST(arr.getSize() == 7);
		HK_TEST(arr.getCapacity() == 7);

		for(int i = 0; i < arr.getSize(); ++i)
		{
			HK_TEST(arr[i] == dstarr[i]);
		}
	}
}

int localarray_main()
{
	local_array();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(localarray_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
