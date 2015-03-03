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
#include <Common/Base/Container/MinHeap/hkMinHeap.h>

namespace // anonymous
{

struct Entry
{
	HK_FORCE_INLINE static hkBool32 lessThan(const Entry& a, const Entry& b) { return a.m_cost < b.m_cost; }
	HK_FORCE_INLINE static void swap(Entry& a, Entry& b) { hkAlgorithm::swap(a.m_index, b.m_index); hkAlgorithm::swap(a, b); }
	HK_FORCE_INLINE static void setIndex(const Entry& entry, int index) { entry.m_index = index; }
	HK_FORCE_INLINE static hkBool32 hasIndex(const Entry& a, int index) { return a.m_index == index; }

	Entry() {}
	Entry(hkReal cost):m_cost(cost) {}
	hkReal m_cost;
	mutable int m_index;
};

}

static void minHeap_selfTest()
{
	hkPseudoRandomGenerator rand(0x23423);

	{
		hkMinHeap<int> heap;

		HK_TEST( heap.isOk());
		heap.addEntry(10);
		HK_TEST( heap.isOk());
		heap.addEntry(20);
		HK_TEST( heap.isOk());
		heap.addEntry(15);
		HK_TEST( heap.isOk());
		heap.addEntry(5);
		HK_TEST( heap.isOk());
		heap.addEntry(25);
		HK_TEST( heap.isOk());

		heap.setEntry(2,1);
		HK_TEST( heap.isOk());

		heap.setEntry(1, 35);
		HK_TEST( heap.isOk());

		heap.removeEntry(2);
		HK_TEST(heap.isOk());
	}
	{
		hkMinHeap<Entry, Entry> heap;

		heap.addEntry(Entry(10));
		heap.addEntry(Entry(3));
		heap.addEntry(Entry(9));
		heap.addEntry(Entry(75));
		heap.addEntry(Entry(6));

		HK_TEST( heap.isOk());
		heap.removeEntry(2);

		HK_TEST( heap.isOk());
	}
}

int minHeap_main()
{
    minHeap_selfTest();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(minHeap_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
