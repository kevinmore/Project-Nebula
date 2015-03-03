/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Types/hkHandle.h>
#include <Common/Base/Container/FreeListArray/hkFreeListArray.h>

namespace FreeListTest
{
	HK_DECLARE_HANDLE(ItemIndex, hkUint16, 0xFFFF);

	class FreeListItem
	{			
		public:

			FreeListItem() : m_isEmpty(false) { s_constructorCalls++; }
			~FreeListItem() { s_destructorCalls++; }
			static void resetCallCounters() { s_constructorCalls = 0; s_destructorCalls = 0; s_isEmptyCalls = 0; }

		public:

		  static int s_constructorCalls;
		  static int s_destructorCalls;
		  static int s_isEmptyCalls;

		  hkUint8 m_padding[4];
		  hkBool m_isEmpty;
	};

	struct FreeListOperations
	{			
		HK_FORCE_INLINE static void setEmpty(FreeListItem& element, hkUint32 next) 
		{ 
			element.m_isEmpty = true;
			(hkUint32&)element = next; 
		}
		
		HK_FORCE_INLINE static hkUint32 getNext(const FreeListItem& element) 
		{ 
			return (hkUint32&)element; 
		}

		HK_FORCE_INLINE static hkBool32 isEmpty(const FreeListItem& element) 
		{
			FreeListItem::s_isEmptyCalls++;
			return element.m_isEmpty;
		}
	};
}

int FreeListTest::FreeListItem::s_constructorCalls = 0;
int FreeListTest::FreeListItem::s_destructorCalls = 0;
int FreeListTest::FreeListItem::s_isEmptyCalls = 0;


int freeListArray_main()
{
	// Test construction and destruction of contained elements
	{	
		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2> freeList;		
		}
		HK_TEST((FreeListTest::FreeListItem::s_constructorCalls == 0) && (FreeListTest::FreeListItem::s_destructorCalls == 0));
		FreeListTest::FreeListItem::resetCallCounters();
		
		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2> freeList;
			freeList.allocate();		
		}
		HK_TEST((FreeListTest::FreeListItem::s_constructorCalls == 1) && (FreeListTest::FreeListItem::s_destructorCalls == 1));
		FreeListTest::FreeListItem::resetCallCounters();

		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2> freeList;
			freeList.allocate();
			freeList.allocate();
			freeList.allocate();
			freeList.release(FreeListTest::ItemIndex(0));
		}
		HK_TEST((FreeListTest::FreeListItem::s_constructorCalls == 3) && (FreeListTest::FreeListItem::s_destructorCalls == 3));
		FreeListTest::FreeListItem::resetCallCounters();
	}

	// Test use of isEmpty depending on the free list operations provided
	{
		// Use default operations (no isEmpty provided)
		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2> freeList;
			freeList.allocate();
			freeList.isAllocated(FreeListTest::ItemIndex(1));	
			HK_TEST(FreeListTest::FreeListItem::s_isEmptyCalls == 0);
		}
		HK_TEST(FreeListTest::FreeListItem::s_isEmptyCalls == 0);
		FreeListTest::FreeListItem::resetCallCounters();

		// Use custom operations structure with isEmpty
		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2, FreeListTest::FreeListOperations> freeList;
			freeList.allocate();
			freeList.isAllocated(FreeListTest::ItemIndex(1));
			
		}
		HK_TEST(FreeListTest::FreeListItem::s_isEmptyCalls == 3);
		FreeListTest::FreeListItem::resetCallCounters();		
	}

	// Test allocation with copy
	{
		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2> freeList;
			FreeListTest::ItemIndex first = freeList.allocate();
			freeList[first].m_padding[0] = 0;
			FreeListTest::ItemIndex second = freeList.allocate(freeList[first]);
			freeList[second].m_padding[0]++;

			HK_TEST(freeList[first].m_padding[0] == 0);
			HK_TEST(freeList[second].m_padding[0] == 1);
		}
		HK_TEST(FreeListTest::FreeListItem::s_constructorCalls == 1);
		HK_TEST(FreeListTest::FreeListItem::s_destructorCalls == 2);
		FreeListTest::FreeListItem::resetCallCounters();
	}

	// Test iterator (with isEmpty provided)
	{
		{
			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2, FreeListTest::FreeListOperations> freeList;
			freeList.allocate();
			freeList.allocate();
			freeList.allocate();
			freeList.release(FreeListTest::ItemIndex(0));

			hkFreeListArray<FreeListTest::FreeListItem, FreeListTest::ItemIndex, 2, FreeListTest::FreeListOperations>::Iterator it( freeList );
			int num = 0;
			for( ; it.isValid(); it.next() )
			{
				num++;
				HK_TEST( it.getIndex().value() != 0 );
			}
			HK_TEST( num == 2 );
		}
		FreeListTest::FreeListItem::resetCallCounters();
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(freeListArray_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
