/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

hkAlgorithm::ListElement* hkAlgorithm::_sortList(hkAlgorithm::ListElement* headPtr)
{
	// Mergesort For Linked Lists. Algorithm described at
	// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html

	if( headPtr == HK_NULL )
	{
		return HK_NULL;
	}

	for( int sortSize = 1; true; sortSize *= 2 )
	{
		int numMerges = 0;

		ListElement* p = headPtr;
		ListElement preHead; preHead.next = HK_NULL;
		ListElement* tail = &preHead;

		while( p )
		{
			numMerges += 1;

			ListElement* q = p;
			int psize;
			for( psize = 0; psize < sortSize && q != HK_NULL; ++psize )
			{
				q = q->next;
			}
			int qsize = sortSize;

			// while current lists not empty
			while( psize>0 && qsize>0 && q != HK_NULL )
			{
				ListElement* next;
				if( p <= q )
				{
					next = p;
					p = p->next;
					psize -= 1;
				}
				else
				{
					next = q;
					q = q->next;
					qsize -= 1;
				}
				tail->next = next;
				tail = next;
			}
			// one or both lists empty
			while( psize > 0 )
			{
				tail->next = p;
				tail = p;
				p = p->next;
				psize -= 1;
			}
			while( qsize>0 && q != HK_NULL )
			{
				tail->next = q;
				tail = q;
				q = q->next;
				qsize -= 1;
			}
			p = q;
		}
		tail->next = HK_NULL;
		
		if( numMerges <= 1 )
		{
			return preHead.next;
		}
		else
		{
			headPtr = preHead.next;
		}
	}
}

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
