/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseDispatcher.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandlePair.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpTypedBroadPhaseHandle.h>
#include <Physics2012/Collide/Dispatch/BroadPhase/hkpNullBroadPhaseListener.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>

#include <Common/Base/Container/PointerMap/hkMap.h>

hkpTypedBroadPhaseDispatcher::hkpTypedBroadPhaseDispatcher()
{
	for (int a = 0; a < HK_MAX_BROADPHASE_TYPE; a++ )
	{
		for (int b = 0; b < HK_MAX_BROADPHASE_TYPE; b++ )
		{
			m_broadPhaseListeners[a][b] = &m_nullBroadPhaseListener;
		}
	}
}

hkpTypedBroadPhaseDispatcher::~hkpTypedBroadPhaseDispatcher()
{
}

void hkpTypedBroadPhaseDispatcher::addPairs(	hkpTypedBroadPhaseHandlePair* newPairs, int numNewPairs, const hkpCollidableCollidableFilter* filter ) const
{
	while ( --numNewPairs >=0 )
	{
		hkpCollidable* collA = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(newPairs->m_a)->getOwner() );
		hkpCollidable* collB = static_cast<hkpCollidable*>( static_cast<hkpTypedBroadPhaseHandle*>(newPairs->m_b)->getOwner() );

		if( filter->isCollisionEnabled( *collA, *collB ) )
		{
			int typeA = newPairs->getElementA()->getType();
			int typeB = newPairs->getElementB()->getType();
			HK_ASSERT(0xf0ff0010, 0 <= typeA && typeA < HK_MAX_BROADPHASE_TYPE);
			HK_ASSERT(0xf0ff0011, 0 <= typeB && typeB < HK_MAX_BROADPHASE_TYPE);
			m_broadPhaseListeners[typeA][typeB]->addCollisionPair(*newPairs);
		}
		newPairs++;
	}
}

void hkpTypedBroadPhaseDispatcher::removePairs( hkpTypedBroadPhaseHandlePair* deletedPairs, int numDeletedPairs ) const
{
	while ( --numDeletedPairs >=0 )
	{
		int typeA = deletedPairs->getElementA()->getType();
		int typeB = deletedPairs->getElementB()->getType();
		HK_ASSERT(0xf0ff0012, 0 <= typeA && typeA < HK_MAX_BROADPHASE_TYPE);
		HK_ASSERT(0xf0ff0013, 0 <= typeB && typeB < HK_MAX_BROADPHASE_TYPE);
		m_broadPhaseListeners[typeA][typeB]->removeCollisionPair(*deletedPairs);
		deletedPairs++;
	}
}

// value = position << 32 + count

#define POSITION_FROM_VALUE(value)    (hkUint32(value) >> 8 )
#define COUNT_FROM_VALUE(value)        (0xff & int(value))
#define VALUE_FROM_POSITION_AND_COUNT(position, count) hkUint64( (position << 8) | count )

#if (HK_POINTER_SIZE == 8)
union hkKeyHandleUnion
{
	hkUint32 m_handleId[2];
	hkUint64 m_key;
};
#endif

static inline hkUint64 keyFromPair( const hkpBroadPhaseHandlePair& pair )
{
#if ( HK_POINTER_SIZE == 4 )
	return reinterpret_cast<const hkUint64&>(pair);
#elif ( HK_POINTER_SIZE == 8 )
	hkKeyHandleUnion tempUnion;
	// merge the m_id value from each hkpBroadPhaseHandle to form a single 64bit value ( pair.m_b->m_id | pair.m_a->m_id )
	// Ollie says this will cause an extra cache miss.
	tempUnion.m_handleId[0] = pair.m_a->m_id;
	tempUnion.m_handleId[1] = pair.m_b->m_id;
	return tempUnion.m_key;
#endif
}

void hkpTypedBroadPhaseDispatcher::removeDuplicates( hkArray<hkpBroadPhaseHandlePair>& newPairs, hkArray<hkpBroadPhaseHandlePair>& delPairs )
{
	hkToiPrintf("rem.dupl", "# removing duplicates: %dn %dd\n", newPairs.getSize(), delPairs.getSize());

	int min = hkMath::min2( newPairs.getSize(), delPairs.getSize() );

	if ( min < 32  ) 
	{
		int outD = 0;
		for (int d = 0; d < delPairs.getSize(); d++)
		{
			int n = 0;
			for (; n < newPairs.getSize(); n++)
			{
				if (		(newPairs[n].m_a == delPairs[d].m_a && newPairs[n].m_b == delPairs[d].m_b )
						||	(newPairs[n].m_b == delPairs[d].m_a && newPairs[n].m_a == delPairs[d].m_b ) )
				{
					break;
				}
			}

			if (n == newPairs.getSize())
			{
				// duplicate not found
				if (outD != d)
				{
					delPairs[outD] = delPairs[d];
				}
				outD++;
			}
			else
			{
				// remove the one duplicated element.
				// we have to keep the order, otherwise our array gets non deterministic
				newPairs.removeAtAndCopy(n); 
			}
		}
		delPairs.setSize(outD);
		return;
	}

	HK_COMPILE_TIME_ASSERT( sizeof(hkpBroadPhaseHandlePair) == 2*sizeof(void*) );
	{
		int bufferSizeBytes = hkMap<hkUint64>::getSizeInBytesFor(newPairs.getSize());
		hkArray<char>::Temp buffer(bufferSizeBytes);
		hkMap<hkUint64> newPairsMap( buffer.begin(), bufferSizeBytes );

		{
			for (int n = 0; n < newPairs.getSize(); n++)
			{
				hkpBroadPhaseHandlePair pair = newPairs[n];
				if (pair.m_a > pair.m_b)
				{
					hkAlgorithm::swap(pair.m_a, pair.m_b);
				}
				hkUint64 key = keyFromPair(pair);
				hkMap<hkUint64>::Iterator it = newPairsMap.findKey( key );
				if (newPairsMap.isValid(it))
				{
					hkInt64 value = newPairsMap.getValue(it);
					// increase count (lower hkInt16)
					value++;
					HK_ASSERT2(0xad000730, COUNT_FROM_VALUE(value) != 0, "Count overflow");
					newPairsMap.setValue(it, value);

					// note: we'd need to store the position of this doubled entry here.
					// but as we may assume that such a doubled entry will have a corresponding
					// deletedPair we mark it invalid in  the newPairsList straight away
					newPairs[n].m_a = HK_NULL;
				}
				else
				{
					hkInt64 value = VALUE_FROM_POSITION_AND_COUNT(n, 1);
					newPairsMap.insert( key, value );
				}
			}
		}

		{
			int outD = 0;
			for (int d = 0; d < delPairs.getSize(); d++)
			{
				hkpBroadPhaseHandlePair pair = delPairs[d];
				if (pair.m_a > pair.m_b)
				{
					hkAlgorithm::swap(pair.m_a, pair.m_b);
				}


				hkMap<hkUint64>::Iterator it = newPairsMap.findKey( keyFromPair(pair) );

				if (newPairsMap.isValid(it))
				{
					// remove both entries from the list
					//hkUint64 n = newPairsMap.getValue(it);
					//newPairsMap.remove(it);
					//newPairs[(int)n].m_a = HK_NULL;
					//delPairs.removeAt(d--);
					// 
					{
						hkUint64 value = newPairsMap.getValue(it);
						hkInt32 count = COUNT_FROM_VALUE(value);
						if (count > 1)
						{
							value--;
							newPairsMap.setValue(it, value);
						}
						else // count == 1
						{
							newPairsMap.remove(it);
							hkInt32 n = POSITION_FROM_VALUE(value);
							newPairs[n].m_a = HK_NULL;
						}
					}
				}
				else
				{
					delPairs[outD] = delPairs[d];
					outD++;
				}
			}
			delPairs.setSize(outD);
		}

		// Shrink back the newPairs list.
		{
			int nextNull = 0;
			for (int i=0 ; i < newPairs.getSize(); i++)
			{
				if (newPairs[i].m_a != HK_NULL)
				{
					newPairs[nextNull++] = newPairs[i];
				}
			}
			newPairs.setSize(nextNull);
		}
	}
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
