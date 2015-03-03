/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#ifndef HK_COLLIDE2_AGENT_DISPATCH_UTIL_H
#define HK_COLLIDE2_AGENT_DISPATCH_UTIL_H

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>

	// note keys needs to be sorted
template<typename KEY, typename ENTRY, class HELPER>
class hkpAgentDispatchUtil
{
	public:
			// update the lists 
		static HK_FORCE_INLINE void HK_CALL update( hkArray<ENTRY>& entries, hkArray<KEY>& keys,
													const hkpCdBody& cA, const hkpCdBody& cB,
													const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, HELPER& base );

			// Same as update, but optimized for small lists. In only requires that the input always has the same sort order, but
			// does not require sorted input
		static HK_FORCE_INLINE void HK_CALL fastUpdate( hkArray<ENTRY>& entries, hkArray<KEY>& keys,
													const hkpCdBody& cA, const hkpCdBody& cB,
													const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, HELPER& base );
};




	// requirements:
	// entries are sorted
	// keys are sorted
	// ENTRY needs:								\n
	// class ENTRY {								\n
	//		KEY& getKey()							\n
	//		hkpCollisionAgent*	m_collisionAgent;	\n
	//	}											\n
	//
	// the < and == operators are defined for KEY		<br>
	// BASE has to implement inline hkpCdBody& getBodyA( hkpCollidable& cIn, hkpCollisionInput& input, const KEY& key );
	// BASE has to implement inline hkpCdBody& getBodyB( hkpCollidable& cIn, hkpCollisionInput& input, const KEY& key );
	// BASE has to implement inline hkpShapeCollection* getShapeCollectionB( );
	//

template<typename KEY, typename ENTRY, class HELPER>
void HK_CALL hkpAgentDispatchUtil<KEY,ENTRY,HELPER>::update( hkArray<ENTRY>& entries, hkArray<KEY>& keys,
														   const hkpCdBody& cA, const hkpCdBody& cB, 
														   const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, HELPER& base )
{
	typename hkArray<ENTRY>::iterator oldEntriesItr = entries.begin();
	typename hkArray<ENTRY>::iterator oldEntriesEnd = entries.end();

	typename hkArray<KEY>::iterator newKeysItr = keys.begin();
	typename hkArray<KEY>::iterator newKeysEnd = keys.end();

	hkLocalArray<ENTRY> newEntries( keys.getSize() );
	newEntries.setSize( keys.getSize() );

	const hkpShapeContainer* shapeContainer = base.getShapeContainerB();

	typename hkArray<ENTRY>::iterator newEntriesItr = newEntries.begin();

	while ( (oldEntriesItr != oldEntriesEnd) && (newKeysItr != newKeysEnd) )
	{
		if (  (*newKeysItr) == (*oldEntriesItr).getKey() )
		{
			// keep element
			*newEntriesItr = *oldEntriesItr;

			newEntriesItr++;
			oldEntriesItr++;
			newKeysItr++;
			continue;
		}

		if (  (*newKeysItr) < (*oldEntriesItr).getKey() )
		{
			// new element
			const hkpCdBody& bodyA = cA;
			const hkpCdBody& bodyB = cB;
			const hkpCdBody& modifiedB = *base.getBodyB( cB, input, *newKeysItr );

			if ( input.m_filter->isCollisionEnabled( input, bodyA, bodyB, *shapeContainer , *newKeysItr ) )
			{
				newEntriesItr->m_collisionAgent = input.m_dispatcher->getNewCollisionAgent( bodyA, modifiedB, input, mgr );
			}
			else
			{
				newEntriesItr->m_collisionAgent = hkpNullAgent::getNullAgent();
			}
			newEntriesItr->setKey( *newKeysItr );

			newEntriesItr++;
			newKeysItr++;
			continue;
		}

		{
			// delete element
			if (oldEntriesItr->m_collisionAgent != HK_NULL)
			{
				oldEntriesItr->m_collisionAgent->cleanup( constraintOwner );
			}

			oldEntriesItr++;
		}
	}

	// now, one of the lists is empty
	// check for elements to delete
	while ( oldEntriesItr != oldEntriesEnd )
	{
		if (oldEntriesItr->m_collisionAgent != HK_NULL)
		{
			oldEntriesItr->m_collisionAgent->cleanup( constraintOwner );
		}
		oldEntriesItr++;
	}

	// append the rest
	while ( newKeysItr != newKeysEnd ) 
	{
		const hkpCdBody& bodyA = cA;
		const hkpCdBody& bodyB = cB;
		const hkpCdBody& modifiedB = *base.getBodyB( cB, input, *newKeysItr );

		if ( input.m_filter->isCollisionEnabled( input, bodyA, bodyB, *shapeContainer , *newKeysItr ) )
		{
			newEntriesItr->m_collisionAgent = input.m_dispatcher->getNewCollisionAgent( bodyA, modifiedB, input, mgr );
		}
		else
		{
			newEntriesItr->m_collisionAgent = hkpNullAgent::getNullAgent();
		}
		newEntriesItr->setKey( *newKeysItr );

		newEntriesItr++;
		newKeysItr++;
	}


	// copy the results
	entries = newEntries;
}

template<typename KEY, typename ENTRY, class HELPER>
void HK_CALL hkpAgentDispatchUtil<KEY,ENTRY,HELPER>::fastUpdate( hkArray<ENTRY>& entries, hkArray<KEY>& hitList,
														   const hkpCdBody& cA, const hkpCdBody& cB, 
														   const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, HELPER& base )
{

		// go through existing list and check whether we can find each item in the hitlist
		{
			ENTRY* itr = entries.begin();
			ENTRY* end = entries.end();
			hkpShapeKey* lastHit = hitList.begin();
			hkpShapeKey* endHitList = hitList.end();

			for ( ;itr != end; itr++ )
			{
				if ( lastHit != endHitList &&  itr->getKey() == *lastHit )
				{
					lastHit++;
					continue;
				}
				// Search the entire array
				{
					for (lastHit = hitList.begin(); lastHit!= endHitList; lastHit++ )
					{
						if ( itr->getKey() == *lastHit )
						{
							lastHit++;
							goto hitFound;
						}
					}

				}
				// not found: remove
				itr->m_collisionAgent->cleanup( constraintOwner );
				entries.removeAtAndCopy( static_cast<int>( (hkUlong)(itr - entries.begin())) );
				itr--;
				end--;
hitFound:;
			}

		}
		//
		// Go through the hitlist and check whether we already have that agent
		//

		const hkpShapeContainer* shapeContainer = base.getShapeContainerB();
		{
			if ( hitList.getSize() != entries.getSize() )
			{
				ENTRY* lastHit = entries.begin();
				ENTRY* end = entries.end();
				hkpShapeKey* hitItr = hitList.begin();
				hkpShapeKey* endHitList = hitList.end();

				for ( ;hitItr != endHitList; hitItr++ )
				{
					if ( lastHit != end &&  lastHit->getKey() == *hitItr )
					{
						lastHit++;
						continue;
					}
					// new found: insert
					int index = static_cast<int>( (hkUlong)(hitItr - hitList.begin()) ); // 64 bit ptr64->ulong->int32
					lastHit = entries.expandAt( index,1  );

					const hkpCdBody& bodyA = cA;
					const hkpCdBody& bodyB = cB;
					const hkpCdBody& modifiedB = *base.getBodyB( cB, input, *hitItr );

					if ( input.m_filter->isCollisionEnabled( input, bodyA, bodyB, *shapeContainer , *hitItr ) )
					{
						lastHit->m_collisionAgent = input.m_dispatcher->getNewCollisionAgent( bodyA, modifiedB, input, mgr );
					}
					else
					{
						lastHit->m_collisionAgent = hkpNullAgent::getNullAgent(  );
					}
					lastHit->m_key = *hitItr;

					end = entries.end();
					lastHit++;
				}
			}
		}
}
#endif // HK_COLLIDE2_AGENT_DISPATCH_UTIL_H

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
