/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Filter/hkpShapeCollectionFilter.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics2012/Collide/Agent3/List3/hkpListAgent3.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>

#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpMidphaseAgentData.h>
#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>

#define HK_THIS_AGENT_SIZE HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpMidphaseAgentData) )
HK_COMPILE_TIME_ASSERT(HK_THIS_AGENT_SIZE <= hkAgent3::MAX_NET_SIZE);

void hkListAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
{
	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = HK_NULL; 
	f.m_cleanupFunc  = HK_NULL;
#if !defined(HK_PLATFORM_SPU)
	f.m_removePointFunc  = HK_NULL;
	f.m_commitPotentialFunc  = HK_NULL;
	f.m_createZombieFunc  = HK_NULL;
	f.m_updateFilterFunc = updateFilter;
	f.m_invalidateTimFunc = invalidateTim;
	f.m_warpTimeFunc = warpTime;
#endif
	f.m_destroyFunc  = destroy;
	f.m_isPredictive = true;
}

#if !defined(HK_PLATFORM_SPU)
void hkListAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::LIST );
}
#endif

hkpAgentData* hkListAgent3::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	HK_ON_DEBUG( hkpShapeType t = input.m_bodyB->getShape()->getType() );

	HK_ASSERT2(0xf031ed46, input.m_input->m_dispatcher->hasAlternateType(t, hkcdShapeType::BV_TREE) || input.m_input->m_dispatcher->hasAlternateType(t, hkcdShapeType::COLLECTION), "The bodyB's shape must implement the hkpShapeCollection or hkpBvTreeShape interface.");

	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	HK_ON_DEBUG( midphaseAgentData->m_numShapeKeys = 0xcdcdcdcd );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	new (agent1nTrack) hkpAgent1nTrack();

	hkAgent1nMachine_Create( *agent1nTrack );

	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}


void  hkListAgent3::destroy ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;

	hkAgent1nMachine_Destroy( *agent1nTrack, dispatcher, mgr, constraintOwner );
}


hkpAgentData* hkListAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_TIMER_BEGIN("List3", this );

	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	
	HK_ASSERT(0xf031ed45, input.m_bodyB->getShape()->getType() == hkcdShapeType::LIST );
	const hkpListShape* lShapeB = static_cast<const hkpListShape*>(input.m_bodyB->getShape());
	

//	if (input.m_input->m_dispatcher->hasAlternateType( input.m_bodyA->getShape()->getType(), hkcdShapeType::CONVEX ) )
//	{
// 		int numShapeKeyPairs ;
// 		hkLocalBuffer<hkpShapeKeyPair> shapeKeyPairs(hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE); 
// 		{
// 
// 			int numKeysInB = lShapeB->getNumChildShapes();
// 			
// 			numKeysInB = hkMath::min2(numKeysInB, hkpListShape::MAX_CHILDREN_FOR_SPU_MIDPHASE);
// 
// 			hkLocalBuffer<hk1AxisSweep::AabbInt> boxesA(1 + 4);
// 			calculate the AABB, hmmm.
// 				boxesA[0].getShapeKey() = 0;
// 			fill in invalid entires at the end.
// 
// 				hkLocalBuffer<hk1AxisSweep::AabbInt> boxesB(numKeysInB + 4);
// 			extractCachedAabbsOrRecalculate(input.m_bodyB, *input.m_input, boxesB.begin(), numKeysInB);  place end markers 			for (int i = numHitsFirst; i < numHitsFirst + 4; i++)	{	aabbsFirst[i].m_min[0] = hkUint32(-1);	}
// 			hkSort(boxesB.begin(), numKeysInB, hk1AxisSweep::AabbInt::less);
// 
// 			numShapeKeyPairs = hk1AxisSweep::collide(boxesA.begin(), 1, boxesB.begin(), numKeysInB, shapeKeyPairs.begin(), HK_MAX_NUM_HITS_PER_AABB_QUERY);
// 
// 			HK_ASSERT2(0xad9755bd, numShapeKeyPairs <= HK_MAX_NUM_HITS_PER_AABB_QUERY, "Num hkpShapeKeyPairs exceeded");
// 
// 
// 		}
// 
// 		// combine the shapeKeyPairs list into a hitList
// 		hkpShapeKey* hitList = reinterpret_cast<hkpShapeKey*>(shapeKeyPairs.begin()); XXX MOVE BOTH OUTSIDE
// 		int idx = 0;
// 		for (int i = 0; i < numShapeKeyPairs; i++)
// 		{
// 			hkpShapeKey key = shapeKeyPairs[i].m_shapeKeyB;
// 			if (lShapeB->isChildEnabled(key))
// 			{
// 				hitList[idx++] = key;
// 			}
// 		}
// 		hitList[idx] = HK_INVALID_SHAPE_KEY;
//	}
//	else
//	{
		int size = lShapeB->m_childInfo.getSize();
		hkLocalBuffer<hkpShapeKey> hitList( size+1, "ListAgent3" );
		{
			int d = 0;
			for ( int i = 0; i < size; i++ )
			{	
				if ( lShapeB->hkpListShape::isChildEnabled(i) )	{	hitList[d++] = hkpShapeKey(i);	}
			}
			hitList[d] = HK_INVALID_SHAPE_KEY;
			size = d;
		}
//	}

	hkAgent1nMachine_Process( *agent1nTrack, input, lShapeB, hitList.begin(), output );
	
	HK_TIMER_END();
	
	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}


#if !defined(HK_PLATFORM_SPU)

void hkListAgent3::updateFilter( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	HK_ASSERT(0xf031ed45, input.m_dispatcher->hasAlternateType(bodyB.getShape()->getType(), hkcdShapeType::COLLECTION) );
	const hkpShapeContainer* shapeContainerB = bodyB.getShape()->getContainer();

	hkpAgent1nMachine_VisitorInput vin;
	vin.m_bodyA = &bodyA;
	vin.m_collectionBodyB = &bodyB;
	vin.m_input = &input;
	vin.m_contactMgr = mgr;
	vin.m_constraintOwner = &constraintOwner;
	vin.m_containerShapeB = shapeContainerB;

	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;

	hkAgent1nMachine_UpdateShapeCollectionFilter( *agent1nTrack, vin );
}

void hkListAgent3::invalidateTim( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCollisionInput& input )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkAgent1nMachine_InvalidateTim(*agent1nTrack, input);
}

void hkListAgent3::warpTime( hkpAgentEntry* entry, hkpAgentData* agentData, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkpMidphaseAgentData* midphaseAgentData = static_cast<hkpMidphaseAgentData*>( agentData );
	hkpAgent1nTrack* agent1nTrack = &midphaseAgentData->m_agent1nTrack;
	hkAgent1nMachine_WarpTime(*agent1nTrack, oldTime, newTime, input);
}

#endif

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
