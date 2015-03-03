/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>
#include <Physics2012/Collide/Filter/hkpShapeCollectionFilter.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics2012/Collide/Agent3/ConvexList3/hkpConvexListAgent3.h>

#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nTrack.h>
#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifold.h>

#include <Physics2012/Collide/Agent/Deprecated/ConvexList/hkpConvexListUtils.h>

#if defined (HK_DEBUG)
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>
#endif

///////////////////////////////////////////////////////////////////////////
//
// Done until here.
//
//////////////////////////////////////////////////////////////////////////




void hkConvexListAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
{
	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = sepNormal;
	f.m_cleanupFunc  = cleanup;
#if !defined(HK_PLATFORM_SPU)
	f.m_removePointFunc  = removePoint;
	f.m_commitPotentialFunc  = commitPotential;
	f.m_createZombieFunc  = createZombie;
	f.m_updateFilterFunc = updateFilter;
	f.m_invalidateTimFunc = invalidateTim;
	f.m_warpTimeFunc = warpTime;
#endif
	f.m_destroyFunc  = destroy;
	f.m_isPredictive = true; 
}

#if !defined(HK_PLATFORM_SPU)
void hkConvexListAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f, hkcdShapeType::CONVEX, hkcdShapeType::CONVEX_LIST );
	dispatcher->registerAgent3( f, hkcdShapeType::CONVEX_LIST, hkcdShapeType::CONVEX_LIST );
}
#endif

struct hkConvexListAgent3Data
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_COLLIDE, hkConvexListAgent3Data);

	hkInt16		m_inStreamModeCounter;
	hkpAgent1nTrack  m_agentTrack;

};

struct hkPredGskAgent3Data
{
	hkpGskCache m_gskCache;
	hkpGskManifold m_gskManifold;
};


hkBool32 hkConvexListAgent3::isInGskMode(const hkpAgentData* agentData)
{
	return hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_CONVEX_LIST_IN_GSK_MODE);
}

// Helper functions

HK_FORCE_INLINE struct hkConvexListAgent3Data* HK_CALL hkConvexListAgent3::getConvexListaData(const hkpAgentEntry* entry, hkpAgentData* data)
{
	HK_ASSERT2(0xad6544aa, !isInGskMode(data), "Wrong mode.");
	return reinterpret_cast<hkConvexListAgent3Data*>( & reinterpret_cast<hkPredGskAgent3Data*>(data)->m_gskManifold );
}

HK_FORCE_INLINE struct hkPredGskAgent3Data* HK_CALL hkConvexListAgent3::getPredGskData(const hkpAgentEntry* entry, hkpAgentData* data)
{
	//HK_ASSERT2(0xad6544aa, isInGskMode(data), "Wrong mode.");
	return reinterpret_cast<hkPredGskAgent3Data*>( data );
}

HK_FORCE_INLINE void hkConvexListAgent3::switchToStreamMode( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkpProcessCollisionOutput& output )
{
	HK_ASSERT2(0xad8755da, isInGskMode(agentData), "Not in Gsk mode ?");

	hkPredGskAgent3::cleanup(entry, getPredGskData(entry, agentData), input.m_contactMgr, *output.m_constraintOwner.val() );

	hkPredGskAgent3::setGskFlagToFalse(agentData, hkpGskCache::GSK_FLAGS_CONVEX_LIST_IN_GSK_MODE);

	new (getConvexListaData(entry, agentData)) hkConvexListAgent3Data();
	hkAgent1nMachine_Create( getConvexListaData(entry, agentData)->m_agentTrack );

	getConvexListaData(entry, agentData)->m_inStreamModeCounter = 25;

}


HK_FORCE_INLINE void hkConvexListAgent3::switchToGskMode(const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkpProcessCollisionOutput& output)
{
	hkAgent1nMachine_Destroy( getConvexListaData(entry, agentData)->m_agentTrack, input.m_input->m_dispatcher, input.m_contactMgr, *output.m_constraintOwner.val() );

	hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_CONVEX_LIST_IN_GSK_MODE);

	getPredGskData(entry, agentData)->m_gskManifold.init();
}

hkpAgent1nTrack* hkConvexListAgent3::getAgent1nTrack(const hkpAgentEntry* entry, hkpAgentData* data)
{
	return &getConvexListaData(entry, data)->m_agentTrack;
}

const hkpAgent1nTrack* hkConvexListAgent3::getAgent1nTrack(const hkpAgentEntry* entry, const hkpAgentData* data)
{
	return &getConvexListaData(entry, const_cast<hkpAgentData*>(data))->m_agentTrack;
}


void hkConvexListAgent3::updateFilter( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	if ( ! isInGskMode(agentData))
	{
		HK_ASSERT(0xf031ed45, bodyB.getShape()->getType() == hkcdShapeType::CONVEX_LIST );
		const hkpConvexListShape* listShapeB = static_cast<const hkpConvexListShape*>(bodyB.getShape());
		const hkpShapeContainer* shapeContainerB = listShapeB->getContainer();

		hkpAgent1nMachine_VisitorInput vin;
		vin.m_bodyA = &bodyA;
		vin.m_collectionBodyB = &bodyB;
		vin.m_input = &input;
		vin.m_contactMgr = mgr;
		vin.m_constraintOwner = &constraintOwner;
		vin.m_containerShapeB = shapeContainerB;

		hkpAgent1nTrack& agent1nTrack = getConvexListaData(entry, agentData)->m_agentTrack;

		hkAgent1nMachine_UpdateShapeCollectionFilter( agent1nTrack, vin );
	}
}

void hkConvexListAgent3::invalidateTim( hkpAgentEntry* entry, hkpAgentData* agentData, const hkpCollisionInput& input )
{
	if ( ! isInGskMode(agentData))
	{
		hkpAgent1nTrack& agent1nTrack = getConvexListaData(entry, agentData)->m_agentTrack;
		hkAgent1nMachine_InvalidateTim(agent1nTrack, input);
	}
}

void hkConvexListAgent3::warpTime( hkpAgentEntry* entry, hkpAgentData* agentData, hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	if ( ! isInGskMode(agentData))
	{
		hkpAgent1nTrack& agent1nTrack = getConvexListaData(entry, agentData)->m_agentTrack;
		hkAgent1nMachine_WarpTime(agent1nTrack, oldTime, newTime, input);
	}
}


hkpAgentData* hkConvexListAgent3::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	HK_ASSERT(0xf031ed46, input.m_bodyB->getShape()->getType() == hkcdShapeType::CONVEX_LIST );

	hkpAgentData* returnValue = hkPredGskAgent3::create(input, entry, agentData);

	hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_CONVEX_LIST_IN_GSK_MODE);
	hkPredGskAgent3::setGskFlagToFalse(agentData, hkpGskCache::GSK_FLAGS_PROCESS_FUNCTION_CALLED);

	return returnValue;
}


void  hkConvexListAgent3::destroy ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher )
{
	if ( isInGskMode(agentData) )
	{
		hkPredGskAgent3::destroy(entry, getPredGskData(entry, agentData), mgr, constraintOwner, dispatcher);
	}
	else
	{
		hkAgent1nMachine_Destroy( getConvexListaData(entry, agentData)->m_agentTrack, dispatcher, mgr, constraintOwner );
	}
}


void hkConvexListAgent3::sepNormal( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4& separatingNormalOut )
{
	//const hkpConvexListShape* clist = static_cast<const hkpConvexListShape*>(input.m_bodyB->m_shape);
	//clist->get
	if (isInGskMode(agentData))
	{
		hkPredGskAgent3::sepNormal(input, entry, agentData, separatingNormalOut);
	}
	else
	{
		separatingNormalOut.setXYZ_W(hkVector4::getConstant<HK_QUADREAL_1000>(), hkSimdReal_Half * hkSimdReal_MinusMax);
	}
}

hkpAgentData* hkConvexListAgent3::cleanup( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	HK_ASSERT2(0xad7644dd, isInGskMode(agentData), "This can only happen in gsk-mode. Other wise the separating distance should be set to -0.5f*HK_REAL_MAX.");

	return hkPredGskAgent3::cleanup(entry, getPredGskData(entry, agentData), mgr, constraintOwner);
}



#if !defined(HK_PLATFORM_SPU)
void hkConvexListAgent3::removePoint ( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToRemove )
{
	if (isInGskMode(agentData))
	{
		hkPredGskAgent3::removePoint(entry, getPredGskData(entry, agentData), idToRemove);
	}
}

void hkConvexListAgent3::commitPotential( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToCommit )
{
	if (isInGskMode(agentData))
	{
		hkPredGskAgent3::commitPotential(entry, getPredGskData(entry, agentData), idToCommit);
	}
}

void hkConvexListAgent3::createZombie( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToConvert )
{
	if (isInGskMode(agentData))
	{
		hkPredGskAgent3::createZombie(entry, getPredGskData(entry, agentData), idToConvert);
	}
}
#endif



hkpAgentData* hkConvexListAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_ASSERT2(0x57213df1,  input.m_contactMgr.val(), HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN_LIST( "CvxLst", "Tim" );

	hkpAgentData* dataEnd = HK_NULL;
	//
	//	Get the relative linear movement (xyz) and the worst case angular movment (w)
	//
	HK_ASSERT(0xf031ed45, input.m_bodyB->getShape()->getType() == hkcdShapeType::CONVEX_LIST );
	const hkpConvexListShape* cls = reinterpret_cast<const hkpConvexListShape*>( input.m_bodyB->getShape() );
	hkVector4 timInfo;
	hkSweptTransformUtil::calcTimInfo( *input.m_bodyA->getMotionState(), *input.m_bodyB->getMotionState(), input.m_input->m_stepInfo.m_deltaTime, timInfo);

	// some values to undo the output
	hkpProcessCollisionOutputBackup outputBackup( output );

	if ( isInGskMode(agentData) )
	{
gskMode:
		// Wrap the contact manager in a version that will convert the points on the hull of the
		// convex list to points on the sub shapes
		hkpMapPointsToSubShapeContactMgr mappingMgr( input.m_contactMgr );

		hkpAgent3ProcessInput tmpInput = input;
		tmpInput.m_contactMgr = &mappingMgr;

		dataEnd = hkPredGskAgent3::process(tmpInput, entry, getPredGskData(entry, agentData), separatingNormal, output);
		tmpInput.m_contactMgr = mappingMgr.m_contactMgr;


		if (mappingMgr.m_invalidPointHit)
		{
			// assert no added TOIs or contact points - this is not always the case currently - see below
			//HK_ASSERT()
switchToStreamModeLabel:

			// XXX - This line is necessary because when addContactPoint is called from line 167 in gskAgentUtil, the INVALID return
			// seems to be ignored and the point added anyway causing an assert in the process contact - some artifact of the welding code I think.
			outputBackup.rollbackOutput( *input.m_bodyA.val(), *input.m_bodyB.val(), output, input.m_contactMgr );
			switchToStreamMode( input, entry, agentData, output );
			dataEnd = HK_NULL;

			goto streamMode;
		}

		//
		// If we get a penetration (which is supported by 1 piece) normally we want to use the outer hull to push it out.
		// However if we start in the penetrating case, we want to use penetrations with the inner pieces
		//
		if ( getPredGskData(entry, agentData)->m_gskManifold.m_numContactPoints)
		{
			if (!hkPredGskAgent3::getGskFlag(agentData, hkpGskCache::GSK_FLAGS_PROCESS_FUNCTION_CALLED))
			{
				hkSimdReal allowA; allowA.load<1>((const hkReal*)&input.m_bodyA->getRootCollidable()->m_allowedPenetrationDepth);
				hkSimdReal allowB; allowB.load<1>((const hkReal*)&input.m_bodyB->getRootCollidable()->m_allowedPenetrationDepth);
				hkSimdReal allowedPenetration; allowedPenetration.setMin(allowA,allowB); allowedPenetration.add(allowedPenetration);
				if ( separatingNormal->getW().isLess(-allowedPenetration) )
				{
					goto switchToStreamModeLabel;
				}
			}
		}

	}
	else
	{
streamMode:
		HK_TIMER_SPLIT_LIST( "Stream" );
		if ( getConvexListaData(entry, agentData)->m_inStreamModeCounter-- < 0)
		{
			getConvexListaData(entry, agentData)->m_inStreamModeCounter = 25;

			hkpGsk::GetClosesetPointInput gskInput;
			hkTransform aTb;	aTb.setMulInverseMul( input.m_bodyA->getTransform(), input.m_bodyB->getTransform());
			{
				gskInput.m_shapeA = static_cast<const hkpConvexShape*>(input.m_bodyA->getShape());
				gskInput.m_shapeB = static_cast<const hkpConvexShape*>(input.m_bodyB->getShape());
				gskInput.m_aTb = &aTb;
				gskInput.m_transformA = &input.m_bodyA->getTransform();
				gskInput.m_collisionTolerance = input.m_input->getTolerance();
			}

			hkVector4 pointOnB;
			if( hkpGsk::getClosestPoint( gskInput, getPredGskData(entry, agentData)->m_gskCache, *separatingNormal, pointOnB ) != HK_SUCCESS )
			{
				switchToGskMode( input, entry, agentData, output );
				outputBackup.rollbackOutput( *input.m_bodyA, *input.m_bodyB, output, input.m_contactMgr );
				goto gskMode;
			}
		}

		// Reset separating distance to disable TIM's
		separatingNormal->setW(hkSimdReal_Half * hkSimdReal_MinusMax);

		{
			int size = cls->m_childShapes.getSize();
			hkLocalBuffer<hkpShapeKey> hitList( size+1 );
			for ( int i = 0; i < size; i++ ){		hitList[i] = static_cast<hkUint32>(i);	}
			hitList[size] = HK_INVALID_SHAPE_KEY;

			hkAgent1nMachine_Process( getConvexListaData(entry, agentData)->m_agentTrack, input, cls, hitList.begin(), output );
		}

		dataEnd = hkAddByteOffset( getConvexListaData(entry, agentData), sizeof(hkConvexListAgent3Data) );
	}


	hkPredGskAgent3::setGskFlagToTrue(agentData, hkpGskCache::GSK_FLAGS_PROCESS_FUNCTION_CALLED);
	HK_TIMER_END_LIST();

	dataEnd = reinterpret_cast<hkpAgentData*>( HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, hkUlong(dataEnd) ) );
	return dataEnd;
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
