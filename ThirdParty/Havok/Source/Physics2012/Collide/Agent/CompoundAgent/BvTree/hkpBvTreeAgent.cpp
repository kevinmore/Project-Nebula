/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.h>

#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#ifdef HK_PLATFORM_SPU	
#include <Physics2012/Collide/Filter/Spu/hkpSpuCollisionFilterUtil.h>
#endif

#include <Common/Base/Config/hkConfigVersion.h>
#if defined(HK_DEBUG) && (HAVOK_BUILD_NUMBER == 0)
	//#define BV_TREE_AGENT_DEBUG_DISPLAY
	#if defined(BV_TREE_AGENT_DEBUG_DISPLAY)
		#include <Common/Visualize/hkDebugDisplay.h>
		#include <Common/Base/Container/String/hkStringBuf.h>
	#endif
#endif

// Platform specific (SPU/PPU) macros
#if defined(HK_PLATFORM_SPU)
	typedef hkpSpuCollisionQueryDispatcher		CollisionDispatcher;
	#define GET_DISPATCHER(COLLISION_INPUT)		(COLLISION_INPUT).m_queryDispatcher	
	#define GET_CONTAINER(BV_TREE,BUFFER)		(BV_TREE)->getContainerImpl(BV_TREE, BUFFER)
	#define IS_COLLISION_ENABLED(COLLISION_INPUT,BODY_A,BODY_B,CONTAINER_B,KEY_B) \
		hkpSpuCollisionFilterUtil::s_shapeContainerIsCollisionEnabled((COLLISION_INPUT).m_filter, COLLISION_INPUT, BODY_A, BODY_B, CONTAINER_B, KEY_B)	
#else	
	typedef hkpCollisionDispatcher				CollisionDispatcher;
	#define GET_DISPATCHER(COLLISION_INPUT)		(COLLISION_INPUT).m_dispatcher	
	#define GET_CONTAINER(BV_TREE,BUFFER)		(BV_TREE)->getContainer()
	#define IS_COLLISION_ENABLED(COLLISION_INPUT,BODY_A,BODY_B,CONTAINER_B,KEY_B) \
		(COLLISION_INPUT).m_filter->isCollisionEnabled(COLLISION_INPUT, BODY_A, BODY_B, CONTAINER_B, KEY_B)	
#endif

#ifndef HK_PLATFORM_SPU

#include <Physics2012/Collide/Dispatch/hkpAgentDispatchUtil.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.inl>

hkpBvTreeAgent::hkpBvTreeAgent( hkpContactMgr* mgr )
:	hkpCollisionAgent( mgr )
{
	m_cachedAabb.m_max.setXYZ(hkSimdReal_Max);
	m_cachedAabb.m_min.setXYZ(hkSimdReal_Max);
}

hkBool hkpBvTreeAgent::m_useFastUpdate = false;
hkBool hkpBvTreeAgent::m_useAabbCaching = true;

hkBool HK_CALL hkpBvTreeAgent::getUseAabbCaching()
{
	return m_useAabbCaching;
}

void HK_CALL hkpBvTreeAgent::setUseAabbCaching( hkBool useIt )
{
	m_useAabbCaching = useIt;
}

void HK_CALL hkpBvTreeAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createBvTreeShapeAgent;
		af.m_getPenetrationsFunc = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpBvTreeAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_TREE, hkcdShapeType::ALL_SHAPE_TYPES );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE, hkcdShapeType::ALL_SHAPE_TYPES);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createShapeBvAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::BV_TREE );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createBvBvAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_TREE, hkcdShapeType::BV_TREE );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE);
	}
}


hkpCollisionAgent* HK_CALL hkpBvTreeAgent::createBvTreeShapeAgent(	const hkpCdBody& bodyA, 	const hkpCdBody& bodyB,
																	const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkpBvTreeAgent* agent = new hkpSymmetricAgent<hkpBvTreeAgent>( mgr );
	return agent;
}


hkpCollisionAgent* HK_CALL hkpBvTreeAgent::createShapeBvAgent(	const hkpCdBody& bodyA, const hkpCdBody& bodyB,
																	const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkpBvTreeAgent* agent = new hkpBvTreeAgent( mgr );
	return agent;
}



hkpCollisionAgent* HK_CALL hkpBvTreeAgent::createBvBvAgent(	const hkpCdBody& bodyA, const hkpCdBody& bodyB,
															const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkSimdReal radiusA; radiusA.load<1>(&bodyA.getMotionState()->m_objectRadius);
	hkSimdReal radiusB; radiusB.load<1>(&bodyB.getMotionState()->m_objectRadius);

		// we should call getAabb only on the smaller MOPP tree, or
		// we risk to tall getAabb on a big landscape.
		// so if radiusA is smaller than radiusB it is allowed
		// to call bodyA->getAabb(). So we want to collide bodyA with MOPP of bodyB
	if ( radiusA.isLess(radiusB) )
	{
		hkpBvTreeAgent* agent = new hkpBvTreeAgent( mgr );
		return agent;
	}
	else
	{
		hkpBvTreeAgent* agent = new hkpSymmetricAgent<hkpBvTreeAgent>( mgr );
		return agent;
	}
}



void hkpBvTreeAgent::cleanup( hkCollisionConstraintOwner& info )
{
	hkArray<hkpBvAgentEntryInfo>::iterator itr = m_collisionPartners.begin();
	hkArray<hkpBvAgentEntryInfo>::iterator end = m_collisionPartners.end();

	while ( itr != end )
	{
		if (itr->m_collisionAgent != HK_NULL)
		{
			itr->m_collisionAgent->cleanup(info);
		}
		itr++;
	}

	delete this;
}

void hkpBvTreeAgent::invalidateTim( const hkpCollisionInput& input )
{
	hkArray<hkpBvAgentEntryInfo>::iterator itr = m_collisionPartners.begin();
	hkArray<hkpBvAgentEntryInfo>::iterator end = m_collisionPartners.end();

	while ( itr != end )
	{
		if (itr->m_collisionAgent != HK_NULL)
		{
			itr->m_collisionAgent->invalidateTim(input);
		}
		itr++;
	}
}

void hkpBvTreeAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	hkArray<hkpBvAgentEntryInfo>::iterator itr = m_collisionPartners.begin();
	hkArray<hkpBvAgentEntryInfo>::iterator end = m_collisionPartners.end();

	while ( itr != end )
	{
		if (itr->m_collisionAgent != HK_NULL)
		{
			itr->m_collisionAgent->warpTime( oldTime, newTime, input );
		}
		itr++;
	}
}

	// A helper class to use the hkpAgentDispatchUtil
class hkAgentDispatchUtilHelper
{
	public:

		hkAgentDispatchUtilHelper( const hkpCdBody& body )
			: m_bodyB( &body )
		{
		}

		const hkpShapeContainer* m_container;
		hkpCdBody m_bodyB;

			// The following alignment command is required on the PlayStation(R)2 SN compiler.
			// Otherwise m_shapeBuffer is not 16-byte aligned,
			// although the definition of hkpShapeBuffer specifies it.
		HK_ALIGN16( hkpShapeBuffer m_shapeBuffer );

		inline const hkpCdBody* getBodyA( const hkpCdBody& cIn, const hkpCollisionInput& input, hkpShapeKey key)
		{
			return &cIn;
		}

		inline const hkpCdBody* getBodyB( const hkpCdBody& cIn, const hkpCollisionInput& input, hkpShapeKey key )
		{
			m_bodyB.setShape( m_container->getChildShape( key, m_shapeBuffer ), key);
			return &m_bodyB;
		}

		inline const hkpShapeContainer* getShapeContainerB( )
		{
			return m_container;
		}
};

hkpCollisionAgent* HK_CALL hkpBvTreeAgent::defaultAgentCreate( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
{
	return input.m_dispatcher->getNewCollisionAgent( bodyA, bodyB, input, mgr );
}

void hkpBvTreeAgent::prepareCollisionPartnersProcess( const hkpCdBody& bodyA,	const hkpCdBody& bodyB,	const hkpProcessCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	hkResult prepareSuccessfull;
	hkInplaceArray<hkpShapeKey,128> hitList;
	{
		hkTransform bTa;
		{
			const hkTransform& wTb = bodyB.getTransform();
			const hkTransform& wTa = bodyA.getTransform();
			bTa.setMulInverseMul( wTb, wTa );
		}

		hkVector4 linearTimInfo;
		{
			const hkMotionState* msA = bodyA.getMotionState();
			const hkMotionState* msB = bodyB.getMotionState();
			hkSweptTransformUtil::calcTimInfo( *msA, *msB, input.m_stepInfo.m_deltaTime, linearTimInfo);
		}
		hkAabb* cachedAabb = (m_useAabbCaching)?&m_cachedAabb:HK_NULL;

		prepareSuccessfull = hkpBvTreeAgent::calcAabbAndQueryTree( bodyA, bodyB, bTa, linearTimInfo, input, cachedAabb, hitList);
	}

	if ( prepareSuccessfull != HK_SUCCESS )
	{
		return;
	}

	//
	//	update the m_collisionPartners
	//
	{
		const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );
		const hkpShapeContainer* shapeContainer = bvB->getContainer();
		hkAgentDispatchUtilHelper helper(bodyB);
		helper.m_container = shapeContainer;

		if(m_useFastUpdate)
		{
			hkpAgentDispatchUtil<hkpShapeKey, hkpBvAgentEntryInfo, hkAgentDispatchUtilHelper>
				::fastUpdate( m_collisionPartners, hitList, bodyA, bodyB, input, m_contactMgr, constraintOwner, helper );
		}
		else
		{
			hkSort( hitList.begin(), hitList.getSize() );
			hkpAgentDispatchUtil<hkpShapeKey, hkpBvAgentEntryInfo, hkAgentDispatchUtilHelper>
				::update( m_collisionPartners, hitList, bodyA, bodyB, input, m_contactMgr, constraintOwner, helper );
		}
	}
	// do a little checking
#if defined(HK_DEBUG)
	{
		for (int i = 0; i < hitList.getSize(); i++ )
		{
			const hkpShapeKey& key = hitList[i];
			if ( ! ( key == m_collisionPartners[i].getKey() ) )
			{
				HK_ASSERT2(0x2e8d58bd,  0, "Internal consistency problem, probably a compiler error when havok libs where build" );
			}
		}
	}
#endif

}

void hkpBvTreeAgent::updateShapeCollectionFilter( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );
	const hkpShapeContainer* shapeContainer = bvB->getContainer();

	hkpShapeBuffer shapeBuffer;

	for ( int i = 0; i < m_collisionPartners.getSize(); ++i )
	{
		const hkpShape* shape = shapeContainer->getChildShape( m_collisionPartners[i].getKey(), shapeBuffer );
		hkpCdBody modifiedBodyB( &bodyB );
		modifiedBodyB.setShape( shape, m_collisionPartners[i].getKey() );

		if ( input.m_filter->isCollisionEnabled( input, bodyA, bodyB, *shapeContainer, m_collisionPartners[i].getKey() ) )
		{
			if ( m_collisionPartners[i].m_collisionAgent == hkpNullAgent::getNullAgent() )
			{
				// Shape which was previously filtered is now not filtered. Create the agent
				m_collisionPartners[i].m_collisionAgent = input.m_dispatcher->getNewCollisionAgent( bodyA, modifiedBodyB, input, this->m_contactMgr );
			}
			else
			{
				// Shape was not previously filtered and is still not filtered
				m_collisionPartners[i].m_collisionAgent->updateShapeCollectionFilter( bodyA, modifiedBodyB, input, constraintOwner );
			}
		}
		else
		{
			// Shape is now filtered. If it was previously filtered do nothing. Check if it was not previously filtered.
			if ( m_collisionPartners[i].m_collisionAgent != hkpNullAgent::getNullAgent() )
			{
				// Shape has just been filtered out. Delete the agent.
				m_collisionPartners[i].m_collisionAgent->cleanup( constraintOwner );
				m_collisionPartners[i].m_collisionAgent = hkpNullAgent::getNullAgent();
			}
		}
	}
}



void hkpBvTreeAgent::processCollision(	const hkpCdBody& bodyA, const hkpCdBody& bodyB,
										const hkpProcessCollisionInput& input,
										hkpProcessCollisionOutput& result )
{
	HK_ASSERT2(0x352618d8,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN_LIST( "BvTree", "QueryTree" );
	prepareCollisionPartnersProcess( bodyA , bodyB , input, *result.m_constraintOwner.val() );

	//
	// recursively process Collisions
	//
	hkpShapeBuffer shapeBuffer;

	hkArray<hkpBvAgentEntryInfo>::iterator itr = m_collisionPartners.begin();
	hkArray<hkpBvAgentEntryInfo>::iterator end = m_collisionPartners.end();

	hkpCdBody modifiedBodyB( &bodyB );

	HK_TIMER_SPLIT_LIST("NarrowPhase");
	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>(bodyB.getShape());
	const hkpShapeContainer* shapeContainer = bvB->getContainer();
	while ( itr != end )
	{
		const hkpShape* shape = shapeContainer->getChildShape( itr->m_key, shapeBuffer );
		modifiedBodyB.setShape( shape, itr->m_key );

		if(0)
		{
			const hkpTriangleShape* t = static_cast<const hkpTriangleShape*>(shape);

			hkVector4 offset = bodyB.getTransform().getTranslation();
			hkVector4 a; a.setAdd(t->getVertex<0>(), offset );
			hkVector4 b; b.setAdd(t->getVertex<1>(), offset );
			hkVector4 c; c.setAdd(t->getVertex<2>(), offset );

			hkVector4 center; center.setAdd( a, b);
			center.add( c);
			center.mul( hkSimdReal::getConstant<HK_QUADREAL_INV_3>() );


			//HK_DISPLAY_LINE( a, b, hkColor::YELLOW );
			//HK_DISPLAY_LINE( a, c, hkColor::YELLOW );
			//HK_DISPLAY_LINE( b, c, hkColor::YELLOW );
			//HK_DISPLAY_LINE( center, a, hkColor::YELLOW );
			//HK_DISPLAY_LINE( center, b, hkColor::YELLOW );
			//HK_DISPLAY_LINE( center, c, hkColor::YELLOW );
		}

		itr->m_collisionAgent->processCollision( bodyA, modifiedBodyB, input, result );
		itr++;
	}
	//HK_ON_DEBUG( hkprintf("Memory %i\n", m_collisionPartners.getSize() * ( 512 + sizeof(hkpBvAgentEntryInfo) ) ) );
	HK_TIMER_END_LIST();
}



void hkpBvTreeAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	hkpBvTreeAgent::staticLinearCast( bodyA, bodyB, input, collector, startCollector );
}

void hkpBvTreeAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_TIMER_BEGIN_LIST( "BvTree", "QueryTree" );

	//
	//	Get the AABB
	//
	hkAabb aabb;
	calcAabbLinearCast( bodyA, bodyB, input, aabb );

	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );

	//
	// query the BvTreeShape
	//
	hkInplaceArray<hkpShapeKey,128> hitList;
	{
		bvB->queryAabb( aabb, hitList );
	}

	//
	// recursively collect linearCast results and Contact Points
	//

	HK_TIMER_SPLIT_LIST( "NarrowPhase" );

	{
		hkpShapeType typeA = bodyA.getShape()->getType();

		hkArray<hkpShapeKey>::iterator itr = hitList.begin();
		hkArray<hkpShapeKey>::iterator end = hitList.end();

		hkpCdBody modifiedBodyB( &bodyB );

		hkpShapeBuffer shapeBuffer;
		const hkpShapeContainer* shapeContainer = bvB->getContainer();

		while ( itr != end )
		{
			if ( input.m_filter->isCollisionEnabled( input, bodyA, bodyB, *shapeContainer , *itr ) )
			{
			        const hkpShape* shape = shapeContainer->getChildShape( *itr, shapeBuffer );
			        modifiedBodyB.setShape( shape, *itr );
			        hkpShapeType typeB = shape->getType();
			        hkpCollisionDispatcher::LinearCastFunc linCastFunc = input.m_dispatcher->getLinearCastFunc( typeA, typeB );
			        linCastFunc( bodyA, modifiedBodyB, input, collector, startCollector );
			}
			itr++;
		}
	}

	HK_TIMER_END_LIST();
}

void hkpBvTreeAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	hkpBvTreeAgent::staticGetPenetrations( bodyA, bodyB, input, collector);
}

void hkpBvTreeAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "BvTree", "QueryTree" );

	//
	//	Get the AABB
	//
	hkAabb aabb;
	staticCalcAabb( bodyA, bodyB, input, aabb );

	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>( bodyB.getShape() );

	//
	// query the BvTreeShape
	//
	hkInplaceArray<hkpShapeKey,128> hitList;
	{
		bvB->queryAabb( aabb, hitList );
	}
	HK_TIMER_SPLIT_LIST( "NarrowPhase" );

	//
	// recursively check penetrations
	//

	{
		hkpShapeType typeA = bodyA.getShape()->getType();

		hkArray<hkpShapeKey>::iterator itr = hitList.begin();
		hkArray<hkpShapeKey>::iterator end = hitList.end();

		hkpCdBody modifiedBodyB( &bodyB );

		hkpShapeBuffer shapeBuffer;
		const hkpShapeContainer* shapeContainer = bvB->getContainer();

		while ( itr != end )
		{
			if ( input.m_filter->isCollisionEnabled( input, bodyA, bodyB, *shapeContainer , *itr ) )
			{
			    const hkpShape* shape = shapeContainer->getChildShape( *itr, shapeBuffer );
			    modifiedBodyB.setShape( shape, *itr );
			    hkpShapeType typeB = shape->getType();
			    hkpCollisionDispatcher::GetPenetrationsFunc getPenetrationsFunc = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );

			    getPenetrationsFunc( bodyA, modifiedBodyB, input, collector );

			    if (collector.getEarlyOut() )
			    {
			 	    break;
				}
			}
			itr++;
		}
	}

	HK_TIMER_END_LIST();
}

#endif // #ifndef HK_PLATFORM_SPU


void hkpBvTreeAgent::getClosestPoints(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, 
									  hkpCdPointCollector& collector)
{
	hkpBvTreeAgent::staticGetClosestPoints(bodyA, bodyB, input, collector);
}

void hkpBvTreeAgent::staticGetClosestPoints(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
											const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN_LIST("BvTree", "QueryTree");

	// Get the AABB of A in B space
	hkAabb aabbAinB;	
	staticCalcAabb(bodyA, bodyB, input, aabbAinB);

	// Allocate space in the stack to collect aabb query results	
	int keysSize = HK_MAX_NUM_HITS_PER_AABB_QUERY;
	hkpShapeKey* keys = hkAllocateStack<hkpShapeKey>(keysSize, "ShapeKeys"); 
	HK_SPU_STACK_POINTER_CHECK();

	// Run the query and finish if there are no results
	const hkpBvTreeShape* bvTree = static_cast<const hkpBvTreeShape*>(bodyB.getShape());
	int numKeys = bvTree->queryAabb(aabbAinB, keys, HK_MAX_NUM_HITS_PER_AABB_QUERY);	
	if (numKeys == 0)
	{
		hkDeallocateStack(keys, keysSize);
		HK_TIMER_END_LIST();
		return;
	}	

	// Shrink stack if possible	
	if (numKeys < HK_MAX_NUM_HITS_PER_AABB_QUERY)
	{
		keysSize = hkShrinkAllocatedStack(keys, numKeys) ? numKeys : HK_MAX_NUM_HITS_PER_AABB_QUERY;
	}
	else
	{
		numKeys = HK_MAX_NUM_HITS_PER_AABB_QUERY;
	}

	HK_TIMER_SPLIT_LIST("NarrowPhase");

	// Obtain shape container
	HK_ON_SPU( hkpShapeBuffer containerBuffer; )	
	const HK_SHAPE_CONTAINER* containerB = GET_CONTAINER(bvTree, containerBuffer);

	// Iterate over the child shapes	
	hkpShapeBuffer shapeBufferB;
	hkpCdBody childBodyB(&bodyB);
	const hkcdShape::ShapeType typeA = bodyA.getShape()->getType();
	for (int i = 0; i < numKeys; i++)
	{
		// Get candidate shape keys
		const hkpShapeKey keyB = keys[i];

		// Check if the collision is enabled
		if (!input.m_filter || !IS_COLLISION_ENABLED(input, bodyA, bodyB, *containerB, keyB))
		{
			continue;
		}

		// Get child shape
		const hkpShape* childShapeB = containerB->getChildShape(keyB, shapeBufferB);
		childBodyB.setShape(childShapeB, keyB);

		// Calculate closest points
		CollisionDispatcher::GetClosestPointsFunc closestPointsFunc = GET_DISPATCHER(input)->getGetClosestPointsFunc(typeA, childShapeB->getType());
		closestPointsFunc(bodyA, childBodyB, input, collector);		
	}

	// Clean-up
	hkDeallocateStack(keys, keysSize);	

	HK_TIMER_END_LIST();
}

void hkpBvTreeAgent::calcAabbLinearCast(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkAabb& aabbOut)
{
	// compose transform
	hkTransform bTa;
	{
		const hkTransform& wTb = bodyB.getTransform();
		const hkTransform& wTa = bodyA.getTransform();
		bTa.setMulInverseMul( wTb, wTa );
	}

	bodyA.getShape()->getAabb( bTa, input.m_tolerance, aabbOut );

	//
	//	expand the AABB
	//
	hkVector4 pathB; pathB._setRotatedInverseDir( bodyB.getTransform().getRotation(), input.m_path );
	hkVector4 zero; zero.setZero();
	hkVector4 pathMin; pathMin.setMin( zero, pathB );
	hkVector4 pathMax; pathMax.setMax( zero, pathB );
	aabbOut.m_min.add( pathMin );
	aabbOut.m_max.add( pathMax );
}

void hkpBvTreeAgent::staticCalcAabb(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkAabb& aabbOut)
{
	// compose transform
	hkTransform bTa;
	{
		const hkTransform& wTb = bodyB.getTransform();
		const hkTransform& wTa = bodyA.getTransform();
		bTa.setMulInverseMul( wTb, wTa );
	}

	bodyA.getShape()->getAabb( bTa, input.m_tolerance, aabbOut );
}


void hkpBvTreeAgent::LinearCastAabbCastCollector::addHit(hkpShapeKey key)
{
	const hkpBvTreeShape* m_bvTree = static_cast<const hkpBvTreeShape*>(m_bvTreeBody.getShape());

	// Check if the collision is enabled	
#if !defined(HK_PLATFORM_SPU)
	const hkpShapeContainer* container = m_bvTree->getContainer();
	if (!m_input.m_filter || !m_input.m_filter->isCollisionEnabled(m_input, m_castBody, m_bvTreeBody, *container, key))
#else			
	hkpShapeBuffer containerBuffer;
	const HK_SHAPE_CONTAINER* container = m_bvTree->getContainerImpl(m_bvTree, containerBuffer);
	if (!m_input.m_filter || !hkpSpuCollisionFilterUtil::s_shapeContainerIsCollisionEnabled(m_input.m_filter, m_input, m_castBody, m_bvTreeBody, *container, key))
#endif		
	{
		return;
	}

	// Obtain child shape
	hkpShapeBuffer shapeBuffer;
	const hkpShape* childShape = container->getChildShape(key, shapeBuffer);	
	hkpCdBody childBody(&m_bvTreeBody);
	childBody.setShape(childShape, key);

	// Linear cast body against the child shape
	CollisionDispatcher::LinearCastFunc linearCastFunction = GET_DISPATCHER(m_input)->getLinearCastFunc(m_castBody.getShape()->getType(), childShape->getType());
	linearCastFunction(m_castBody, childBody, m_input, m_castCollector, m_startCollector);
	hkSimdReal earlyOutDistance; earlyOutDistance.setFromFloat(m_castCollector.getEarlyOutDistance());		
	m_earlyOutFraction.setMin(m_earlyOutFraction, earlyOutDistance);	

	// Draw hit aabb
#if defined(BV_TREE_AGENT_DEBUG_DISPLAY) && defined(HK_DEBUG) && (HAVOK_BUILD_NUMBER == 0)
	{		
		hkAabb aabb; childShape->getAabb(m_bvTreeBody.getTransform(), 0, aabb);
		HK_DISPLAY_BOUNDING_BOX(aabb, hkColor::RED);		
		/*++m_numHits;
		hkStringBuf text; text.printf("%d", m_numHits);
		hkVector4 pos; aabb.getCenter(pos);
		HK_DISPLAY_3D_TEXT(text, pos, hkColor::WHITE);*/
	}
#endif
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
