/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTreeStream/hkpBvTreeStreamAgent.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Collide/Filter/hkpConvexListFilter.h>
#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>

#ifdef HK_DEBUG
//#	define HK_BV_TREE_DISPLAY_AABB
//#	define HK_DISPLAY_TRIANGLES
#endif

#if defined(HK_BV_TREE_DISPLAY_AABB) || defined( HK_DISPLAY_TRIANGLES )
#	include <Common/Visualize/hkDebugDisplay.h>
#endif

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvTreeAgent.inl>



hkpBvTreeStreamAgent::hkpBvTreeStreamAgent( const hkpCdBody& bodyA, 	const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
:	hkpCollisionAgent( mgr )
{
	m_dispatcher = input.m_dispatcher;
	m_cachedAabb.m_min.setZero();
	m_cachedAabb.m_max.setZero();
	hkAgent1nMachine_Create( m_agentTrack );
}

void HK_CALL hkpBvTreeStreamAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = createBvTreeShapeAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpSymmetricAgent<hkpBvTreeAgent>::staticLinearCast;
		af.m_isFlipped            = true;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_TREE, hkcdShapeType::CONVEX );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE, hkcdShapeType::CONVEX );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          =  createShapeBvAgent;
		af.m_getPenetrationsFunc  = hkpBvTreeAgent::staticGetPenetrations;
		af.m_getClosestPointFunc =  hkpBvTreeAgent::staticGetClosestPoints;
		af.m_linearCastFunc      =  hkpBvTreeAgent::staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX, hkcdShapeType::BV_TREE );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE );
	}
}

void HK_CALL hkpBvTreeStreamAgent::registerConvexListAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = dispatchBvTreeConvexList;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpSymmetricAgent<hkpBvTreeAgent>::staticLinearCast;
		af.m_isFlipped            = true;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_TREE, hkcdShapeType::CONVEX_LIST );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE, hkcdShapeType::CONVEX_LIST );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          =  dispatchConvexListBvTree;
		af.m_getPenetrationsFunc  = hkpBvTreeAgent::staticGetPenetrations;
		af.m_getClosestPointFunc =  hkpBvTreeAgent::staticGetClosestPoints;
		af.m_linearCastFunc      =  hkpBvTreeAgent::staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX_LIST, hkcdShapeType::BV_TREE );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX_LIST, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE );
	}
}

void HK_CALL hkpBvTreeStreamAgent::registerMultiRayAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = createBvTreeShapeAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpBvTreeAgent>::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpSymmetricAgent<hkpBvTreeAgent>::staticLinearCast;
		af.m_isFlipped            = true;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_TREE, hkcdShapeType::MULTI_RAY );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE, hkcdShapeType::MULTI_RAY );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          =  createShapeBvAgent;
		af.m_getPenetrationsFunc  = hkpBvTreeAgent::staticGetPenetrations;
		af.m_getClosestPointFunc =  hkpBvTreeAgent::staticGetClosestPoints;
		af.m_linearCastFunc      =  hkpBvTreeAgent::staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MULTI_RAY, hkcdShapeType::BV_TREE );
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MULTI_RAY, hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE );
	}
}

hkpCollisionAgent* HK_CALL hkpBvTreeStreamAgent::dispatchBvTreeConvexList(	const hkpCdBody& bodyA, 	const hkpCdBody& bodyB,
																		const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkpCollisionAgent* agent;
	if ( mgr )
	{
		hkpConvexListFilter::ConvexListCollisionType collisionType = input.m_convexListFilter->getConvexListCollisionType( bodyB, bodyA, input );
		switch( collisionType )
		{
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_NORMAL:
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_CONVEX:
			{
				// If we treat the object as a convex list (or convex), dispatch to the bvTree stream
				// in this case welding will work for triangles colliding with the outer hull of the object.
				agent = new hkpSymmetricAgent<hkpBvTreeStreamAgent>(bodyA, bodyB, input, mgr);
				break;
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_LIST:
			{
				// If we treat the object as a list shape, dispatch to the shape collection agent
				// (the convex list shape is treated as the shape collection)
				// In this case welding fully works
				agent = new hkpSymmetricAgent<hkpShapeCollectionAgent>(bodyA, bodyB, input, mgr);
				break;
			}
		default:
			{
				agent = HK_NULL;
				HK_ASSERT2(0xeaf09646, 0, "Unknown ConvexListCollisionType returned");
			}
		}
	}
	else
	{
		agent = new hkpSymmetricAgent<hkpBvTreeStreamAgent>( bodyA, bodyB, input, mgr );
	}
	return agent;
}

hkpCollisionAgent* HK_CALL hkpBvTreeStreamAgent::dispatchConvexListBvTree(	const hkpCdBody& bodyA, 	const hkpCdBody& bodyB,
																			const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkpCollisionAgent* agent;
	if ( mgr )
	{
		hkpConvexListFilter::ConvexListCollisionType collisionType = input.m_convexListFilter->getConvexListCollisionType( bodyA, bodyB, input );
		switch( collisionType )
		{
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_NORMAL:
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_CONVEX:
			{
				// If we treat the object as a convex list (or convex), dispatch to the bvTree stream
				// in this case welding will work for triangles colliding with the outer hull of the object.
				agent = new hkpBvTreeStreamAgent(bodyA, bodyB, input, mgr);
				break;
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_LIST:
			{
				// If we treat the object as a list shape, dispatch to the shape collection agent
				// (the convex list shape is treated as the shape collection)
				// In this case welding fully works
				agent = new hkpShapeCollectionAgent(bodyA, bodyB, input, mgr);
				break;
			}
		default:
			{
				agent = HK_NULL;
				HK_ASSERT2(0xeaf09646, 0, "Unknown ConvexListCollisionType returned");
			}
		}
	}
	else
	{
		agent = new hkpSymmetricAgent<hkpBvTreeStreamAgent>( bodyA, bodyB, input, mgr );
	}
	return agent;
}


hkpCollisionAgent* HK_CALL hkpBvTreeStreamAgent::createBvTreeShapeAgent(	const hkpCdBody& bodyA, 	const hkpCdBody& bodyB,
																	  const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkpBvTreeStreamAgent* agent = new hkpSymmetricAgent<hkpBvTreeStreamAgent>( bodyA, bodyB, input, mgr );
	return agent;
}


hkpCollisionAgent* HK_CALL hkpBvTreeStreamAgent::createShapeBvAgent(	const hkpCdBody& bodyA, const hkpCdBody& bodyB,
																  const hkpCollisionInput& input,	hkpContactMgr* mgr )
{
	hkpBvTreeStreamAgent* agent = new hkpBvTreeStreamAgent( bodyA, bodyB, input, mgr );
	return agent;
}

void hkpBvTreeStreamAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	hkpBvTreeAgent::staticGetPenetrations( bodyA, bodyB, input, collector);
}


void hkpBvTreeStreamAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	hkpBvTreeAgent::staticGetClosestPoints( bodyA, bodyB, input, collector );
}


void hkpBvTreeStreamAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	hkpBvTreeAgent::staticLinearCast( bodyA, bodyB, input, collector, startCollector );
}




void hkpBvTreeStreamAgent::cleanup( hkCollisionConstraintOwner& info )
{
	hkAgent1nMachine_Destroy( m_agentTrack, m_dispatcher, m_contactMgr, info );
	delete this;
}

#ifdef HK_DISPLAY_TRIANGLES
static inline void HK_CALL hkBvTreeStreamAgent_displayTriangle( const hkTransform& transform, const hkpShapeCollection* collection, hkpShapeKey key )
{
	hkpShapeBuffer shapeBuffer;

	const hkpShape* shape = collection->getChildShape( key, shapeBuffer );
	if ( shape->getType() != hkcdShapeType::TRIANGLE)
	{
		return;
	}

	const hkpTriangleShape* t = static_cast<const hkpTriangleShape*>(shape);

	hkVector4 a; a.setTransformedPos(transform, t->getVertex(0));
	hkVector4 b; b.setTransformedPos(transform, t->getVertex(1));
	hkVector4 c; c.setTransformedPos(transform, t->getVertex(2));

	hkVector4 center; center.setAdd4( a, b);
	center.add4( c);
	center.mul4( 1.0f/ 3.0f);


	HK_DISPLAY_LINE( a, b, hkColor::YELLOW );
	HK_DISPLAY_LINE( a, c, hkColor::YELLOW );
	HK_DISPLAY_LINE( b, c, hkColor::YELLOW );
}
#endif

void hkpBvTreeStreamAgent::processCollision(	const hkpCdBody& bodyA, const hkpCdBody& bodyB,
											const hkpProcessCollisionInput& input, 	hkpProcessCollisionOutput& output )
{
	HK_TIMER_BEGIN_LIST( "BvTree3", "QueryTree" );

	//
	//	Set the input structure
	//
	hkpAgent3ProcessInput in3;
	{
		in3.m_bodyA = &bodyA;
		in3.m_bodyB = &bodyB;
		in3.m_contactMgr = m_contactMgr;
		in3.m_input = &input;

		const hkMotionState* msA = bodyA.getMotionState();
		const hkMotionState* msB = bodyB.getMotionState();
		hkSweptTransformUtil::calcTimInfo( *msA, *msB, input.m_stepInfo.m_deltaTime, in3.m_linearTimInfo);
		in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());
	}

	hkInplaceArray<hkpShapeKey,128> hitList;
	hitList.pushBackUnchecked( HK_INVALID_SHAPE_KEY );

	hkResult prepareSuccessful;
	{
		hkTransform bTa;	bTa.setInverse( in3.m_aTb );
		prepareSuccessful = hkpBvTreeAgent::calcAabbAndQueryTree( bodyA, bodyB, bTa, in3.m_linearTimInfo, input, &m_cachedAabb, hitList);
	}
#ifdef HK_DISPLAY_TRIANGLES
	{
		for ( int i =1; i < hitList.getSize();i++ )
		{
			const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>(bodyB.getShape());
			const hkpShapeCollection* shapeCollection = bvB->getShapeCollection();
			hkBvTreeStreamAgent_displayTriangle( bodyB.getTransform(), shapeCollection, hitList[i] );
		}
	}
#endif

	HK_TIMER_SPLIT_LIST("Narrow");
	if ( prepareSuccessful == HK_SUCCESS )
	{
		int newMemNeeded = ((hitList.getSize() / 4) + 1 - m_agentTrack.m_sectors.getSize()) * hkpAgent1nSector::SECTOR_SIZE;

        if ( !hkHasMemoryAvailable(3, newMemNeeded) )
		{
			hkSetOutOfMemoryState( hkMemoryAllocator::MEMORY_STATE_OUT_OF_MEMORY );
			HK_TIMER_END_LIST();
			return;
		}

		hkSort( hitList.begin(), hitList.getSize() );

		HK_ASSERT2 (0xf0487345, hitList[ hitList.getSize()-1 ] == HK_INVALID_SHAPE_KEY,
								"Your result from queryAabb deleted the HK_INVALID_SHAPE_KEY entry" );
		const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>(bodyB.getShape());
		const hkpShapeContainer* shapeContainer = bvB->getContainer();
			hkAgent1nMachine_Process( m_agentTrack, in3, shapeContainer, hitList.begin(), output );
	}
	else
	{
		const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>(bodyB.getShape());
		const hkpShapeContainer* shapeContainer = bvB->getContainer();
		hkAgent1nMachine_Process( m_agentTrack, in3, shapeContainer, HK_NULL, output );
	}
	HK_TIMER_END_LIST();
}

void hkpBvTreeStreamAgent::updateShapeCollectionFilter( const hkpCdBody& bodyA, const hkpCdBody& collectionBodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
		// invalid cached
	m_cachedAabb.m_min.setZero();
	m_cachedAabb.m_max.setZero();

	const hkpBvTreeShape* bvB = static_cast<const hkpBvTreeShape*>(collectionBodyB.getShape());
	const hkpShapeContainer* shapeContainer = bvB->getContainer();

	hkpAgent1nMachine_VisitorInput vin;
	vin.m_bodyA = &bodyA;
	vin.m_collectionBodyB = &collectionBodyB;
	vin.m_input = &input;
	vin.m_contactMgr = m_contactMgr;
	vin.m_constraintOwner = &constraintOwner;
	vin.m_containerShapeB = shapeContainer;

	hkAgent1nMachine_UpdateShapeCollectionFilter( m_agentTrack, vin );
}


void hkpBvTreeStreamAgent::invalidateTim( const hkpCollisionInput& input)
{
	hkAgent1nMachine_InvalidateTim(m_agentTrack, input);
}

void hkpBvTreeStreamAgent::warpTime(hkTime oldTime, hkTime newTime, const hkpCollisionInput& input)
{
	hkAgent1nMachine_WarpTime(m_agentTrack, oldTime, newTime, input);
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
