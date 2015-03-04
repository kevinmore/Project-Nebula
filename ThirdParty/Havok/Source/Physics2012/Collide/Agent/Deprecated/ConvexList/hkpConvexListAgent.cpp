/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>
#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>

#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Agent3/Machine/1n/hkpAgent1nMachine.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent/Deprecated/ConvexList/hkpConvexListAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/List/hkpListAgent.h>

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>

#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Internal/Collide/Gjk/Agent/hkpGskAgentUtil.h>
#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpFlagCdBodyPairCollector.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpSimpleClosestContactCollector.h>
#include <Physics2012/Collide/Agent/CompoundAgent/ShapeCollection/hkpShapeCollectionAgent.h>

#include <Physics2012/Collide/Filter/hkpConvexListFilter.h>

#include <Physics2012/Collide/Agent/Deprecated/ConvexList/hkpConvexListUtils.h>

#include <Common/Visualize/hkDebugDisplay.h>

extern hkReal hkConvexShapeDefaultRadius;

//HK_COMPILE_TIME_ASSERT( sizeof( hkpConvexListAgent ) == 12/*base*/ + 20/*tim*/ + 16/*cache*/ + 64/*manifold*/ );
//HK_COMPILE_TIME_ASSERT( sizeof( hkpConvexListAgent::StreamData) < sizeof(hkpGskManifold  ) );

hkpConvexListAgent::hkpConvexListAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
: hkpPredGskfAgent( bodyA, bodyB, input, mgr )
{

	m_dispatcher = input.m_dispatcher;
	m_inGskMode = true;
	m_processFunctionCalled = false;

	const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());
	hkTransform t; t.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform() );
	m_cache.init( shapeA, shapeB, t );


	m_separatingNormal(3) = -1.f;
	m_timeOfSeparatingNormal = hkTime(-1.0f);
}



hkpCollisionAgent* HK_CALL hkpConvexListAgent::createConvexConvexListAgent(const 	hkpCdBody& bodyA, const hkpCdBody& bodyB,
															 const  hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpCollisionAgent* agent;
	if ( mgr )
	{
		hkpConvexListFilter::ConvexListCollisionType collisionType = input.m_convexListFilter->getConvexListCollisionType( bodyB, bodyA, input );
		switch( collisionType )
		{
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_NORMAL:
			{
				agent = new hkpConvexListAgent( bodyA, bodyB, input, mgr );
				break;
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_CONVEX:
			{
				agent = new hkpPredGskfAgent( bodyA, bodyB, input, mgr );
				break;
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_LIST:
			{
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
		agent = new hkpSymmetricAgent<hkpShapeCollectionAgent>(bodyA, bodyB, input, mgr);
	}
	return agent;
}

hkpCollisionAgent* HK_CALL hkpConvexListAgent::createConvexListConvexAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB,
																	const  hkpCollisionInput& input, hkpContactMgr* mgr )
{
	hkpCollisionAgent* agent;
	if ( mgr )
	{
		hkpConvexListFilter::ConvexListCollisionType collisionType = input.m_convexListFilter->getConvexListCollisionType( bodyA, bodyB, input );
		switch( collisionType )
		{
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_NORMAL:
			{
				agent = new hkpSymmetricAgent<hkpConvexListAgent>(bodyA, bodyB, input, mgr);
				break;
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_CONVEX:
			{
				agent = new hkpPredGskfAgent( bodyA, bodyB, input, mgr );
				break;
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_LIST:
			{
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
		agent = new hkpShapeCollectionAgent(bodyA, bodyB, input, mgr);
	}
	return agent;
}


// Special dispatch function for convex list vs convex list
hkpCollisionAgent* HK_CALL hkpConvexListAgent::createConvexListConvexListAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB,
																	const  hkpCollisionInput& input, hkpContactMgr* mgr )
{

	if ( mgr )
	{
		hkpConvexListFilter::ConvexListCollisionType collisionTypeA = input.m_convexListFilter->getConvexListCollisionType( bodyA, bodyB, input );
		switch( collisionTypeA )
		{
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_NORMAL:
			{
				return new hkpSymmetricAgent<hkpConvexListAgent>(bodyA, bodyB, input, mgr);
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_LIST:
			{
				return new hkpShapeCollectionAgent(bodyA, bodyB, input, mgr); // DONE
			}
		case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_CONVEX:
			{
				hkpConvexListFilter::ConvexListCollisionType collisionTypeB = input.m_convexListFilter->getConvexListCollisionType( bodyB, bodyA, input );

				switch( collisionTypeB )
				{
				case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_NORMAL:
					{
						return new hkpConvexListAgent( bodyA, bodyB, input, mgr );
					}
				case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_CONVEX:
					{
						return new hkpPredGskfAgent( bodyA, bodyB, input, mgr );
					}
				case hkpConvexListFilter::TREAT_CONVEX_LIST_AS_LIST:
					{
						return new hkpSymmetricAgent<hkpShapeCollectionAgent>(bodyA, bodyB, input, mgr); // DONE
					}
				default:
					{
						HK_ASSERT2(0xeaf09646, 0, "Unknown ConvexListCollisionType returned");
					}
				}
			}
		default:
			{
				HK_ASSERT2(0xeaf09646, 0, "Unknown ConvexListCollisionType returned");
			}
		}
	}
	else
	{
		return new hkpShapeCollectionAgent(bodyA, bodyB, input, mgr);
	}

	return HK_NULL;
}

void HK_CALL hkpConvexListAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = createConvexListConvexAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpConvexListAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpConvexListAgent>::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpSymmetricAgent<hkpConvexListAgent>::staticLinearCast;
		af.m_isFlipped            = true;
		af.m_isPredictive		  = true;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX_LIST, hkcdShapeType::CONVEX);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = createConvexConvexListAgent;
		af.m_getPenetrationsFunc  = hkpConvexListAgent::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpConvexListAgent::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpConvexListAgent::staticLinearCast;
		af.m_isFlipped            = false;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX, hkcdShapeType::CONVEX_LIST);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = createConvexListConvexListAgent;
		af.m_getPenetrationsFunc  = hkpConvexListAgent::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpConvexListAgent::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpConvexListAgent::staticLinearCast;
		af.m_isFlipped            = false;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX_LIST, hkcdShapeType::CONVEX_LIST );
	}
}

void hkpConvexListAgent::updateShapeCollectionFilter( const hkpCdBody& bodyA, const hkpCdBody& listShapeBodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	if ( m_inGskMode )
	{
		// do nothing
	}
	else
	{
		hkpAgent1nMachine_VisitorInput vin;
		vin.m_bodyA = &bodyA;
		vin.m_collectionBodyB = &listShapeBodyB;
		vin.m_input = &input;
		vin.m_contactMgr = m_contactMgr;
		vin.m_constraintOwner = &constraintOwner;
		vin.m_containerShapeB = static_cast<const hkpShapeCollection*>(listShapeBodyB.getShape())->getContainer();

		hkAgent1nMachine_UpdateShapeCollectionFilter( getStream().m_agentTrack, vin );
	}
}

void hkpConvexListAgent::invalidateTim( const hkpCollisionInput& input)
{
	if ( m_inGskMode )
	{
		hkpPredGskfAgent::invalidateTim(input);
	}
	else
	{
		hkAgent1nMachine_InvalidateTim(getStream().m_agentTrack, input);
	}
}

void hkpConvexListAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	if ( m_inGskMode )
	{
		hkpPredGskfAgent::warpTime( oldTime, newTime, input );
	}
	else
	{
		hkAgent1nMachine_WarpTime(getStream().m_agentTrack, oldTime, newTime, input);
	}
}
void hkpConvexListAgent::removePoint( hkContactPointId idToRemove )
{
	if ( m_inGskMode )
	{
		hkpGskfAgent::removePoint( idToRemove );
	}
}

void hkpConvexListAgent::commitPotential( hkContactPointId idToCommit )
{
	if ( m_inGskMode )
	{
		hkpGskfAgent::commitPotential( idToCommit );
	}
}

void hkpConvexListAgent::createZombie( hkContactPointId idTobecomeZombie )
{
	if ( m_inGskMode )
	{
		hkpGskfAgent::createZombie( idTobecomeZombie );
	}
}


// hkpCollisionAgent interface implementation.
void hkpConvexListAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "CvxList", "checkHull" );
	{
		hkpFlagCdBodyPairCollector checker;
		hkpGskBaseAgent::staticGetPenetrations( bodyA, bodyB, input, checker );
		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("children");
			hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetClosestPoints( bodyA, bodyB, input, collector );
		}
		else
		{
			hkpClosestCdPointCollector closestPoint;
			hkpGskBaseAgent::staticGetClosestPoints( bodyA, bodyB, input, closestPoint );

			// if we have a hit, we need to check whether we are closer than our m_minDistanceToUseConvexHullForGetClosestPoints
			if ( closestPoint.hasHit() )
			{
				const hkpConvexListShape* convexList = reinterpret_cast<const hkpConvexListShape*>( bodyB.getShape() );
				if ( closestPoint.getHitContact().getDistance() > convexList->m_minDistanceToUseConvexHullForGetClosestPoints )
				{
					hkpCdPoint hit( bodyA, bodyB );
					hit.setContact( closestPoint.getHitContact() );
					collector.addCdPoint( hit );
				}
				else
				{
					HK_TIMER_SPLIT_LIST("children");
					hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetClosestPoints( bodyA, bodyB, input, collector );
				}
			}
		}
	}
	HK_TIMER_END_LIST();
}

void hkpConvexListAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector )
{
	hkpConvexListAgent::staticGetClosestPoints( bodyA, bodyB, input, collector );
}

void hkpConvexListAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "CvxList", "checkHull" );
	{
		hkpFlagCdBodyPairCollector checker;
		hkpGskBaseAgent::staticGetPenetrations( bodyA, bodyB, input, checker );
		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("children");
			hkpSymmetricAgent<hkpShapeCollectionAgent>::staticGetPenetrations( bodyA, bodyB, input, collector );
		}
	}
	HK_TIMER_END_LIST();
}

void hkpConvexListAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	hkpConvexListAgent::staticGetPenetrations( bodyA, bodyB, input, collector);
}


void hkpConvexListAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_TIMER_BEGIN_LIST( "CvsListAgent", "checkHull" );
	{
		hkpSimpleClosestContactCollector checker;
		hkpGskBaseAgent::staticLinearCast( bodyA, bodyB, input, checker, &checker );
		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("child");
			hkpSymmetricAgent<hkpShapeCollectionAgent>::staticLinearCast( bodyA, bodyB, input, collector, startCollector );
		}
	}
	HK_TIMER_END_LIST();
}

void hkpConvexListAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	hkpConvexListAgent::staticLinearCast( bodyA, bodyB, input, collector, startCollector );
}


void hkpConvexListAgent::switchToStreamMode(hkCollisionConstraintOwner& constraintOwner)
{
	hkGskManifold_cleanup( m_manifold, m_contactMgr, constraintOwner );
	m_inGskMode = false;
	new ( &getStream().m_agentTrack ) hkpAgent1nTrack();
	hkAgent1nMachine_Create( getStream().m_agentTrack );

	m_inStreamModeCounter = 25;
	getStream().m_inStreamModeTimDist = 0.0f;
}


void hkpConvexListAgent::switchToGskMode(hkCollisionConstraintOwner& constraintOwner)
{
	hkAgent1nMachine_Destroy( getStream().m_agentTrack, m_dispatcher, m_contactMgr, constraintOwner );
	m_manifold.init();
	m_inGskMode = true;
}



void hkpConvexListAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	if ( m_inGskMode )
	{
		hkGskManifold_cleanup( m_manifold, m_contactMgr, constraintOwner );
	}
	else
	{
		hkAgent1nMachine_Destroy( getStream().m_agentTrack, m_dispatcher, m_contactMgr, constraintOwner );
	}
	delete this;
}



void hkpConvexListAgent::processCollision(const hkpCdBody& bodyA,						const hkpCdBody& bodyB,
										 const hkpProcessCollisionInput& input,		hkpProcessCollisionOutput& output)
{
 	HK_ASSERT2(0x57213df1,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN_LIST( "CvxLst", "Tim" );

	//
	//	Get the relative linear movement (xyz) and the worst case angular movment (w)
	//
	const hkpConvexListShape* cls = reinterpret_cast<const hkpConvexListShape*>( bodyB.getShape() );
	hkVector4 timInfo;
	hkSweptTransformUtil::calcTimInfo( *bodyA.getMotionState(), *bodyB.getMotionState(), input.m_stepInfo.m_deltaTime, timInfo);

	// some values to undo the output
	hkpProcessCollisionOutputBackup outputBackup( output );

	if ( m_inGskMode )
	{
gskMode:
		if ( m_separatingNormal(3) > input.getTolerance() )
		{
			m_separatingNormal(3) -= timInfo.dot4xyz1( m_separatingNormal ).getReal();
			if ( m_separatingNormal(3) > input.getTolerance() )
			{
				if ( m_manifold.m_numContactPoints)
				{
			  		hkGskManifold_cleanup( m_manifold, m_contactMgr, *output.m_constraintOwner.val() );
				}
				goto END;
			}
		}
 		HK_TIMER_SPLIT_LIST( "Gsk" );


		// Wrap the contact manager in a version that will convert the points on the hull of the
		// convex list to points on the sub shapes
		hkpMapPointsToSubShapeContactMgr mappingMgr( m_contactMgr );
		m_contactMgr = &mappingMgr;

		hkpPredGskfAgent::processCollision( bodyA, bodyB, input, output );

		m_contactMgr = mappingMgr.m_contactMgr;


		if (mappingMgr.m_invalidPointHit)
		{
			// assert no added TOIs or contact points - this is not always the case currently - see below
			//HK_ASSERT()
switchToStreamModeLabel:

			// XXX - This line is necessary because when addContactPoint is called from line 167 in gskAgentUtil, the INVALID return
			// seems to be ignored and the point added anyway causing an assert in the process contact - some artifact of the welding code I think.
			outputBackup.rollbackOutput( bodyA, bodyB, output, m_contactMgr );
			switchToStreamMode( *output.m_constraintOwner.val() );
			goto streamMode;
		}

		//
		// If we get a penetration (which is supported by 1 piece) normally we want to use the outer hull to push it out.
		// However if we start in the penetrating case, we want to use penetrations with the inner pieces
		//
		if ( m_manifold.m_numContactPoints)
		{
			if (!m_processFunctionCalled)
			{
				hkReal allowedPenetration = 2.0f * hkMath::min2(bodyA.getRootCollidable()->getAllowedPenetrationDepth(), bodyB.getRootCollidable()->getAllowedPenetrationDepth());
				if ( m_separatingNormal(3) < -allowedPenetration )
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
		if ( m_inStreamModeCounter-- < 0)
		{
			m_inStreamModeCounter = 25;
			//if ( getStream().m_inStreamModeTimDist < 0.0f)
			{
				hkpGsk::GetClosesetPointInput gskInput;
				hkTransform aTb;	aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform());
				{
					gskInput.m_shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
					gskInput.m_shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());
					gskInput.m_aTb = &aTb;
					gskInput.m_transformA = &bodyA.getTransform();
					gskInput.m_collisionTolerance = input.getTolerance();
				}

				hkVector4 pointOnB;
				if( hkpGsk::getClosestPoint( gskInput, m_cache, m_separatingNormal, pointOnB ) != HK_SUCCESS )
				{
					switchToGskMode( *output.m_constraintOwner.val() );
					outputBackup.rollbackOutput( bodyA, bodyB, output, m_contactMgr );
					goto gskMode;
				}
				getStream().m_inStreamModeTimDist = -m_separatingNormal(3);
			}
		}
		getStream().m_inStreamModeTimDist -= timInfo.length<3>().getReal();
		//
		//	Set the input structure
		//
		hkpAgent3ProcessInput in3;
		{
			in3.m_bodyA = &bodyA;
			in3.m_bodyB = &bodyB;
			in3.m_contactMgr = m_contactMgr;
			in3.m_input = &input;
			in3.m_linearTimInfo = timInfo;

			const hkMotionState* msA = bodyA.getMotionState();
			const hkMotionState* msB = bodyB.getMotionState();

			in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());
		}

		{
		  int size = cls->m_childShapes.getSize();
		  hkLocalBuffer<hkpShapeKey> hitList( size+1 );
		  for ( int i = 0; i < size; i++ ){		hitList[i] = static_cast<hkUint32>(i);	}
		  hitList[size] = HK_INVALID_SHAPE_KEY;

		  hkAgent1nMachine_Process( getStream().m_agentTrack, in3, cls, hitList.begin(), output );
		}
	}

END:;
	m_processFunctionCalled = true;
	HK_TIMER_END_LIST();
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
