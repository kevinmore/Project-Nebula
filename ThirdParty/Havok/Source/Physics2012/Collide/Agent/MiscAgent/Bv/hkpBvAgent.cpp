/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>

#include <Physics2012/Collide/Agent/MiscAgent/Bv/hkpBvAgent.h>

#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>

#include <Physics2012/Collide/Dispatch/ContactMgr/hkpContactMgrFactory.h>
#include <Physics2012/Collide/Dispatch/hkpAgentDispatchUtil.h>
#include <Physics2012/Collide/Agent/Query/hkpLinearCastCollisionInput.h>

#include <Physics2012/Collide/Query/Collector/BodyPairCollector/hkpFlagCdBodyPairCollector.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpSimpleClosestContactCollector.h>


hkpBvAgent::hkpBvAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
:hkpCollisionAgent( mgr )
{

	const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());

	hkpCdBody newA( &bodyA );
	newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

	m_boundingVolumeAgent = input.m_dispatcher->getNewCollisionAgent( newA, bodyB, input, mgr );
	m_childAgent = HK_NULL;
}

void HK_CALL hkpBvAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createShapeBvAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpBvAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpBvAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpBvAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;	// there is no fallback
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::BV );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createBvShapeAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;	// there is no fallback
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV, hkcdShapeType::ALL_SHAPE_TYPES );
	}
}



hkpCollisionAgent* HK_CALL hkpBvAgent::createBvShapeAgent(const hkpCdBody& A, const hkpCdBody& B, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpBvAgent* agent = new hkpBvAgent(A, B, input, mgr);
	return agent;
}


hkpCollisionAgent* HK_CALL hkpBvAgent::createShapeBvAgent(const hkpCdBody& A, const hkpCdBody& B, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpBvAgent* agent = new hkpSymmetricAgent<hkpBvAgent>( A, B, input, mgr );
	return agent;
}


void hkpBvAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	m_boundingVolumeAgent->cleanup( constraintOwner );
	if ( m_childAgent )
	{
		m_childAgent->cleanup( constraintOwner);
		m_childAgent = HK_NULL;
	}

	delete this;
}

void hkpBvAgent::invalidateTim( const hkpCollisionInput& input)
{
	m_boundingVolumeAgent->invalidateTim(input);
	if ( m_childAgent )
	{
		m_childAgent->invalidateTim(input);
	}
}

void hkpBvAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	m_boundingVolumeAgent->warpTime( oldTime, newTime, input );
	if ( m_childAgent )
	{
		m_childAgent->warpTime( oldTime, newTime, input );
	}
}

void hkpBvAgent::removePoint( hkContactPointId idToRemove )
{
	if ( m_childAgent )
	{
		m_childAgent->removePoint( idToRemove );
	}
}

void hkpBvAgent::commitPotential( hkContactPointId newId )
{
	if ( m_childAgent )
	{
		m_childAgent->commitPotential( newId );
	}
}

void hkpBvAgent::createZombie( hkContactPointId idTobecomeZombie )
{
	if ( m_childAgent )
	{
		m_childAgent->createZombie( idTobecomeZombie );
	}
}

void hkpBvAgent::processCollision(const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x325d7b4b,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );

	const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyAin.getShape());

	hkpCdBody newA( &bodyAin);
	newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

	hkpFlagCdBodyPairCollector checker;
	m_boundingVolumeAgent->getPenetrations( newA, bodyB, input, checker );

	if ( checker.hasHit() )
	{
		HK_TIMER_SPLIT_LIST("child");
		newA.setShape( bvShape->getChildShape(), 0 );
		if ( ! m_childAgent )
		{
			m_childAgent = input.m_dispatcher->getNewCollisionAgent( newA, bodyB, input, m_contactMgr );
		}
		m_childAgent->processCollision( newA, bodyB, input, result );
	}
	else
	{
		if ( m_childAgent )
		{
			m_childAgent->cleanup( *(hkCollisionConstraintOwner*)result.m_constraintOwner );
			m_childAgent = HK_NULL;
		}
	}
	HK_TIMER_END_LIST();
}


void hkpBvAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& pointDetails, hkpCdPointCollector* startCollector)
{
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );

	const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());
	hkpCdBody newA( &bodyA);
	newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

	hkpSimpleClosestContactCollector checker;
	m_boundingVolumeAgent->linearCast( newA, bodyB, input, checker, &checker );
	if ( checker.hasHit() )
	{
		HK_TIMER_SPLIT_LIST("child");
		newA.setShape( bvShape->getChildShape(), 0 );
		if ( ! m_childAgent )
		{
			m_childAgent = input.m_dispatcher->getNewCollisionAgent( newA, bodyB, input, m_contactMgr );
		}
		m_childAgent->linearCast( newA, bodyB, input, pointDetails, startCollector );
	}
	else
	{
// we do not have a constraintOwner, so we cannot delete the child agent
// 		if ( m_childAgent )
// 		{
// 			m_childAgent->cleanup( ) ;
// 			m_childAgent = HK_NULL;
// 		}
	}
	HK_TIMER_END_LIST();
}

void hkpBvAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& pointDetails, hkpCdPointCollector* startCollector)
{
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );
	{

		const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());
		hkpCdBody newA( &bodyA );
		newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );


		hkpShapeType typeA = newA.getShape()->getType();
		hkpShapeType typeB = bodyB.getShape()->getType();

		hkpCollisionDispatcher::LinearCastFunc bvLinearCastFunc = input.m_dispatcher->getLinearCastFunc( typeA, typeB );

		hkpSimpleClosestContactCollector checker;
		bvLinearCastFunc( newA, bodyB, input, checker, &checker );
		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("child");
			newA.setShape( bvShape->getChildShape(), 0 );
			typeA = newA.getShape()->getType();

			hkpCollisionDispatcher::LinearCastFunc childLinearCast = input.m_dispatcher->getLinearCastFunc( typeA, typeB );
			childLinearCast( newA, bodyB, input, pointDetails, startCollector );
		}
	}
	HK_TIMER_END_LIST();
}


// hkpCollisionAgent interface implementation.
void hkpBvAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );
	{
		const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());

		hkpCdBody newA( &bodyA );
		newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );
		hkpFlagCdBodyPairCollector checker;
		m_boundingVolumeAgent->getPenetrations( newA, bodyB, input, checker );
		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("child");
			newA.setShape( bvShape->getChildShape(), 0 );
			if ( ! m_childAgent )
			{
				m_childAgent = input.m_dispatcher->getNewCollisionAgent( newA, bodyB, input, m_contactMgr );
			}
			m_childAgent->getClosestPoints( newA, bodyB, input, collector );
		}
		else
		{
// we do not have a constraintOwner, so we cannot delete the child agent
// 			if ( m_childAgent )
// 			{
// 				m_childAgent->cleanup() ;
// 				m_childAgent = HK_NULL;
// 			}
		}
	}
	HK_TIMER_END_LIST();
}


// hkpCollisionAgent interface implementation.
void hkpBvAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );
	{
		const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());

		hkpCdBody newA( &bodyA );
		newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

		hkpShapeType typeA = newA.getShape()->getType();
		hkpShapeType typeB = bodyB.getShape()->getType();

		hkpCollisionDispatcher::GetPenetrationsFunc boundingVolumeGetPenetrations = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );

		hkpFlagCdBodyPairCollector checker;
		boundingVolumeGetPenetrations( newA, bodyB, input, checker );

		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("child");
			newA.setShape( bvShape->getChildShape(), 0 );
			typeA = newA.getShape()->getType();

			hkpCollisionDispatcher::GetClosestPointsFunc childGetClosestPoint = input.m_dispatcher->getGetClosestPointsFunc( typeA, typeB );
			childGetClosestPoint( newA, bodyB, input, collector );

		}
	}
	HK_TIMER_END_LIST();
}




void hkpBvAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );
	{
		const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());

		hkpCdBody newA( &bodyA );
		newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

		hkpFlagCdBodyPairCollector checker;
		m_boundingVolumeAgent->getPenetrations( newA, bodyB, input, checker );

		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("child");
			newA.setShape( bvShape->getChildShape(), 0 );
			if ( ! m_childAgent )
			{
				m_childAgent = input.m_dispatcher->getNewCollisionAgent( newA, bodyB, input, m_contactMgr );
			}
			m_childAgent->getPenetrations(newA, bodyB, input, collector);
		}
	}
	HK_TIMER_END_LIST();
}

void hkpBvAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "hkpBvAgent", "checkBvShape" );
	{
		const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());

		hkpCdBody newA( &bodyA );
		newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

		hkpShapeType typeA = newA.getShape()->getType();
		hkpShapeType typeB = bodyB.getShape()->getType();

		hkpCollisionDispatcher::GetPenetrationsFunc boundingVolumeGetPenetrations = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );

		hkpFlagCdBodyPairCollector checker;
		boundingVolumeGetPenetrations( newA, bodyB, input, checker );

		if ( checker.hasHit() )
		{
			HK_TIMER_SPLIT_LIST("child");
			newA.setShape( bvShape->getChildShape(), 0 );
			typeA = newA.getShape()->getType();

			hkpCollisionDispatcher::GetPenetrationsFunc childGetPenetrations = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );
			childGetPenetrations(newA, bodyB, input, collector);
		}
	}
	HK_TIMER_END_LIST();
}

void hkpBvAgent::updateShapeCollectionFilter( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(bodyA.getShape());
	hkpCdBody newA( &bodyA);
	newA.setShape( bvShape->getBoundingVolumeShape(), HK_INVALID_SHAPE_KEY );

	m_boundingVolumeAgent->updateShapeCollectionFilter( newA, bodyB, input, constraintOwner);

	if (m_childAgent)
	{
		newA.setShape( bvShape->getChildShape(), 0 );
		m_childAgent->updateShapeCollectionFilter(newA, bodyB, input, constraintOwner);
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
