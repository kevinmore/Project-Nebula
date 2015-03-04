/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/MiscAgent/Transform/hkpTransformAgent.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>

#if ! defined (HK_PLATFORM_SPU)
void HK_CALL hkpTransformAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
		// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createTransformBAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpTransformAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpTransformAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpTransformAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::TRANSFORM );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createTransformAAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRANSFORM, hkcdShapeType::ALL_SHAPE_TYPES );
	}
}
#endif

#if defined(HK_PLATFORM_SPU)
void HK_CALL hkpTransformAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = hkpTransformAgent::staticGetClosestPoints;
	af.m_linearCastFunc	     = hkpTransformAgent::staticLinearCast;
}


void HK_CALL hkpTransformAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = hkpSymmetricAgent<hkpTransformAgent>::staticGetClosestPoints;
	af.m_linearCastFunc	     = hkpSymmetricAgent<hkpTransformAgent>::staticLinearCast;
}
#endif

#if ! defined (HK_PLATFORM_SPU)
void hkpTransformAgent::cleanup(	hkCollisionConstraintOwner& constraintOwner )
{
	m_childAgent->cleanup( constraintOwner );
	delete this;
}

hkpTransformAgent::hkpTransformAgent(const hkpCdBody& bodyAIn, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
: hkpCollisionAgent( mgr )
{
	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAIn.getShape());
	const hkpShape* childShape = tShapeA->getChildShape();

	hkMotionState ms = *bodyAIn.getMotionState();

	ms.getTransform().setMul( bodyAIn.getTransform(), tShapeA->getTransform());

	hkpCdBody bodyA( &bodyAIn, &ms);
	bodyA.setShape( childShape, 0 );

	m_childAgent = input.m_dispatcher->getNewCollisionAgent( bodyA, bodyB, input, mgr );
}

hkpCollisionAgent* HK_CALL hkpTransformAgent::createTransformAAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpTransformAgent* agent = new hkpTransformAgent( bodyA, bodyB, input, mgr );
	return agent;
}


hkpCollisionAgent* HK_CALL hkpTransformAgent::createTransformBAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB,
														const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpTransformAgent* agent = new hkpSymmetricAgent<hkpTransformAgent>(bodyA, bodyB, input,mgr);
	return agent;
}

void hkpTransformAgent::processCollision(const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x1792c490,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN( "Transform", HK_NULL );

	const hkpTransformShape* tShapeA =  static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkMotionState ms;

	//
	//	Calc transform
	//
	ms.getTransform().setMul( bodyAin.getTransform(), tShapeA->getTransform());

	//
	//	Calc swept transform
	//
	{
		hkSweptTransform& st = ms.getSweptTransform();
		const hkSweptTransform& ss = bodyAin.getMotionState()->getSweptTransform();

		st.m_centerOfMass0 = ss.m_centerOfMass0;
		st.m_centerOfMass1 = ss.m_centerOfMass1;

		st.m_rotation0.setMul( ss.m_rotation0, tShapeA->getRotation() );
		st.m_rotation1.setMul( ss.m_rotation1, tShapeA->getRotation() );

		st.m_centerOfMassLocal._setTransformedInversePos( tShapeA->getTransform(), ss.m_centerOfMassLocal );
	}
	const hkMotionState& ss = *bodyAin.getMotionState();
	ms.m_deltaAngle = ss.m_deltaAngle;
	ms.m_objectRadius    = ss.m_objectRadius;

	hkpCdBody copyBodyA( &bodyAin,  &ms);
	copyBodyA.setShape( tShapeA->getChildShape(), 0 );

	m_childAgent->processCollision( copyBodyA, bodyB, input, result );

	HK_TIMER_END();
}

		// hkpCollisionAgent interface implementation.
void hkpTransformAgent::linearCast( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_INTERNAL_TIMER_BEGIN( "Transform", HK_NULL);

	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody copyBodyA( &bodyAin,  &t);
	copyBodyA.setShape( tShapeA->getChildShape(), 0 );

	m_childAgent->linearCast( copyBodyA, bodyB, input, collector, startCollector);

	HK_INTERNAL_TIMER_END();
}
#endif

void hkpTransformAgent::staticLinearCast( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_INTERNAL_TIMER_BEGIN( "Transform", HK_NULL );

	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody childBodyA( &bodyAin,  &t);
	childBodyA.setShape( tShapeA->getChildShape(), 0 );

	hkpShapeType typeA = childBodyA.getShape()->getType();
	hkpShapeType typeB = bodyB.getShape()->getType();
#if ! defined (HK_PLATFORM_SPU)
	hkpCollisionDispatcher::LinearCastFunc linearCastFunc = input.m_dispatcher->getLinearCastFunc( typeA, typeB );
#else
	hkpSpuCollisionQueryDispatcher::LinearCastFunc linearCastFunc = input.m_queryDispatcher->getLinearCastFunc( typeA, typeB );
#endif
	linearCastFunc( childBodyA, bodyB, input, collector, startCollector);

	HK_INTERNAL_TIMER_END();
}

#if ! defined (HK_PLATFORM_SPU)
void hkpTransformAgent::getClosestPoints( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& pointDetails)
{
	HK_INTERNAL_TIMER_BEGIN( "Transform", HK_NULL );

	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody copyBodyA( &bodyAin,  &t);
	copyBodyA.setShape( tShapeA->getChildShape(), 0 );

	m_childAgent->getClosestPoints( copyBodyA, bodyB, input, pointDetails);

	HK_INTERNAL_TIMER_END();
}
#endif

void hkpTransformAgent::staticGetClosestPoints( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& pointDetails)
{
	HK_INTERNAL_TIMER_BEGIN( "Transform", HK_NULL );

	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody childBodyA( &bodyAin,  &t);
	childBodyA.setShape( tShapeA->getChildShape(), 0 );

	hkpShapeType typeA = childBodyA.getShape()->getType();
	hkpShapeType typeB = bodyB.getShape()->getType();
#if ! defined (HK_PLATFORM_SPU)
	hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointFunc = input.m_dispatcher->getGetClosestPointsFunc( typeA, typeB );
#else
	hkpSpuCollisionQueryDispatcher::GetClosestPointsFunc getClosestPointFunc = input.m_queryDispatcher->getGetClosestPointsFunc( typeA, typeB );
#endif

	getClosestPointFunc( childBodyA, bodyB, input, pointDetails);

	HK_INTERNAL_TIMER_END();
}

#if ! defined (HK_PLATFORM_SPU)
void hkpTransformAgent::getPenetrations( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN( "Transform", HK_NULL );

	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody copyBodyA( &bodyAin,  &t);
	copyBodyA.setShape( tShapeA->getChildShape(), 0 );

	HK_INTERNAL_TIMER_END();
	m_childAgent->getPenetrations( copyBodyA, bodyB, input, collector );
}

void hkpTransformAgent::staticGetPenetrations( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_INTERNAL_TIMER_BEGIN( "Transform", HK_NULL );

	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody copyBodyA( &bodyAin,  &t);
	copyBodyA.setShape( tShapeA->getChildShape(), 0 );

	hkpShapeType typeA = copyBodyA.getShape()->getType();
	hkpShapeType typeB = bodyB.getShape()->getType();
	hkpCollisionDispatcher::GetPenetrationsFunc getPenetrationsFunc = input.m_dispatcher->getGetPenetrationsFunc( typeA, typeB );
	getPenetrationsFunc( copyBodyA, bodyB, input, collector );

	HK_INTERNAL_TIMER_END();
}

void hkpTransformAgent::updateShapeCollectionFilter( const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkCollisionConstraintOwner& constraintOwner )
{
	const hkpTransformShape* tShapeA = static_cast<const hkpTransformShape*>(bodyAin.getShape());

	hkTransform t;	t.setMul( bodyAin.getTransform(), tShapeA->getTransform());

	hkpCdBody copyBodyA( &bodyAin,  &t);
	copyBodyA.setShape( tShapeA->getChildShape(), 0 );

	m_childAgent->updateShapeCollectionFilter( copyBodyA, bodyB, input, constraintOwner );

}

void hkpTransformAgent::invalidateTim( const hkpCollisionInput& input )
{
	m_childAgent->invalidateTim( input );
}

void hkpTransformAgent::warpTime( hkTime oldTime, hkTime newTime, const hkpCollisionInput& input )
{
	m_childAgent->warpTime( oldTime, newTime, input );
}

void hkpTransformAgent::removePoint( hkContactPointId idToRemove )
{
	m_childAgent->removePoint( idToRemove );
}

void hkpTransformAgent::commitPotential( hkContactPointId newId )
{
	m_childAgent->commitPotential( newId );
}

void hkpTransformAgent::createZombie( hkContactPointId idTobecomeZombie )
{
	m_childAgent->createZombie( idTobecomeZombie );
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
