/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/MiscAgent/Phantom/hkpPhantomAgent.h>
#include <Physics2012/Collide/Shape/Misc/PhantomCallback/hkpPhantomCallbackShape.h>


void HK_CALL hkpPhantomAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// symmetric version = normal version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = createPhantomAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isPredictive        = true;	// its really not predictive but we have no other fallback

		dispatcher->registerCollisionAgent(af, hkcdShapeType::PHANTOM_CALLBACK, hkcdShapeType::ALL_SHAPE_TYPES);
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::PHANTOM_CALLBACK);
	}
}

hkpPhantomAgent::hkpPhantomAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, hkpContactMgr* contactMgr)
: hkpCollisionAgent( contactMgr )
{
	m_collidableA = bodyA.getRootCollidable();
	m_collidableB = bodyB.getRootCollidable();

	m_bodyTypeA = bodyA.getShape()->getType();
	m_bodyTypeB = bodyB.getShape()->getType();
}

hkpCollisionAgent* HK_CALL hkpPhantomAgent::createPhantomAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
	hkpPhantomAgent* pa = new hkpPhantomAgent( bodyA, bodyB, contactMgr );
	if ( pa->m_bodyTypeA == hkcdShapeType::PHANTOM_CALLBACK )
	{
		const hkpPhantomCallbackShape* constPhantomShape = static_cast<const hkpPhantomCallbackShape*>( bodyA.getShape() );
		hkpPhantomCallbackShape* phantomShape = const_cast<hkpPhantomCallbackShape*>( constPhantomShape );
		phantomShape->phantomEnterEvent( bodyA.getRootCollidable(), bodyB.getRootCollidable(), input );
		pa->m_shapeA = phantomShape;
	}

	if ( pa->m_bodyTypeB == hkcdShapeType::PHANTOM_CALLBACK )
	{
		const hkpPhantomCallbackShape* constPhantomShape = static_cast<const hkpPhantomCallbackShape*>( bodyB.getShape() );
		hkpPhantomCallbackShape* phantomShape = const_cast<hkpPhantomCallbackShape*>( constPhantomShape );
		phantomShape->phantomEnterEvent( bodyB.getRootCollidable(), bodyA.getRootCollidable(), input );
		pa->m_shapeB = phantomShape;
	}

	return pa;
}

hkpCollisionAgent* HK_CALL hkpPhantomAgent::createNoPhantomAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{

	if ( bodyA.getShape()->getType() == hkcdShapeType::PHANTOM_CALLBACK )
	{
		const hkpPhantomCallbackShape* constPhantomShape = static_cast<const hkpPhantomCallbackShape*>( bodyA.getShape() );
		hkpPhantomCallbackShape* phantomShape = const_cast<hkpPhantomCallbackShape*>( constPhantomShape );
		phantomShape->phantomEnterEvent( bodyA.getRootCollidable(), bodyB.getRootCollidable(), input );
	}

	if ( bodyB.getShape()->getType() == hkcdShapeType::PHANTOM_CALLBACK )
	{
		const hkpPhantomCallbackShape* constPhantomShape = static_cast<const hkpPhantomCallbackShape*>( bodyB.getShape() );
		hkpPhantomCallbackShape* phantomShape = const_cast<hkpPhantomCallbackShape*>( constPhantomShape );
		phantomShape->phantomEnterEvent( bodyB.getRootCollidable(), bodyA.getRootCollidable(), input );
	}

	return HK_NULL;	
}




void hkpPhantomAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	if ( m_bodyTypeA == hkcdShapeType::PHANTOM_CALLBACK )
	{
		m_shapeA->phantomLeaveEvent( m_collidableA, m_collidableB );
	}

	if ( m_bodyTypeB == hkcdShapeType::PHANTOM_CALLBACK )
	{
		m_shapeB->phantomLeaveEvent( m_collidableB, m_collidableA );
	}
	delete this;
}






void hkpPhantomAgent::processCollision(const hkpCdBody& bodyAin, const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
}

void hkpPhantomAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	collector.addCdBodyPair( bodyA, bodyB );
}

void hkpPhantomAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	collector.addCdBodyPair( bodyA, bodyB );
}

void hkpPhantomAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& pointDetails)
{
}

void hkpPhantomAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& pointDetails)
{
}

void hkpPhantomAgent::linearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
}

void hkpPhantomAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
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
