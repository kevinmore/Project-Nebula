/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Agent/ConvexAgent/BoxBox/hkpBoxBoxAgent.h>

#include <Physics2012/Collide/BoxBox/hkpBoxBoxCollisionDetection.h>

#include <Physics2012/Collide/BoxBox/hkpBoxBoxContactPoint.h>

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskfAgent.h>

hkBool hkpBoxBoxAgent::m_attemptToFindAllEdges = false;


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpBoxBoxAgent::initAgentFunc(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc          = createBoxBoxAgent;
	af.m_getPenetrationsFunc = staticGetPenetrations;
	af.m_getClosestPointFunc = hkpGskfAgent::staticGetClosestPoints;
	af.m_linearCastFunc      = staticLinearCast;
	af.m_isFlipped           = false;
	af.m_isPredictive		 = false;
}

#else

void HK_CALL hkpBoxBoxAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc	 = staticLinearCast;
}

#endif


#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkpBoxBoxAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::AgentFuncs af;
	initAgentFunc(af);
	dispatcher->registerCollisionAgent(af, hkcdShapeType::BOX, hkcdShapeType::BOX);	
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpBoxBoxAgent::createBoxBoxAgent(const hkpCdBody& bodyA, 	const hkpCdBody& bodyB, 	const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
	const hkpBoxShape* boxA = static_cast<const hkpBoxShape*>(bodyA.getShape());
	const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());

	const hkVector4& extA = boxA->getHalfExtents();
	const hkVector4& extB = boxB->getHalfExtents();

	// box box breaks down when the tolerance becomes larger than the minimum extent size
	// if this is the case create a convex-convex agent.
	const hkSimdReal tc = hkSimdReal::fromFloat(hkReal(1.999f));
	const hkSimdReal aTol = extA.getW() * tc;
	const hkSimdReal bTol = extB.getW() * tc;
	hkSimdReal tol; tol.load<1>(&(input.m_tolerance.ref()));
	hkVector4Comparison either; either.setOr(tol.greaterEqual(aTol), tol.greaterEqual(bTol));
	if ( either.anyIsSet() )
	{
		return hkpGskfAgent::createGskfAgent( bodyA, bodyB, input, contactMgr ); 	
	}
	else
	{
		hkpBoxBoxAgent* agent = new hkpBoxBoxAgent(contactMgr);
		return agent;
	}

}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpBoxBoxAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	for (int i = 0; i < m_manifold.getNumPoints(); i++)
	{
		m_contactMgr->removeContactPoint(m_manifold[i].m_contactPointId, constraintOwner );
	}
	delete this;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpBoxBoxAgent::processCollision(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
									const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x3b792884,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );
	HK_TIMER_BEGIN("BoxBox", this);
	const hkpBoxShape* boxA = static_cast<const hkpBoxShape*>(bodyA.getShape());
	const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());

	const hkVector4& extA = boxA->getHalfExtents();
	const hkVector4& extB = boxB->getHalfExtents();

	hkSimdReal rA; rA.load<1>(&boxA->getRadius());
	hkSimdReal rB; rB.load<1>(&boxB->getRadius());
	hkVector4 rA4; rA4.setAdd(extA, rA);
	hkVector4 rB4; rB4.setAdd(extB, rB);

	hkSimdReal tolerance = hkSimdReal::fromFloat(input.getTolerance());

	hkTransform aTb; aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform() );
	hkpBoxBoxCollisionDetection detector( bodyA, bodyB, &input, m_contactMgr, &result, aTb, bodyA.getTransform(), rA4, bodyB.getTransform(), rB4, tolerance );

	detector.calcManifold( m_manifold );

	HK_TIMER_END();
}
#endif


void hkpBoxBoxAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector  )
{
	HK_TIMER_BEGIN("BoxBox", this);
	const hkpBoxShape* boxA = static_cast<const hkpBoxShape*>(bodyA.getShape());
	const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());

	hkSimdReal rA; rA.load<1>(&boxA->getRadius());
	hkSimdReal rB; rB.load<1>(&boxB->getRadius());
	hkVector4 rA4; rA4.setAdd(boxA->getHalfExtents(), rA);
	hkVector4 rB4; rB4.setAdd(boxB->getHalfExtents(), rB);

	hkSimdReal tolerance = hkSimdReal::fromFloat(input.getTolerance());

	hkTransform aTb; aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform() );
	hkpBoxBoxCollisionDetection detector( bodyA, bodyB, HK_NULL, HK_NULL, HK_NULL,
										 aTb, bodyA.getTransform(), rA4,
										 bodyB.getTransform(), rB4, tolerance );
	
	hkContactPoint contact;
	hkBool result = detector.calculateClosestPoint( contact );

	if (result)
	{ 
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}
	HK_TIMER_END();
}


#if !defined(HK_PLATFORM_SPU)
void hkpBoxBoxAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector  )
{
	staticGetClosestPoints( bodyA, bodyB, input, collector );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpBoxBoxAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("BoxBox", this);
	const hkpBoxShape* boxA = static_cast<const hkpBoxShape*>(bodyA.getShape());
	const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());

	hkSimdReal rA; rA.load<1>(&boxA->getRadius());
	hkSimdReal rB; rB.load<1>(&boxB->getRadius());
	hkVector4 rA4; rA4.setAdd(boxA->getHalfExtents(), rA);
	hkVector4 rB4; rB4.setAdd(boxB->getHalfExtents(), rB);

	hkSimdReal tolerance = hkSimdReal::fromFloat(input.getTolerance());

	hkTransform aTb; aTb.setMulInverseMul( bodyA.getTransform(), bodyB.getTransform() );
	hkpBoxBoxCollisionDetection detector( bodyA, bodyB, HK_NULL, HK_NULL, HK_NULL,
										 aTb, bodyA.getTransform(), rA4,
										 bodyB.getTransform(), rB4, tolerance );

	if (detector.getPenetrations())
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}
	HK_TIMER_END();
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpBoxBoxAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	staticGetPenetrations( bodyA, bodyB, input, collector );
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
