/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereCapsule/hkpSphereCapsuleAgent.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>


hkpSphereCapsuleAgent::hkpSphereCapsuleAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
: hkpIterativeLinearCastAgent( mgr )
{
	m_contactPointId = HK_INVALID_CONTACT_POINT;
}

void HK_CALL hkpSphereCapsuleAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createCapsuleSphereAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpSphereCapsuleAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpSphereCapsuleAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpSphereCapsuleAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::CAPSULE, hkcdShapeType::SPHERE);	
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createSphereCapsuleAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::SPHERE, hkcdShapeType::CAPSULE);	
	}
}


hkpCollisionAgent* HK_CALL hkpSphereCapsuleAgent::createCapsuleSphereAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpSphereCapsuleAgent* agent = new hkpSymmetricAgentLinearCast<hkpSphereCapsuleAgent>(bodyA, bodyB, input, mgr);
	return agent;
}


hkpCollisionAgent* HK_CALL hkpSphereCapsuleAgent::createSphereCapsuleAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
    return new hkpSphereCapsuleAgent( bodyA, bodyB, input, mgr );
}

void hkpSphereCapsuleAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	if(m_contactPointId != HK_INVALID_CONTACT_POINT)
	{
		m_contactMgr->removeContactPoint(m_contactPointId, constraintOwner );
	}
	delete this;
}

hkpSphereCapsuleAgent::ClosestPointResult hkpSphereCapsuleAgent::getClosestPointInl(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkContactPoint& cpoint )
{
    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpCapsuleShape* capsuleB = static_cast<const hkpCapsuleShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();

	hkVector4 capsB[2];
	hkVector4Util::transformPoints( bodyB.getTransform(), capsuleB->getVertices(), 2, &capsB[0]);

	hkLineSegmentUtil::ClosestPointLineSegResult result;
	hkLineSegmentUtil::closestPointLineSeg( posA, capsB[0], capsB[1], result );

	hkVector4 aMinusB; aMinusB.setSub( posA, result.m_pointOnEdge );

	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + capsuleB->getRadius());
	const hkSimdReal refDist = radiusSum + hkSimdReal::fromFloat(input.getTolerance());

	const hkSimdReal distSquared = aMinusB.lengthSquared<3>();
	if ( distSquared.isGreaterEqual(refDist * refDist) )
	{
		return ST_CP_MISS;
	}

	hkSimdReal dist;
	hkVector4 sepNormal;
	if ( distSquared.isGreaterZero() )
	{
		dist = distSquared.sqrt<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		sepNormal = aMinusB;
	}
	else
	{
		dist.setZero();
		hkVector4 edge; edge.setSub( capsB[1], capsB[0] );
		hkVector4Util::calculatePerpendicularVector( edge, sepNormal );
	}
	sepNormal.normalize<3>();
	cpoint.setSeparatingNormal(sepNormal, dist - radiusSum);
	hkVector4 cpPos; cpPos.setAddMul( posA, cpoint.getNormal(), hkSimdReal::fromFloat(capsuleB->getRadius()) - dist );
	cpoint.setPosition(cpPos);
	return ST_CP_HIT;
}

hkpSphereCapsuleAgent::ClosestPointResult HK_CALL hkpSphereCapsuleAgent::getClosestPoint(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkContactPoint& cpoint )
{
	return getClosestPointInl( bodyA, bodyB, input, cpoint );
}





void hkpSphereCapsuleAgent::processCollision(const  hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x1beb1a11,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("SphereCapsule", HK_NULL);

	hkpProcessCdPoint& point = *result.reserveContactPoints(1);

	if (getClosestPointInl( bodyA, bodyB, input, point.m_contact) != ST_CP_MISS)
	{
		if(m_contactPointId == HK_INVALID_CONTACT_POINT)
		{
			m_contactPointId = m_contactMgr->addContactPoint(bodyA, bodyB, input, result, HK_NULL, point.m_contact );
		}

		if ( m_contactPointId != HK_INVALID_CONTACT_POINT )
		{
			point.m_contactPointId = m_contactPointId;
			result.commitContactPoints(1);
		}
		else
		{
			result.abortContactPoints(1);
		}

	}
	else
	{
		result.abortContactPoints(1);
		if(m_contactPointId != HK_INVALID_CONTACT_POINT)
		{
			m_contactMgr->removeContactPoint(m_contactPointId, *result.m_constraintOwner.val() );
			m_contactPointId = HK_INVALID_CONTACT_POINT;
		}
	}

	HK_TIMER_END();
}

void hkpSphereCapsuleAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereCapsule", HK_NULL);

	hkContactPoint contact;
	if (getClosestPointInl( bodyA, bodyB, input, contact))
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}

void hkpSphereCapsuleAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereCapsule", HK_NULL);
	
	hkContactPoint contact;
	if (getClosestPointInl( bodyA, bodyB, input, contact))
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}
	
void hkpSphereCapsuleAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("SphereCapsule", HK_NULL);

    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpCapsuleShape* capsuleB = static_cast<const hkpCapsuleShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();

	hkVector4 capsB[2]; hkVector4Util::transformPoints( bodyB.getTransform(), capsuleB->getVertices(), 2, &capsB[0]);

	hkLineSegmentUtil::ClosestPointLineSegResult result;
	hkLineSegmentUtil::closestPointLineSeg( posA, capsB[0], capsB[1], result );

	hkVector4 aMinusB; aMinusB.setSub( result.m_pointOnEdge, posA );

	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + capsuleB->getRadius());
	const hkSimdReal refDist = radiusSum;

	const hkSimdReal distSquared = aMinusB.lengthSquared<3>();
	if ( distSquared.isLess(refDist * refDist) )
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}
    HK_TIMER_END();
}


void hkpSphereCapsuleAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	staticGetPenetrations( bodyA, bodyB, input, collector );
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
