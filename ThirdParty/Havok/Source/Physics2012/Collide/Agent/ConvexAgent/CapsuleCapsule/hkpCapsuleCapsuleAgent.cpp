/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Agent/ConvexAgent/CapsuleCapsule/hkpCapsuleCapsuleAgent.h>
#include <Physics2012/Collide/Util/hkpCollideCapsuleUtil.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>



hkpCapsuleCapsuleAgent::hkpCapsuleCapsuleAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
: hkpIterativeLinearCastAgent( mgr )
{
	m_contactPointId[0] = HK_INVALID_CONTACT_POINT;
	m_contactPointId[1] = HK_INVALID_CONTACT_POINT;
	m_contactPointId[2] = HK_INVALID_CONTACT_POINT;
}

hkpCollisionAgent* HK_CALL hkpCapsuleCapsuleAgent::createCapsuleCapsuleAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
    return new hkpCapsuleCapsuleAgent( bodyA, bodyB, input, mgr );
}


void HK_CALL hkpCapsuleCapsuleAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createCapsuleCapsuleAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::CAPSULE, hkcdShapeType::CAPSULE);	
	}
}


// hkAgent interface implementation
void hkpCapsuleCapsuleAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	for (int i = 0; i < 3; i++ )
	{
		if(m_contactPointId[i] != HK_INVALID_CONTACT_POINT)
		{
			m_contactMgr->removeContactPoint(m_contactPointId[i], constraintOwner );
		}
	}
	delete this;
}



static HK_FORCE_INLINE void getThreeClosestPointsInl( const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkContactPoint* points, hkSimdRealParameter maxD )
{
	points[0].setDistanceSimdReal( maxD );
	points[1].setDistanceSimdReal( maxD );
	points[2].setDistanceSimdReal( maxD );

    const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*>(bodyA.getShape());
    const hkpCapsuleShape* capsuleB = static_cast<const hkpCapsuleShape*>(bodyB.getShape());
	hkVector4 pA[2]; hkVector4Util::transformPoints( bodyA.getTransform(), capsuleA->getVertices(), 2, pA );
	hkVector4 pB[2]; hkVector4Util::transformPoints( bodyB.getTransform(), capsuleB->getVertices(), 2, pB );

	hkLineSegmentUtil::capsuleCapsuleManifold( pA, capsuleA->getRadius(), pB, capsuleB->getRadius(), points );
}


static HK_FORCE_INLINE hkResult HK_CALL getClosestPointInternal(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkContactPoint& cpoint )
{
    const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*>(bodyA.getShape());
    const hkpCapsuleShape* capsuleB = static_cast<const hkpCapsuleShape*>(bodyB.getShape());

	hkVector4 pA[2]; hkVector4Util::transformPoints( bodyA.getTransform(), capsuleA->getVertices(), 2, pA );
	hkVector4 pB[2]; hkVector4Util::transformPoints( bodyB.getTransform(), capsuleB->getVertices(), 2, pB );

	return hkCollideCapsuleUtilClostestPointCapsVsCaps( pA, capsuleA->getRadius(), pB, capsuleB->getRadius(), input.getTolerance(), cpoint );
}





void hkpCapsuleCapsuleAgent::processCollision(const  hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x16962f6f,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("CapsCaps", HK_NULL);

	hkContactPoint points[3];
	hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref()));
	getThreeClosestPointsInl( bodyA, bodyB, input, points, inputTol );
	{
		for (int p = 0; p < 3; p++ )
		{
			if ( points[p].getDistanceSimdReal().isLess(inputTol) )
			{
				if(m_contactPointId[p] == HK_INVALID_CONTACT_POINT)
				{
					m_contactPointId[p] = m_contactMgr->addContactPoint(bodyA, bodyB, input, result, HK_NULL, points[p] );
				}

				if ( m_contactPointId[p] != HK_INVALID_CONTACT_POINT )
				{
					hkpProcessCdPoint& point = *result.reserveContactPoints(1);
					result.commitContactPoints(1);
					point.m_contact.setPosition(points[p].getPosition());
					point.m_contact.setSeparatingNormal(points[p].getSeparatingNormal());
					point.m_contactPointId = m_contactPointId[p];
				}
			}
			else
			{
				if(m_contactPointId[p] != HK_INVALID_CONTACT_POINT)
				{
					m_contactMgr->removeContactPoint( m_contactPointId[p], *result.m_constraintOwner.val() );
					m_contactPointId[p] = HK_INVALID_CONTACT_POINT;
				}

			}
		}
	}

	HK_TIMER_END();
}

void hkpCapsuleCapsuleAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("CapsCaps", HK_NULL);

	hkContactPoint contact;
	if ( getClosestPointInternal( bodyA, bodyB, input, contact) == HK_SUCCESS)
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}

void hkpCapsuleCapsuleAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("CapsCaps", HK_NULL);

	hkContactPoint contact;
	if (getClosestPointInternal( bodyA, bodyB, input, contact) == HK_SUCCESS)
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}
	
void hkpCapsuleCapsuleAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("CapsCaps", HK_NULL);

    const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*>(bodyA.getShape());
    const hkpCapsuleShape* capsuleB = static_cast<const hkpCapsuleShape*>(bodyB.getShape());

	hkVector4 pA[2]; hkVector4Util::transformPoints( bodyA.getTransform(), capsuleA->getVertices(), 2, pA );
	hkVector4 pB[2]; hkVector4Util::transformPoints( bodyB.getTransform(), capsuleB->getVertices(), 2, pB );

	hkVector4 dA; dA.setSub( pA[1], pA[0] );
	hkVector4 dB; dB.setSub( pB[1], pB[0] );

	hkLineSegmentUtil::ClosestLineSegLineSegResult result;
	hkLineSegmentUtil::closestLineSegLineSeg( pA[0], dA, pB[0], dB, result );

	hkReal radiusSum = capsuleA->getRadius() + capsuleB->getRadius();
	if ( result.m_distanceSquared < radiusSum * radiusSum )
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}

    HK_TIMER_END();
}

void hkpCapsuleCapsuleAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
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
