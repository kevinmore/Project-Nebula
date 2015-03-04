/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Misc/MultiRay/hkpMultiRayShape.h>
#include <Physics2012/Collide/Agent/MiscAgent/MultirayConvex/hkpMultiRayConvexAgent.h>
#include <Physics2012/Collide/Agent/Query/hkpCdPointCollector.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>


void HK_CALL hkpMultiRayConvexAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createConvexMultiRayAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgentLinearCast<hkpMultiRayConvexAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgentLinearCast<hkpMultiRayConvexAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgentLinearCast<hkpMultiRayConvexAgent>::staticLinearCast;
		af.m_isFlipped           = true;
			// the agent is not really predictive, however there is no fallback available
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent( af, hkcdShapeType::CONVEX, hkcdShapeType::MULTI_RAY );	
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createMultiRayConvexAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
			// the agent is not really predictive, however there is no fallback available
		af.m_isPredictive		 = true;
	    dispatcher->registerCollisionAgent( af, hkcdShapeType::MULTI_RAY, hkcdShapeType::CONVEX );	
	}
}


hkpMultiRayConvexAgent::hkpMultiRayConvexAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
: hkpIterativeLinearCastAgent(mgr)
{
	const hkpMultiRayShape* msA = static_cast<const hkpMultiRayShape*>(bodyA.getShape());
	int nRay = msA->getRays().getSize();

	m_contactInfo.setSize(nRay);
	for (int i = 0; i < nRay;i++) 
	{
		m_contactInfo[i].m_contactPointId = HK_INVALID_CONTACT_POINT;
	}
}



hkpCollisionAgent* HK_CALL hkpMultiRayConvexAgent::createConvexMultiRayAgent(const hkpCdBody& bodyA,const  hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpSymmetricAgentLinearCast<hkpMultiRayConvexAgent>(bodyA, bodyB, input, mgr);
}


hkpCollisionAgent* HK_CALL hkpMultiRayConvexAgent::createMultiRayConvexAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
    return new hkpMultiRayConvexAgent( bodyA, bodyB, input, mgr );
}



void hkpMultiRayConvexAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	int nRay = m_contactInfo.getSize();
	for(int i = 0; i < nRay; ++i)
	{
		if(m_contactInfo[i].m_contactPointId != HK_INVALID_CONTACT_POINT)
		{
			m_contactMgr->removeContactPoint(m_contactInfo[i].m_contactPointId, constraintOwner );
		}
	}

	delete this;
}


void hkpMultiRayConvexAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector  )
{
	HK_TIMER_BEGIN("multiRay-cvx", HK_NULL);

    const hkpMultiRayShape* msA = static_cast<const hkpMultiRayShape*>(bodyA.getShape());
    const hkpConvexShape*  convexB = static_cast<const hkpConvexShape*>(bodyB.getShape());

	hkTransform bTa;	bTa.setMulInverseMul( bodyB.getTransform(), bodyA.getTransform());

	const hkArray<hkpMultiRayShape::Ray>& RaysA = msA->getRays();

	int nRays = RaysA.getSize();

	const hkpMultiRayShape::Ray* rayA = RaysA.begin();


	hkpShapeRayCastInput rayInput;
	for (int i = 0; i < nRays; i++)
	{
		hkpShapeRayCastOutput rayResults;

		rayInput.m_from._setTransformedPos( bTa, rayA->m_start );
		rayInput.m_to._setTransformedPos( bTa, rayA->m_end );
		
		hkBool rayHit = convexB->castRay( rayInput, rayResults);
		if ( rayHit )
		{
			const hkVector4& normal = rayResults.m_normal;

			hkSimdReal dist; dist.load<1>(&rayResults.m_hitFraction);
			hkVector4 hitpoint;	hitpoint.setInterpolate(rayInput.m_from,rayInput.m_to, dist );	

			hkContactPoint contact;
			hkVector4 cpPos; cpPos._setTransformedPos( bodyB.getTransform(), hitpoint );
			contact.setPosition(cpPos);
			hkVector4 cpN; cpN._setRotatedDir( bodyB.getTransform().getRotation(), normal);
			contact.setSeparatingNormal(cpN, (dist-hkSimdReal_1) * rayA->m_start.getW() + hkSimdReal::fromFloat(msA->getRayPenetrationDistance()) );// + input.m_tolerance;

			hkpCdPoint event( bodyA, bodyB, contact );
			collector.addCdPoint(event);
		}
		rayA++;
	}

	HK_TIMER_END();
}

void hkpMultiRayConvexAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector  )
{
	HK_TIMER_BEGIN("multiRay-cvx", HK_NULL);

    const hkpMultiRayShape* msA = static_cast<const hkpMultiRayShape*>(bodyA.getShape());
    const hkpConvexShape*  convexB = static_cast<const hkpConvexShape*>(bodyB.getShape());

	hkTransform bTa;	bTa.setMulInverseMul( bodyB.getTransform(), bodyA.getTransform());

	const hkArray<hkpMultiRayShape::Ray>& RaysA = msA->getRays();

	int nRays = RaysA.getSize();

	const hkpMultiRayShape::Ray* rayA = RaysA.begin();


	hkpShapeRayCastInput rayInput;
	for (int i = 0; i < nRays; i++)
	{
		hkpShapeRayCastOutput rayResults;

		rayInput.m_from._setTransformedPos( bTa, rayA->m_start );
		rayInput.m_to._setTransformedPos( bTa, rayA->m_end );
		
		hkBool rayHit = convexB->castRay( rayInput, rayResults);
		if ( rayHit )
		{
			const hkVector4& normal = rayResults.m_normal;

			hkSimdReal dist; dist.load<1>(&rayResults.m_hitFraction);
			hkVector4 hitpoint;	hitpoint.setInterpolate(rayInput.m_from,rayInput.m_to, dist );	

			hkContactPoint contact;
			hkVector4 cpPos; cpPos._setTransformedPos( bodyB.getTransform(), hitpoint );
			contact.setPosition(cpPos);
			hkVector4 cpN; cpN._setRotatedDir( bodyB.getTransform().getRotation(), normal);
			contact.setSeparatingNormal(cpN, (dist-hkSimdReal_1) * rayA->m_start.getW() + hkSimdReal::fromFloat(msA->getRayPenetrationDistance()) );// + input.m_tolerance;
			hkpCdPoint event( bodyA, bodyB, contact );
			collector.addCdPoint(event);
		}
		rayA++;
	}

	HK_TIMER_END();
}
	
void hkpMultiRayConvexAgent::processCollision( const hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x4e7eca78,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("multiRay-cvx", HK_NULL);
    
	const hkpMultiRayShape* msA = static_cast<const hkpMultiRayShape*>(bodyA.getShape());
    const hkpConvexShape*  convexB = static_cast<const hkpConvexShape*>(bodyB.getShape());

	hkTransform bTa;	bTa.setMulInverseMul( bodyB.getTransform(), bodyA.getTransform() );

	const hkArray<hkpMultiRayShape::Ray>& RaysA = msA->getRays();

	int nRays = RaysA.getSize();

	hkpShapeRayCastInput rayInput;

	const hkpMultiRayShape::Ray* rayA = RaysA.begin();
	for (int i = 0; i < nRays; i++)
	{
		rayInput.m_from._setTransformedPos( bTa, rayA->m_start );
		rayInput.m_to._setTransformedPos( bTa, rayA->m_end );

		hkpShapeRayCastOutput rayResults;
		hkBool rayHit = convexB->castRay( rayInput, rayResults);
		if ( !rayHit )
		{
			if(m_contactInfo[i].m_contactPointId != HK_INVALID_CONTACT_POINT)
			{
				m_contactMgr->removeContactPoint(m_contactInfo[i].m_contactPointId, *result.m_constraintOwner.val() );
				m_contactInfo[i].m_contactPointId = HK_INVALID_CONTACT_POINT;
			}
		}
		else
		{
			hkpProcessCdPoint& point = *result.reserveContactPoints(1);
			const hkVector4& normal = rayResults.m_normal;

			hkSimdReal dist; dist.load<1>(&rayResults.m_hitFraction);
			hkVector4 hitpoint;	hitpoint.setInterpolate(rayInput.m_from,rayInput.m_to, dist);
			hkVector4 cpPos; cpPos.setTransformedPos( bodyB.getTransform(), hitpoint );
			point.m_contact.setPosition(cpPos);

			hkVector4 cpN; cpN._setRotatedDir( bodyB.getTransform().getRotation(), normal);
			point.m_contact.setSeparatingNormal(cpN, (dist - hkSimdReal_1) * rayA->m_start.getW() + hkSimdReal::fromFloat(msA->getRayPenetrationDistance()));// + input.m_tolerance;

			if( m_contactInfo[i].m_contactPointId == HK_INVALID_CONTACT_POINT)
			{
				m_contactInfo[i].m_contactPointId = m_contactMgr->addContactPoint(bodyA, bodyB, input, result, HK_NULL, point.m_contact );
			}

			if ( m_contactInfo[i].m_contactPointId != HK_INVALID_CONTACT_POINT )
			{
				result.commitContactPoints(1);
				point.m_contactPointId = m_contactInfo[i].m_contactPointId;
			}
			else
			{
				result.abortContactPoints(1);
			}			
		}
		rayA++;
	}

	HK_TIMER_END();
}



void hkpMultiRayConvexAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	staticGetPenetrations(bodyA, bodyB, input, collector);
}

void hkpMultiRayConvexAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("multiRay-cnvx-getPen", HK_NULL);

	const hkpMultiRayShape*	rayShape	= static_cast<const hkpMultiRayShape*>(bodyA.getShape());
	const hkpConvexShape*	convexShape = static_cast<const hkpConvexShape*>(bodyB.getShape());

	hkTransform convexTRay;
	convexTRay.setMulInverseMul( bodyB.getTransform(), bodyA.getTransform() );

	int rayCount = rayShape->getRays().getSize();

	const hkpMultiRayShape::Ray* ray = rayShape->getRays().begin();

	hkpShapeRayCastInput		rayInput;
	hkpShapeRayCastOutput	rayOuput;

	// Cast each ray in the multi-ray shape against the convex
	// shape. Each time a hit found, add it to the collector.
	// When the collectors early out flag is set, stop.
	while (rayCount-- && !collector.getEarlyOut())
	{		
		rayInput.m_from._setTransformedPos( convexTRay, ray->m_start );
		rayInput.m_to._setTransformedPos(   convexTRay, ray->m_end );

		if ( convexShape->castRay( rayInput, rayOuput) ) // If hit
		{
			collector.addCdBodyPair(bodyA, bodyB); // Collect the hit
		}

		ray++; // Get the next ray
	}

	HK_TIMER_END();
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
