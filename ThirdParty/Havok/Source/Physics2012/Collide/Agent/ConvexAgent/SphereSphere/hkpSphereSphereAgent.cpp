/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>

#include <Physics2012/Collide/Agent/ConvexAgent/SphereSphere/hkpSphereSphereAgent.h>


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpSphereSphereAgent::initAgentFunc(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc				= createSphereSphereAgent;
	af.m_getPenetrationsFunc	= staticGetPenetrations;
	af.m_getClosestPointFunc	= staticGetClosestPoints;
	af.m_linearCastFunc			= staticLinearCast;
}

#else

void HK_CALL hkpSphereSphereAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc	= staticGetClosestPoints;
	af.m_linearCastFunc			= staticLinearCast;
}

#endif


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpSphereSphereAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::AgentFuncs af;
	initAgentFunc(af);
    dispatcher->registerCollisionAgent(af, hkcdShapeType::SPHERE, hkcdShapeType::SPHERE);	
}

void HK_CALL hkpSphereSphereAgent::registerAgent2(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::AgentFuncs af;
	initAgentFunc(af);
    dispatcher->registerCollisionAgent2(af, hkcdShapeType::SPHERE, hkcdShapeType::SPHERE);	
}

#endif


#if !defined(HK_PLATFORM_SPU)
hkpSphereSphereAgent::hkpSphereSphereAgent( hkpContactMgr* contactMgr): hkpIterativeLinearCastAgent( contactMgr )
{
	m_contactPointId = HK_INVALID_CONTACT_POINT;
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpSphereSphereAgent::createSphereSphereAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, 
                                                                const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
    return new hkpSphereSphereAgent( contactMgr );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereSphereAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	if(m_contactPointId != HK_INVALID_CONTACT_POINT)
	{
		m_contactMgr->removeContactPoint(m_contactPointId, constraintOwner );
		m_contactPointId = HK_INVALID_CONTACT_POINT;
	}

	delete this;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereSphereAgent::processCollision( const hkpCdBody& bodyA,  const hkpCdBody& bodyB, 
                                            const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x5103c8d8,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );
	HK_TIMER_BEGIN("SphereSphere", HK_NULL);

	const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpSphereShape* sphereB = static_cast<const hkpSphereShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();
    const hkVector4& posB = bodyB.getTransform().getTranslation();

    hkVector4 vec;    vec.setSub( posA, posB );

    const hkSimdReal distSquared = vec.dot<3>(vec);
	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + sphereB->getRadius());
	hkSimdReal radiusSum2; radiusSum2.load<1>(&(input.m_tolerance.ref())); radiusSum2.add(radiusSum); radiusSum2.mul(radiusSum2);

    if ( distSquared.isLess(radiusSum2) )
    {
		hkpProcessCdPoint& point = *result.reserveContactPoints(1);

#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		const hkVector4Comparison gt0 = distSquared.greaterZero();

		const hkSimdReal invDist = distSquared.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		hkVector4 cpN; 
		cpN.setMul( invDist, vec );
		cpN.setSelect(gt0, cpN, hkVector4::getConstant<HK_QUADREAL_1000>());

		hkSimdReal distance;
		distance.setMul(distSquared, invDist);
		distance.zeroIfFalse(gt0);
		distance.sub(radiusSum);

		point.m_contact.setSeparatingNormal(cpN, distance);
#else
		if ( distSquared.isGreaterZero() )
		{
			const hkSimdReal invDist = distSquared.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
			hkVector4 cpN; cpN.setMul( invDist, vec );
			point.m_contact.setSeparatingNormal(cpN, distSquared * invDist - radiusSum);
		}
		else
		{
			point.m_contact.setSeparatingNormal(hkVector4::getConstant<HK_QUADREAL_1000>(), -radiusSum);
		}
#endif
		hkVector4 cpPos; cpPos.setAddMul(  posB, point.m_contact.getNormal(), hkSimdReal::fromFloat(sphereB->getRadius()) );
		point.m_contact.setPosition(cpPos);

		if(m_contactPointId == HK_INVALID_CONTACT_POINT)
		{
			m_contactPointId = m_contactMgr->addContactPoint(bodyA, bodyB, input, result, HK_NULL, point.m_contact );
		}
		if (m_contactPointId != HK_INVALID_CONTACT_POINT )
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
		if(m_contactPointId != HK_INVALID_CONTACT_POINT)
		{
			m_contactMgr->removeContactPoint(m_contactPointId, *result.m_constraintOwner.val() );
			m_contactPointId = HK_INVALID_CONTACT_POINT;
		}
    }

	HK_TIMER_END();
}
#endif


hkBool hkpSphereSphereAgent::getClosestPoint( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkContactPoint& contact )
{
	const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpSphereShape* sphereB = static_cast<const hkpSphereShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();
    const hkVector4& posB = bodyB.getTransform().getTranslation();

    hkVector4 vec;    vec.setSub( posA, posB );

    const hkSimdReal distSquared = vec.dot<3>(vec);
	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + sphereB->getRadius());
	hkSimdReal radiusSum2; radiusSum2.load<1>(&(input.m_tolerance.ref())); radiusSum2.add(radiusSum); radiusSum2.mul(radiusSum2);

    if ( distSquared.isLess(radiusSum2) )
    {
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
		const hkVector4Comparison gt0 = distSquared.greaterZero();

		const hkSimdReal invDist = distSquared.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		hkVector4 cpN; 
		cpN.setMul( invDist, vec );
		cpN.setSelect(gt0, cpN, hkVector4::getConstant<HK_QUADREAL_1000>());

		hkSimdReal distance;
		distance.setMul(distSquared, invDist);
		distance.zeroIfFalse(gt0);
		distance.sub(radiusSum);

		contact.setSeparatingNormal(cpN, distance);
#else
		if (distSquared.isGreaterZero())
		{
	        const hkSimdReal invDist = distSquared.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
			hkVector4 sepN; sepN.setMul( invDist, vec );
			contact.setSeparatingNormal( sepN, distSquared * invDist - radiusSum );
		}
		else
		{
			contact.setSeparatingNormal( hkVector4::getConstant<HK_QUADREAL_1000>(), -radiusSum );
		}
#endif
		hkVector4 cpPos; cpPos.setAddMul(  posB, contact.getNormal(), hkSimdReal::fromFloat(sphereB->getRadius()) );
		contact.setPosition(cpPos);
		return true;
    }
	return false;
}


#if !defined(HK_PLATFORM_SPU)
void hkpSphereSphereAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereSphere", HK_NULL);

	hkContactPoint contact;
	if (getClosestPoint( bodyA, bodyB, input, contact ))
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint(event);
	}
	HK_TIMER_END();
}
#endif


void hkpSphereSphereAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereSphere", HK_NULL);

	hkContactPoint contact;
	if ( getClosestPoint( bodyA, bodyB, input, contact ) )
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint(event);
	}
	HK_TIMER_END();
}


#if !defined(HK_PLATFORM_SPU)
void hkpSphereSphereAgent::getPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	staticGetPenetrations(bodyA, bodyB, input, collector);
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereSphereAgent::staticGetPenetrations( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("SphereSphere", HK_NULL);

    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpSphereShape* sphereB = static_cast<const hkpSphereShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();
    const hkVector4& posB = bodyB.getTransform().getTranslation();

    hkVector4 vec;
	vec.setSub( posB, posA );

    const hkSimdReal distSquared = vec.dot<3>(vec);
	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + sphereB->getRadius());

	if( distSquared.isLess(radiusSum * radiusSum) )
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}

	HK_TIMER_END();
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
