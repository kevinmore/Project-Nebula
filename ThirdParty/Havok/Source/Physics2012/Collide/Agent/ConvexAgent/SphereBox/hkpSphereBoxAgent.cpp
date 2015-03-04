/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereBox/hkpSphereBoxAgent.h>

#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

void HK_CALL hkpSphereBoxAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createBoxSphereAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpSphereBoxAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpSphereBoxAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpSphereBoxAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::BOX, hkcdShapeType::SPHERE);	
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createSphereBoxAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::SPHERE, hkcdShapeType::BOX);	
	}

}



hkpCollisionAgent* HK_CALL hkpSphereBoxAgent::createBoxSphereAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpSphereBoxAgent* agent = new hkpSymmetricAgentLinearCast<hkpSphereBoxAgent>(mgr);
	return agent;
}


hkpCollisionAgent* HK_CALL hkpSphereBoxAgent::createSphereBoxAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
    return new hkpSphereBoxAgent( mgr );
}


void hkpSphereBoxAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	if(m_contactPointId != HK_INVALID_CONTACT_POINT)
	{
		m_contactMgr->removeContactPoint(m_contactPointId, constraintOwner );
	}

	delete this;
}



hkBool  hkpSphereBoxAgent::getClosestPoint(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,  const hkpCollisionInput& input, hkContactPoint& contactOut)
{

    const hkVector4& posA = bodyA.getTransform().getTranslation();
	hkVector4 posLocalB; posLocalB._setTransformedInversePos( bodyB.getTransform(), posA );

    const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());
    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());

	hkVector4 absPosB; absPosB.setAbs( posLocalB );

	hkVector4 clippedAbsPos; clippedAbsPos.setMin( absPosB, boxB->getHalfExtents() );
	hkVector4 delta; delta.setSub( clippedAbsPos, absPosB );	

	const hkSimdReal boxPlusSphere = hkSimdReal::fromFloat(sphereA->getRadius() + boxB->getRadius());
	hkSimdReal distance;
	if ( delta.lessZero().anyIsSet<hkVector4ComparisonMask::MASK_XYZ>() )
	{
		//
		// Now we are outside
		//
		distance = delta.normalizeWithLength<3>() - boxPlusSphere;
		hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref()));
		if ( distance.isGreater(inputTol) )
		{
			return false;
		}
		delta.setFlipSign(delta, posLocalB);
		delta.setNeg<4>( delta );
		hkVector4 cpN; cpN._setRotatedDir( bodyB.getTransform().getRotation(), delta );
		contactOut.setSeparatingNormal(cpN, distance);
	}
	else
	{
		//
		// Completely inside, search the smallest penetration
		//
		hkVector4 dd; dd.setSub(absPosB, boxB->getHalfExtents());

		int i = dd.getIndexOfMaxComponent<3>();

		distance = dd.horizontalMax<3>() - boxPlusSphere; // indep max calc
		hkVector4 cpN; cpN.setFlipSign( bodyB.getTransform().getColumn(i), posLocalB.getComponent(i).lessZero() );
		contactOut.setSeparatingNormal(cpN, distance);
	}

	hkVector4 cpPos; cpPos.setAddMul( posA, contactOut.getNormal(), -distance - hkSimdReal::fromFloat(sphereA->getRadius()) );
	cpPos.setW(hkSimdReal_1); // for determinism checks.
	contactOut.setPosition(cpPos);

	return true;
}



void hkpSphereBoxAgent::processCollision(const  hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x4944b7cf,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("SphereBox", HK_NULL);

	hkpProcessCdPoint& point = *result.reserveContactPoints(1);

	if (getClosestPoint( bodyA, bodyB, input, point.m_contact))
	{
		if(m_contactPointId == HK_INVALID_CONTACT_POINT)
		{
			m_contactPointId = m_contactMgr->addContactPoint(bodyA, bodyB, input, result, HK_NULL, point.m_contact );
		}

		if (m_contactPointId != HK_INVALID_CONTACT_POINT)
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

void hkpSphereBoxAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereBox", HK_NULL);

	hkContactPoint contact;
	if (getClosestPoint( bodyA, bodyB, input, contact))
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}

void hkpSphereBoxAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereBox", HK_NULL);

	hkContactPoint contact;
	if (getClosestPoint( bodyA, bodyB, input, contact))
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}
	
void hkpSphereBoxAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("SphereBox", HK_NULL);

    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();

	hkVector4 posLocalB; posLocalB._setTransformedInversePos( bodyB.getTransform(), posA );
	hkVector4 absPosB; absPosB.setAbs( posLocalB );
	absPosB.setMin( absPosB, boxB->getHalfExtents() );
	absPosB.setFlipSign(absPosB, posLocalB);
	hkVector4 posB; posB._setTransformedPos( bodyB.getTransform(), absPosB );

    hkVector4 vec;    vec.setSub( posB, posA );

    const hkSimdReal distSquared = vec.dot<3>(vec);
	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + boxB->getRadius());
	if (distSquared.isLess(radiusSum * radiusSum))
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}
 
    HK_TIMER_END();

}

void hkpSphereBoxAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
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
