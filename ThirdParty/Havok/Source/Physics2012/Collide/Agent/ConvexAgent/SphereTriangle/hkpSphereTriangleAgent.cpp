/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Agent/ConvexAgent/SphereTriangle/hkpSphereTriangleAgent.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpSphereTriangleAgent::initAgentFunc(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc				= createSphereTriangleAgent;
	af.m_getPenetrationsFunc	= staticGetPenetrations;
	af.m_getClosestPointFunc	= staticGetClosestPoints;
	af.m_linearCastFunc			= staticLinearCast;
	af.m_isFlipped				= false;
	af.m_isPredictive			= false;
}


void HK_CALL hkpSphereTriangleAgent::initAgentFuncInverse(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc				= createTriangleSphereAgent;
	af.m_getPenetrationsFunc	= hkpSymmetricAgent<hkpSphereTriangleAgent>::staticGetPenetrations;
	af.m_getClosestPointFunc	= hkpSymmetricAgent<hkpSphereTriangleAgent>::staticGetClosestPoints;
	af.m_linearCastFunc			= hkpSymmetricAgent<hkpSphereTriangleAgent>::staticLinearCast;
	af.m_isFlipped				= true;
	af.m_isPredictive			= false;
}

#else

void HK_CALL hkpSphereTriangleAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc	= staticGetClosestPoints;
	af.m_linearCastFunc			= staticLinearCast;
}


void HK_CALL hkpSphereTriangleAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc	= hkpSymmetricAgent<hkpSphereTriangleAgent>::staticGetClosestPoints;
	af.m_linearCastFunc			= hkpSymmetricAgent<hkpSphereTriangleAgent>::staticLinearCast;
}

#endif


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpSphereTriangleAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFuncInverse(af);
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::TRIANGLE, hkcdShapeType::SPHERE);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFunc(af);
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::SPHERE, hkcdShapeType::TRIANGLE);
	}
}

void HK_CALL hkpSphereTriangleAgent::registerAgent2(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFuncInverse(af);
	    dispatcher->registerCollisionAgent2(af, hkcdShapeType::TRIANGLE, hkcdShapeType::SPHERE);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFunc(af);
	    dispatcher->registerCollisionAgent2(af, hkcdShapeType::SPHERE, hkcdShapeType::TRIANGLE);
	}
}

#endif


#if !defined(HK_PLATFORM_SPU)
hkpSphereTriangleAgent::hkpSphereTriangleAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
: hkpIterativeLinearCastAgent( mgr )
{
	m_contactPointId = HK_INVALID_CONTACT_POINT;
	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());
	hkpCollideTriangleUtil::setupClosestPointTriangleCache( triB->getVertices(), m_closestPointTriangleCache );
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpSphereTriangleAgent::createTriangleSphereAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpSphereTriangleAgent* agent = new hkpSymmetricAgentLinearCast<hkpSphereTriangleAgent>(bodyA, bodyB, input, mgr);
	return agent;
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpSphereTriangleAgent::createSphereTriangleAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
    return new hkpSphereTriangleAgent( bodyA, bodyB, input, mgr );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereTriangleAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	if(m_contactPointId != HK_INVALID_CONTACT_POINT)
	{
		m_contactMgr->removeContactPoint(m_contactPointId, constraintOwner );
	}
	delete this;
}
#endif


hkpSphereTriangleAgent::ClosestPointResult hkpSphereTriangleAgent::getClosestPointInl(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkpCollideTriangleUtil::ClosestPointTriangleCache& cache, hkContactPoint& cpoint , hkpFeatureOutput* featureOutput)
{
    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();

	hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), triB->getVertices(), 3, &triVertices[0]);

	hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;

	hkpCollideTriangleUtil::ClosestPointTriangleStatus res = hkpCollideTriangleUtil::closestPointTriangle( posA, &triVertices[0], cache, cptr , featureOutput);

	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + triB->getRadius());
	hkSimdReal triResult; triResult.load<1>((const hkReal*)&cptr.distance);

	if ( triResult < radiusSum + hkSimdReal::fromFloat(input.getTolerance()) )
	{
		hkVector4 cpPos; cpPos.setAddMul(  posA, cptr.hitDirection, hkSimdReal::fromFloat(triB->getRadius()) - triResult );		
		cpoint.setPosition(cpPos);
		cpoint.setSeparatingNormal( cptr.hitDirection, triResult - radiusSum );

		if ( res == hkpCollideTriangleUtil::HIT_TRIANGLE_FACE )
		{
			return ST_CP_FACE;
		}
		else
		{
			return ST_CP_EDGE;
		}
	}
	return ST_CP_MISS;
}


#if !defined(HK_PLATFORM_SPU)
hkpSphereTriangleAgent::ClosestPointResult HK_CALL hkpSphereTriangleAgent::getClosestPoint(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkpCollideTriangleUtil::ClosestPointTriangleCache& cache, hkContactPoint& cpoint )
{
	return getClosestPointInl( bodyA, bodyB, input, cache, cpoint );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereTriangleAgent::processCollision(const  hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x611dfe18,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("SphereTri", HK_NULL);

	hkpProcessCdPoint& point = *result.reserveContactPoints(1);

	hkpFeatureOutput featureOutput;
	if (getClosestPointInl( bodyA, bodyB, input, m_closestPointTriangleCache, point.m_contact , &featureOutput) != ST_CP_MISS)
	{	
		hkUint8	numFeatures	= featureOutput.numFeatures;
		// weld collision point normal
		if( numFeatures > 0 )
		{
			const hkpSphereShape*	sphereA			= static_cast<const hkpSphereShape*>(bodyA.getShape());
			const hkpTriangleShape* triB			= static_cast<const hkpTriangleShape*>(bodyB.getShape());
			hkVector4				weldedNormal	= point.m_contact.getSeparatingNormal();
			hkVector4				pointPos		= point.m_contact.getPosition();
			

			hkpConvexShape::WeldResult weldResult = (hkpConvexShape::WeldResult)triB->weldContactPoint(	featureOutput.featureIds , numFeatures , pointPos , 
				&bodyB.getTransform(), sphereA , &bodyA.getTransform(), weldedNormal );

			if (weldResult == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT)
			{
				result.abortContactPoints(1);
				if(m_contactPointId != HK_INVALID_CONTACT_POINT)
				{
					m_contactMgr->removeContactPoint(m_contactPointId, *result.m_constraintOwner.val() );
					m_contactPointId = HK_INVALID_CONTACT_POINT;
				}
				return;
			}
			else if (weldResult == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
			{
				point.m_contact.setNormalOnly( weldedNormal );			
			}
		}
		

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
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereTriangleAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("SphereTri", HK_NULL);

	hkContactPoint contact;
	if (getClosestPointInl( bodyA, bodyB, input, m_closestPointTriangleCache, contact))
	{
		hkpCdPoint event( bodyA, bodyB, contact );
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}
#endif


void hkpSphereTriangleAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIME_CODE_BLOCK("SphereTri", HK_NULL);
	
	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());
	
	hkpCollideTriangleUtil::ClosestPointTriangleCache cache;
	{
		hkpCollideTriangleUtil::setupClosestPointTriangleCache( triB->getVertices(), cache );
	}


	hkpFeatureOutput featureOutput;
	hkContactPoint contact;
	if (getClosestPointInl( bodyA, bodyB, input, cache, contact, &featureOutput))
	{
		hkpCdPoint event( bodyA, bodyB, contact );

		// weld closest point normal
		hkUint8 numFeatures = featureOutput.numFeatures;
		if( input.m_weldClosestPoints.val() && numFeatures > 0 )
		{
			const hkpSphereShape*	sphereA			= static_cast<const hkpSphereShape*>(bodyA.getShape());
			hkVector4				weldedNormal	= contact.getSeparatingNormal();
			hkVector4				pointPos		= contact.getPosition();			

			hkpConvexShape::WeldResult weldResult = (hkpConvexShape::WeldResult)triB->weldContactPoint(	featureOutput.featureIds , numFeatures , pointPos , 
				&bodyB.getTransform(), sphereA , &bodyA.getTransform(), weldedNormal );

			if (!input.m_forceAcceptContactPoints.val() && (weldResult == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT))
			{
				return;
			}
			else if(weldResult == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
			{
				event.setContactNormal( weldedNormal );
			}
		}

		collector.addCdPoint( event );
	}
}


#if !defined(HK_PLATFORM_SPU)
void hkpSphereTriangleAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("SphereTri", HK_NULL);

    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

    const hkVector4& posA = bodyA.getTransform().getTranslation();
	hkVector4 posAinB; posAinB._setTransformedInversePos( bodyB.getTransform(), posA );


	hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
	hkpCollideTriangleUtil::closestPointTriangle( posAinB, &triB->getVertex<0>(), m_closestPointTriangleCache, cptr );
	hkSimdReal triResult; triResult.load<1>((const hkReal*)&cptr.distance);

    const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + triB->getRadius());
	if( triResult.isLess(radiusSum) )
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}

    HK_TIMER_END();

}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpSphereTriangleAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("SphereTri", HK_NULL);

    const hkpSphereShape* sphereA = static_cast<const hkpSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	hkpCollideTriangleUtil::ClosestPointTriangleCache cache;
	hkpCollideTriangleUtil::setupClosestPointTriangleCache( triB->getVertices(), cache );

    const hkVector4& posA = bodyA.getTransform().getTranslation();
	hkVector4 posAinB; posAinB._setTransformedInversePos( bodyB.getTransform(), posA );


	hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
	hkpCollideTriangleUtil::closestPointTriangle( posAinB, &triB->getVertex<0>(), cache, cptr );
	hkSimdReal triResult; triResult.load<1>((const hkReal*)&cptr.distance);

	const hkSimdReal radiusSum = hkSimdReal::fromFloat(sphereA->getRadius() + triB->getRadius());
	if( triResult.isLess(radiusSum) )
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
