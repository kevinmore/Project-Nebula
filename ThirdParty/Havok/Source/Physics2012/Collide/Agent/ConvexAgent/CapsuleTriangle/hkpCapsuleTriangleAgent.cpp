/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Agent/ConvexAgent/CapsuleTriangle/hkpCapsuleTriangleAgent.h>
#include <Physics2012/Collide/Util/hkpCollideCapsuleUtil.h>


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpCapsuleTriangleAgent::initAgentFunc(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc          = createCapsuleTriangleAgent;
	af.m_getPenetrationsFunc = staticGetPenetrations;
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc      = staticLinearCast;
	af.m_isFlipped           = false;
	af.m_isPredictive		 = false;
}


void HK_CALL hkpCapsuleTriangleAgent::initAgentFuncInverse(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc          = createTriangleCapsuleAgent;
	af.m_getPenetrationsFunc = hkpSymmetricAgent<hkpCapsuleTriangleAgent>::staticGetPenetrations;
	af.m_getClosestPointFunc = hkpSymmetricAgent<hkpCapsuleTriangleAgent>::staticGetClosestPoints;
	af.m_linearCastFunc      = hkpSymmetricAgent<hkpCapsuleTriangleAgent>::staticLinearCast;
	af.m_isFlipped           = true;
	af.m_isPredictive		 = false;
}

#else

void HK_CALL hkpCapsuleTriangleAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc	 = staticLinearCast;
}


void HK_CALL hkpCapsuleTriangleAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpCapsuleTriangleAgent>::staticGetClosestPoints;
	af.m_linearCastFunc	  = hkpSymmetricAgent<hkpCapsuleTriangleAgent>::staticLinearCast;
}

#endif


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpCapsuleTriangleAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFuncInverse(af);
		dispatcher->registerCollisionAgent(af, hkcdShapeType::TRIANGLE, hkcdShapeType::CAPSULE);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFunc(af);
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CAPSULE, hkcdShapeType::TRIANGLE);
	}
}

void HK_CALL hkpCapsuleTriangleAgent::registerAgent2(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFuncInverse(af);
		dispatcher->registerCollisionAgent2(af, hkcdShapeType::TRIANGLE, hkcdShapeType::CAPSULE);
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		initAgentFunc(af);
		dispatcher->registerCollisionAgent2(af, hkcdShapeType::CAPSULE, hkcdShapeType::TRIANGLE);
	}
}

#endif


#if !defined(HK_PLATFORM_SPU)
hkpCapsuleTriangleAgent::hkpCapsuleTriangleAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
: hkpIterativeLinearCastAgent( mgr )
{
	m_contactPointId[0] = HK_INVALID_CONTACT_POINT;
	m_contactPointId[1] = HK_INVALID_CONTACT_POINT;
	m_contactPointId[2] = HK_INVALID_CONTACT_POINT;
	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());
	hkpCollideTriangleUtil::setupPointTriangleDistanceCache( triB->getVertices(), m_triangleCache );
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpCapsuleTriangleAgent::createTriangleCapsuleAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpCapsuleTriangleAgent* agent = new hkpSymmetricAgentLinearCast<hkpCapsuleTriangleAgent>(bodyA, bodyB, input, mgr);
	return agent;
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpCapsuleTriangleAgent::createCapsuleTriangleAgent(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpCapsuleTriangleAgent( bodyA, bodyB, input, mgr );
}
#endif


#if !defined(HK_PLATFORM_SPU)
// hkAgent interface implementation
void hkpCapsuleTriangleAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
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
#endif


// note the searchManifold parameter had to be made int due to an Internal Compiler Error in gcc 2.95.3 when using hkBool
void hkpCapsuleTriangleAgent::getClosestPointsInl( const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCollideTriangleUtil::PointTriangleDistanceCache& cache, int searchManifold, hkContactPoint* points , hkpFeatureOutput* featureOutput)
{
	const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*>(bodyA.getShape());
	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	hkVector4 endPoints[2];

	hkVector4Util::transformPoints( bodyA.getTransform(), capsuleA->getVertices(), 2, endPoints );

	hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), triB->getVertices(), 3, &triVertices[0]);

 	hkCollideCapsuleUtilCapsVsTri( endPoints, capsuleA->getRadius(), triVertices, triB->getRadius(), cache, input.getTolerance(), searchManifold, points , featureOutput );
}


#if !defined(HK_PLATFORM_SPU)
// note the searchManifold parameter had to be made int due to an Internal Compiler Error in gcc 2.95.3 when using hkBool
void hkpCapsuleTriangleAgent::getClosestPointsPublic( const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCollideTriangleUtil::PointTriangleDistanceCache& cache, int searchManifold, hkContactPoint* points )
{	
	getClosestPointsInl( bodyA, bodyB, input, cache, searchManifold, points);
}
#endif


hkpCapsuleTriangleAgent::ClosestPointResult hkpCapsuleTriangleAgent::getClosestPointInternal(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkpCollideTriangleUtil::PointTriangleDistanceCache& cache, hkpCdPoint& cpoint )
{
	const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*>(bodyA.getShape());
	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	hkContactPoint points[3];
	hkpFeatureOutput featureOutput[3];
	getClosestPointsInl( bodyA, bodyB, input, cache, false, points , featureOutput );

	int Id = -1;
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkSimdReal dist0 = points[0].getDistanceSimdReal();
	hkSimdReal dist1 = points[1].getDistanceSimdReal();
	hkSimdReal tol; tol.load<1>(&(input.m_tolerance.ref()));

	hkVector4Comparison d0LTd1 = dist0.less(dist1);
	hkVector4Comparison d0LTtol = dist0.less(tol);
	hkVector4Comparison d1LTtol = dist1.less(tol);

	hkVector4Comparison valid0; valid0.setAnd   (d0LTtol, d0LTd1);
	hkVector4Comparison valid1; valid1.setAndNot(d1LTtol, d0LTd1);

	Id = valid0.anyIsSet() ? 0 : (valid1.anyIsSet() ? 1 : Id);
#else
	hkReal dist0 = points[0].getDistance();
	hkReal dist1 = points[1].getDistance();

	if ( dist0 < dist1 )
	{
		if ( dist0 < input.getTolerance() )
		{
			Id = 0;
		}
	}
	else
	{
		if ( dist1 < input.getTolerance() )
		{
			Id = 1;
		}
	}

#endif
	if( Id != -1 )
	{
		// weld closest point normal 
		hkVector4 unweldedNormal = points[Id].getSeparatingNormal();
		hkUint8 numFeaturePoints = featureOutput[Id].numFeatures;
		if( input.m_weldClosestPoints.val() && numFeaturePoints > 0 )
		{
			hkVector4 weldedNormal = unweldedNormal;
			hkVector4 cpPos = points[Id].getPosition();
			hkpConvexShape::WeldResult result = (hkpConvexShape::WeldResult)triB->weldContactPoint(	featureOutput[Id].featureIds , numFeaturePoints , cpPos , 
				&bodyB.getTransform(), capsuleA , &bodyA.getTransform(), weldedNormal );
			points[Id].setPosition(cpPos);
			
			if (!input.m_forceAcceptContactPoints.val() && (result == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT))
			{						
				return ST_CP_MISS;
			}
 			else if(result == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
 			{
 				points[Id].setNormalOnly(weldedNormal);
 			}
		}

		cpoint.setContact(points[Id]);
		cpoint.setUnweldedNormal(unweldedNormal);
		return ST_CP_HIT;
	}

	return ST_CP_MISS;
}


#if !defined(HK_PLATFORM_SPU)
hkpCapsuleTriangleAgent::ClosestPointResult HK_CALL hkpCapsuleTriangleAgent::getClosestPoint(const  hkpCdBody& bodyA, const hkpCdBody& bodyB,	const hkpCollisionInput& input, hkpCollideTriangleUtil::PointTriangleDistanceCache& cache, hkContactPoint& cpoint )
{
	hkContactPoint contact;
	contact.setPosition(hkVector4::getZero());
	contact.setSeparatingNormal(hkVector4::getZero());
	hkpCdPoint event(bodyA, bodyB, contact);

	hkpCapsuleTriangleAgent::ClosestPointResult res = getClosestPointInternal( bodyA, bodyB, input, cache, event );
	cpoint = event.getContact();
	return res;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpCapsuleTriangleAgent::processCollision(const  hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x4d200eea,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("CapsuleTri", HK_NULL);

	const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*>(bodyA.getShape());
	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	hkContactPoint points[3];
	hkpFeatureOutput featureOutput[3];
  	getClosestPointsInl( bodyA, bodyB, input, m_triangleCache, true, points , featureOutput);
	{
		hkSimdReal inputTol; inputTol.load<1>(&(input.m_tolerance.ref()));
		for (int p = 0; p < 3; p++ )
		{
			if ( points[p].getDistanceSimdReal().isLess(inputTol) )
			{
				// weld collision point normal
				hkUint8 numFeaturePoints = featureOutput[p].numFeatures;
				if( numFeaturePoints > 0 )
				{					
					hkVector4 weldedNormal = points[p].getSeparatingNormal();
					hkVector4 cpPos = points[p].getPosition();
					hkpConvexShape::WeldResult weldResult = (hkpConvexShape::WeldResult)triB->weldContactPoint(	featureOutput[p].featureIds , numFeaturePoints , cpPos , 
						&bodyB.getTransform(), capsuleA , &bodyA.getTransform(), weldedNormal );
					points[p].setPosition(cpPos);

					if (weldResult == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT)
					{	
						if(m_contactPointId[p] != HK_INVALID_CONTACT_POINT)
						{
							m_contactMgr->removeContactPoint( m_contactPointId[p], *result.m_constraintOwner.val() );
							m_contactPointId[p] = HK_INVALID_CONTACT_POINT;
						}
						continue;
					}
					else if(weldResult == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
					{
						points[p].setNormalOnly(weldedNormal);
					}
				}

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
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpCapsuleTriangleAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("CapsTriangle", HK_NULL);

	hkContactPoint contact;
	contact.setPosition(hkVector4::getZero());
	contact.setSeparatingNormal(hkVector4::getZero());
	hkpCdPoint event( bodyA, bodyB, contact );
	if (getClosestPointInternal( bodyA, bodyB, input, m_triangleCache, event))
	{
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}
#endif


void hkpCapsuleTriangleAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("CapsTriangle", HK_NULL);

	hkpCollideTriangleUtil::PointTriangleDistanceCache cache;
	{
		const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());
		hkpCollideTriangleUtil::setupPointTriangleDistanceCache( triB->getVertices(), cache );
	}

	hkContactPoint contact;
	contact.setPosition(hkVector4::getZero());
	contact.setSeparatingNormal(hkVector4::getZero());
	hkpCdPoint event( bodyA, bodyB, contact );

	if (getClosestPointInternal( bodyA, bodyB, input, cache, event))
	{		
		collector.addCdPoint( event );
	}

	HK_TIMER_END();
}


#if !defined(HK_PLATFORM_SPU)
void hkpCapsuleTriangleAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("CapsTriangle", HK_NULL);
	hkContactPoint points[3];
	getClosestPointsInl( bodyA, bodyB, input, m_triangleCache, false, points );

	const hkSimdReal dist0 = points[0].getDistanceSimdReal();
	const hkSimdReal dist1 = points[1].getDistanceSimdReal();
	hkVector4Comparison either; either.setOr(dist0.lessZero(), dist1.lessZero());
	if ( either.anyIsSet() )
	{
		collector.addCdBodyPair( bodyA, bodyB );
	}

	HK_TIMER_END();

}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpCapsuleTriangleAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN("CapsTriangle", HK_NULL);
	hkpCollideTriangleUtil::PointTriangleDistanceCache cache;
	{
		const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());
		hkpCollideTriangleUtil::setupPointTriangleDistanceCache( triB->getVertices(), cache );
	}
	hkContactPoint points[3];
	getClosestPointsInl( bodyA, bodyB, input, cache, false, points);

	const hkSimdReal dist0 = points[0].getDistanceSimdReal();
	const hkSimdReal dist1 = points[1].getDistanceSimdReal();
	hkVector4Comparison either; either.setOr(dist0.lessZero(), dist1.lessZero());
	if ( either.anyIsSet() )
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
