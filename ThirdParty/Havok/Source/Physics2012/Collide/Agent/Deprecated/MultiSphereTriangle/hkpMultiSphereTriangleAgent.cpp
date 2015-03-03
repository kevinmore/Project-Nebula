/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Agent/Deprecated/MultiSphereTriangle/hkpMultiSphereTriangleAgent.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>



//#define USE_ONE_SIDED_TRIANGLES
void HK_CALL hkpMultiSphereTriangleAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createTriangleMultiSphereAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpMultiSphereTriangleAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpMultiSphereTriangleAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpMultiSphereTriangleAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = false;
	    dispatcher->registerCollisionAgent( af, hkcdShapeType::TRIANGLE, hkcdShapeType::MULTI_SPHERE);	
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createMultiSphereTriangleAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = false;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MULTI_SPHERE, hkcdShapeType::TRIANGLE);	
	}    
}

hkpMultiSphereTriangleAgent::hkpMultiSphereTriangleAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* contactMgr )
: hkpIterativeLinearCastAgent( contactMgr )
{
	for (int i = 0; i < hkpMultiSphereShape::MAX_SPHERES; i++)
	{
		m_contactPointId[i] = HK_INVALID_CONTACT_POINT;
	}

    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	const hkVector4* vertices = triB->getVertices();
	hkpCollideTriangleUtil::setupClosestPointTriangleCache( &vertices[0], m_closestPointTriangleCache );
}

hkpCollisionAgent* HK_CALL hkpMultiSphereTriangleAgent::createTriangleMultiSphereAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
	return new hkpSymmetricAgentLinearCast<hkpMultiSphereTriangleAgent>(bodyA, bodyB, input, contactMgr);
}


hkpCollisionAgent* HK_CALL hkpMultiSphereTriangleAgent::createMultiSphereTriangleAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* contactMgr)
{
    return new hkpMultiSphereTriangleAgent( bodyA, bodyB, input, contactMgr );
}






void hkpMultiSphereTriangleAgent::cleanup(hkCollisionConstraintOwner& constraintOwner)
{
	for(int i = 0; i < hkpMultiSphereShape::MAX_SPHERES; ++i)
	{
		if(m_contactPointId[i] != HK_INVALID_CONTACT_POINT)
		{
			m_contactMgr->removeContactPoint(m_contactPointId[i], constraintOwner );
		}
	}

	delete this;
}

void hkpMultiSphereTriangleAgent::processCollision( const hkpCdBody& bodyA,  const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x70e70d18,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN("MultiSphereTri", HK_NULL);

    const hkpMultiSphereShape* msA = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	const hkVector4* vertices = triB->getVertices();

    hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), vertices, 3, triVertices );

	const hkVector4* localSpheres = msA->getSpheres();
	hkVector4 worldSpheres[hkpMultiSphereShape::MAX_SPHERES];

	const int nsphere = msA->getNumSpheres();

	hkVector4Util::transformPoints( bodyA.getTransform(), &localSpheres[0], nsphere, &worldSpheres[0]  );

#ifdef USE_ONE_SIDED_TRIANGLES
	hkVector4 edge0; edge0.setSub4(triVertices[1], triVertices[0]);
	hkVector4 edge1; edge1.setSub4(triVertices[2], triVertices[1]);
	hkVector4 trinormal; trinormal.setCross( edge0, edge1 );
	trinormal.mul4( m_invTriangleNormalLength );
#endif

	const hkVector4* curSphere = &worldSpheres[0];
	const hkVector4* localSphere = &localSpheres[0];


	for(int j = nsphere-1; j>=0 ; curSphere++,localSphere++,j--)
	{

#ifdef USE_ONE_SIDED_TRIANGLES
		{
			hkVector4 vec; vec.setSub4( triVertices[0], curSphere );
			hkReal dist = trinormal.dot3( vec );
			if( dist > 0 )
			{
				goto removeContactPoint;
			}
			if ( dist < -radiusSum )
			{
				goto removeContactPoint;
			}
		}
#endif
		const hkReal sphereRadius = (*localSphere)(3);
		const hkReal radiusSum = sphereRadius + triB->getRadius();
    
		{
			hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
			hkpCollideTriangleUtil::closestPointTriangle( *curSphere, &triVertices[0], m_closestPointTriangleCache, cptr );

			if ( cptr.distance < radiusSum + input.getTolerance() )
			{
				hkpProcessCdPoint& point = *result.reserveContactPoints(1);

				hkVector4 cpPos; cpPos.setAddMul(  *curSphere, cptr.hitDirection, hkSimdReal::fromFloat(triB->getRadius() - cptr.distance) );
				point.m_contact.setPosition(cpPos);
				point.m_contact.setSeparatingNormal( cptr.hitDirection );
				point.m_contact.setDistance( cptr.distance - radiusSum );

				if(m_contactPointId[j] == HK_INVALID_CONTACT_POINT)
				{
					m_contactPointId[j] = m_contactMgr->addContactPoint(bodyA, bodyB, input, result, HK_NULL, point.m_contact );
				}

				if ( m_contactPointId[j] != HK_INVALID_CONTACT_POINT )
				{
					result.commitContactPoints(1);
					point.m_contactPointId = m_contactPointId[j];
				}
				else
				{
					result.abortContactPoints(1);
				}

				continue;
			}
		}
		{
#ifdef USE_ONE_SIDED_TRIANGLES
	removeContactPoint:
#endif
			if(m_contactPointId[j] != HK_INVALID_CONTACT_POINT)
			{
				m_contactMgr->removeContactPoint(m_contactPointId[j], *result.m_constraintOwner.val() );
				m_contactPointId[j] = HK_INVALID_CONTACT_POINT;
			}
		}
	}

	HK_TIMER_END();
}

// hkpCollisionAgent interface implementation.
void hkpMultiSphereTriangleAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("MultiSphereTriangle", HK_NULL);

    const hkpMultiSphereShape* msA = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	const hkVector4* vertices = triB->getVertices();

    hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), vertices, 3, triVertices );

	const hkVector4* localSpheres = msA->getSpheres();
	hkVector4 worldSpheres[hkpMultiSphereShape::MAX_SPHERES];
	const int nsphere = msA->getNumSpheres();

	hkVector4Util::transformPoints( bodyA.getTransform(), &localSpheres[0], nsphere, &worldSpheres[0]  );

	//HK_ASSERT(0x6294297a, localSpheres.getSize() == m_contactPointId.getSize() );


	for(int i = 0; i < nsphere; ++i)
	{
		const hkVector4& curSphere = worldSpheres[i];

		const hkReal sphereRadius = localSpheres[i](3);
		const hkReal radiusSum = sphereRadius + triB->getRadius();

		{
			hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
			hkpCollideTriangleUtil::closestPointTriangle( curSphere, &triVertices[0], m_closestPointTriangleCache, cptr );

			if ( cptr.distance < radiusSum + input.getTolerance() )
			{
				hkContactPoint contact;
				hkVector4 cpPos; cpPos.setAddMul(  curSphere, cptr.hitDirection, hkSimdReal::fromFloat(-cptr.distance + triB->getRadius()));
				contact.setPosition(cpPos);
				contact.setSeparatingNormal( cptr.hitDirection, cptr.distance - radiusSum );
				hkpCdPoint event( bodyA, bodyB, contact );
				collector.addCdPoint( event );
			}
		}
	}

	HK_TIMER_END();
}

// hkpCollisionAgent interface implementation.
void hkpMultiSphereTriangleAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	HK_TIMER_BEGIN("MultiSphereTriangle", HK_NULL);

    const hkpMultiSphereShape* msA = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	const hkVector4* vertices = triB->getVertices();

    hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), vertices, 3, triVertices );

	const hkVector4* localSpheres = msA->getSpheres();
	hkVector4 worldSpheres[hkpMultiSphereShape::MAX_SPHERES];
	const int nsphere = msA->getNumSpheres();

	hkVector4Util::transformPoints( bodyA.getTransform(), &localSpheres[0], nsphere, &worldSpheres[0]  );


	hkpCollideTriangleUtil::ClosestPointTriangleCache closestPointTriangleCache;
	{
		const hkVector4* verticesB = triB->getVertices();
		hkpCollideTriangleUtil::setupClosestPointTriangleCache( &verticesB[0], closestPointTriangleCache );
	}

	for(int i = 0; i < nsphere; ++i)
	{
		const hkVector4& curSphere = worldSpheres[i];

		const hkReal sphereRadius = localSpheres[i](3);
		const hkReal radiusSum = sphereRadius + triB->getRadius();

		{
			hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
			hkpCollideTriangleUtil::closestPointTriangle( curSphere, &triVertices[0], closestPointTriangleCache, cptr );

			if ( cptr.distance < radiusSum + input.getTolerance() )
			{

				hkContactPoint contact;
				hkVector4 cpPos; cpPos.setAddMul(  curSphere, cptr.hitDirection, hkSimdReal::fromFloat(triB->getRadius() -cptr.distance) );
				contact.setPosition(cpPos);
				contact.setSeparatingNormal(  cptr.hitDirection, cptr.distance - radiusSum );
				hkpCdPoint event( bodyA, bodyB, contact );
				collector.addCdPoint( event );
			}
		}
	}

	HK_TIMER_END();
}
	

void hkpMultiSphereTriangleAgent::getPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{

    const hkpMultiSphereShape* msA = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	const hkVector4* vertices = triB->getVertices();

    hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), vertices, 3, triVertices );

	const hkVector4* localSpheres = msA->getSpheres();
	hkVector4 worldSpheres[hkpMultiSphereShape::MAX_SPHERES];
	const int nsphere = msA->getNumSpheres();

	hkVector4Util::transformPoints( bodyA.getTransform(), &localSpheres[0], nsphere, &worldSpheres[0]  );

	hkpCdPoint event( bodyA, bodyB );

	for(int i = 0; i < nsphere; ++i)
	{
		const hkVector4& curSphere = worldSpheres[i];

		const hkReal sphereRadius = localSpheres[i](3);
		const hkReal radiusSum = sphereRadius + triB->getRadius();

		{
			hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
			hkpCollideTriangleUtil::closestPointTriangle( curSphere, &triVertices[0], m_closestPointTriangleCache, cptr );

			if ( cptr.distance < radiusSum )
			{
				collector.addCdBodyPair( bodyA, bodyB );
				break;
			}
		}
	}
}

void hkpMultiSphereTriangleAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{

    const hkpMultiSphereShape* msA = static_cast<const hkpMultiSphereShape*>(bodyA.getShape());
    const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(bodyB.getShape());

	const hkVector4* vertices = triB->getVertices();

    hkVector4 triVertices[3];
	hkVector4Util::transformPoints( bodyB.getTransform(), vertices, 3, triVertices );

	const hkVector4* localSpheres = msA->getSpheres();
	hkVector4 worldSpheres[hkpMultiSphereShape::MAX_SPHERES];
	const int nsphere = msA->getNumSpheres();

	hkVector4Util::transformPoints( bodyA.getTransform(), &localSpheres[0], nsphere, &worldSpheres[0]  );

	hkpCollideTriangleUtil::ClosestPointTriangleCache closestPointTriangleCache;
	{
		const hkVector4* verticesB = triB->getVertices();
		hkpCollideTriangleUtil::setupClosestPointTriangleCache( &verticesB[0], closestPointTriangleCache );
	}

	hkpCdPoint event( bodyA, bodyB );

	for(int i = 0; i < nsphere; ++i)
	{
		const hkVector4& curSphere = worldSpheres[i];

		const hkReal sphereRadius = localSpheres[i](3);
		const hkReal radiusSum = sphereRadius + triB->getRadius();

		{
			hkpCollideTriangleUtil::ClosestPointTriangleResult cptr;
			hkpCollideTriangleUtil::closestPointTriangle( curSphere, &triVertices[0], closestPointTriangleCache, cptr );

			if ( cptr.distance < radiusSum )
			{
				collector.addCdBodyPair( bodyA, bodyB );
				break;
			}
		}
	}
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
