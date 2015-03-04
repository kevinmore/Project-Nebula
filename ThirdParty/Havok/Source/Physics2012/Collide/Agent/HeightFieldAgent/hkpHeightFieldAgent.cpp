/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Shape/Query/hkpRayShapeCollectionFilter.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpRayHitCollector.h>
#include <Physics2012/Collide/Shape/HeightField/hkpSphereRepShape.h>
#include <Physics2012/Collide/Shape/HeightField/hkpHeightFieldShape.h>
#include <Physics2012/Collide/Agent/HeightFieldAgent/hkpHeightFieldAgent.h>

#include <Common/Base/Config/hkOptionalComponent.h>

HK_OPTIONAL_COMPONENT_DEFINE(hkpHeightFieldAgent, hkpCollisionAgent::registerHeightFieldAgent, hkpHeightFieldAgent::registerAgent);

hkpHeightFieldAgent::hkpHeightFieldAgent(const hkpCdBody& A, const hkpCdBody& B, const hkpCollisionInput& input, hkpContactMgr* mgr)
: hkpCollisionAgent( mgr )
{
	// find the maximum number of collision spheres and initialize the m_contactPointId array
	if ( mgr )
	{
		const hkpSphereRepShape* csShape = static_cast<const hkpSphereRepShape*>(A.getShape());

		int numSpheres = csShape->getNumCollisionSpheres( );
		m_contactPointId.setSize( numSpheres, HK_INVALID_CONTACT_POINT );
	}
}


void HK_CALL hkpHeightFieldAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	HK_OPTIONAL_COMPONENT_MARK_USED(hkpHeightFieldAgent);

	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createHeightFieldAAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpHeightFieldAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpHeightFieldAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpHeightFieldAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;	// the heightfield agent is non really predictive, but we do not have an alternative
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::HEIGHT_FIELD, hkcdShapeType::SPHERE_REP);	
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createHeightFieldBAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
	    dispatcher->registerCollisionAgent(af, hkcdShapeType::SPHERE_REP, hkcdShapeType::HEIGHT_FIELD);	
	}
}


hkpCollisionAgent* hkpHeightFieldAgent::createHeightFieldBAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
															  const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpHeightFieldAgent* agent = new hkpHeightFieldAgent(bodyA, bodyB, input, mgr);
	return agent;
}


hkpCollisionAgent* hkpHeightFieldAgent::createHeightFieldAAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
															  const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpHeightFieldAgent* agent = new hkpSymmetricAgent<hkpHeightFieldAgent>(bodyA, bodyB, input, mgr);
	return agent;
}


void hkpHeightFieldAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	if ( m_contactMgr )
	{
		// Remove any unneeded contact points
		for (int i=0; i< m_contactPointId.getSize(); i++)
		{
			if (m_contactPointId[i] != HK_INVALID_CONTACT_POINT)
			{
				// Remove from contact manager
				m_contactMgr->removeContactPoint(m_contactPointId[i], constraintOwner );
			}
		}
	}
	delete this;
}

void hkpHeightFieldAgent::processCollision(	const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& processOutput)
{
	HK_ASSERT2(0x7371164d,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIME_CODE_BLOCK( "HeightField", HK_NULL );

	const hkpSphereRepShape* csShape = static_cast<const hkpSphereRepShape*>(csBody.getShape());
	const hkpHeightFieldShape* hfShape      = static_cast<const hkpHeightFieldShape*>(hfBody.getShape());

	// Get the relative transform for the two bodies for the collision detector
	hkTransform bTa; bTa.setMulInverseMul( hfBody.getTransform(), csBody.getTransform() );

	const int numSpheres = m_contactPointId.getSize();

	hkSphere* sphereBuffer = hkAllocateStack<hkSphere>(numSpheres);

	hkContactPointId* cpId = m_contactPointId.begin();

	  //
	  //	Get the collision spheres in CollisionSphereSpace and transform them into heightfield space
	  //
	{
		const hkSphere* spheres = csShape->getCollisionSpheres( &sphereBuffer[0] );
		hkVector4Util::transformSpheres( bTa, &spheres[0].getPosition(), numSpheres, const_cast<hkVector4*>(&sphereBuffer->getPosition()) );
	}

		//
		// collide
		//
	hkpHeightFieldShape::SphereCollisionOutput* out2 = hkAllocateStack<hkpHeightFieldShape::SphereCollisionOutput>(numSpheres);
	{
		hkpHeightFieldShape::CollideSpheresInput in2;
		in2.m_spheres = &sphereBuffer[0];
		in2.m_numSpheres = numSpheres;
		in2.m_tolerance = input.getTolerance();

		hfShape->collideSpheres( in2, &out2[0] );
	}

	//
	//	examine results
	//
	{
		hkpHeightFieldShape::SphereCollisionOutput* outP = &out2[0];
		hkSphere*   sphereP = &sphereBuffer[0];

		hkSimdReal tolerance; tolerance.load<1>(&(input.m_tolerance.ref()));
		for (int i = numSpheres-1; i>=0; i--)
		{
			if ( outP[0].getW().isGreater(tolerance) )
			{
				if ( cpId[0] != HK_INVALID_CONTACT_POINT)
				{
					// Remove from contact manager
					m_contactMgr->removeContactPoint(cpId[0], *processOutput.m_constraintOwner.val() );

					// Mark this point as INVALID
					cpId[0] = HK_INVALID_CONTACT_POINT;
				}
			}
			else
			{
				// Add point to manifold
				hkpProcessCdPoint& point = *processOutput.reserveContactPoints(1);

				hkVector4 position; position.setAddMul( sphereP->getPosition(), outP[0], -sphereP->getRadiusSimdReal() );

				hkVector4 cpPos; cpPos._setTransformedPos( hfBody.getTransform(), position );
				point.m_contact.setPosition(cpPos);
				hkVector4 cpN; cpN._setRotatedDir( hfBody.getTransform().getRotation(), outP[0] );
				point.m_contact.setSeparatingNormal(cpN, outP[0].getW() );

							// If this point does not already exist
				if(*cpId == HK_INVALID_CONTACT_POINT)
				{
					// Add it to the contact manager
					*cpId = m_contactMgr->addContactPoint(csBody, hfBody, input, processOutput, HK_NULL, point.m_contact );
					if(*cpId == HK_INVALID_CONTACT_POINT)
					{
						processOutput.abortContactPoints(1);
					}
					else
					{
						processOutput.commitContactPoints(1);
					}
				}
				else
				{
					processOutput.commitContactPoints(1);
				}
				// Update ID
				point.m_contactPointId = *cpId;
			}
			sphereP++;
			outP++;
			cpId++;
		}
	}
	hkDeallocateStack<hkpHeightFieldShape::SphereCollisionOutput>(out2, numSpheres);
	hkDeallocateStack<hkSphere>(sphereBuffer, numSpheres);
}


			// hkpCollisionAgent interface implementation.
void hkpHeightFieldAgent::getPenetrations(const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	staticGetPenetrations( csBody, hfBody, input, collector );
}

			// hkpCollisionAgent interface implementation.
void HK_CALL hkpHeightFieldAgent::staticGetPenetrations(const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_TIMER_BEGIN_LIST( "HeightField", "GetSpheres" );

	const hkpSphereRepShape* csShape = static_cast<const hkpSphereRepShape*>(csBody.getShape());
	const hkpHeightFieldShape* hfShape      = static_cast<const hkpHeightFieldShape*>(hfBody.getShape());

	// Get the relative transform for the two bodies for the collision detector
	hkTransform bTa; bTa.setMulInverseMul( hfBody.getTransform(), csBody.getTransform() );

	const int numSpheres = csShape->getNumCollisionSpheres( );
	hkSphere* sphereBuffer = hkAllocateStack<hkSphere>(numSpheres);


	  //
	  //	Get the collision spheres in CollisionSphereSpace and transform them into heightfield space
	  //
	{
		HK_INTERNAL_TIMER_SPLIT_LIST("getSpheres");
		const hkSphere* spheres = csShape->getCollisionSpheres( &sphereBuffer[0] );

		HK_INTERNAL_TIMER_SPLIT_LIST("transform");
		hkVector4Util::transformSpheres( bTa, &spheres[0].getPosition(), numSpheres, reinterpret_cast<hkVector4*>(&sphereBuffer[0]) );
	}

		//
		// collide
		//
	HK_TIMER_SPLIT_LIST("Collide");
	hkpHeightFieldShape::SphereCollisionOutput* out2 = hkAllocateStack<hkpHeightFieldShape::SphereCollisionOutput>(numSpheres);
	{
		hkpHeightFieldShape::CollideSpheresInput in2;
		in2.m_spheres = &sphereBuffer[0];
		in2.m_numSpheres = numSpheres;
		in2.m_tolerance = input.getTolerance();

		hfShape->collideSpheres( in2, &out2[0] );
	}

	//
	//	examine results
	//
	HK_TIMER_SPLIT_LIST("Examine");
	{
		hkpHeightFieldShape::SphereCollisionOutput* outP = &out2[0];
		for (int i = numSpheres-1; i>=0; i--)
		{
			if ( outP[0](3) < 0.0f )
			{
				collector.addCdBodyPair( csBody, hfBody );
				break;
			}
			outP++;
		}
	}
	hkDeallocateStack<hkpHeightFieldShape::SphereCollisionOutput>(out2, numSpheres);
	hkDeallocateStack<hkSphere>(sphereBuffer, numSpheres);
	HK_TIMER_END_LIST();
}

			// hkpCollisionAgent interface implementation.
void hkpHeightFieldAgent::getClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdPointCollector& collector)
{
	staticGetClosestPoints( bodyA, bodyB, input, collector );
}
			
			// hkpCollisionAgent interface implementation.
void HK_CALL hkpHeightFieldAgent::staticGetClosestPoints( const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpCollisionInput& input, class hkpCdPointCollector& collector  )
{
	HK_TIMER_BEGIN_LIST( "HeightField", "bTA" );

	const hkpSphereRepShape* csShape = static_cast<const hkpSphereRepShape*>(csBody.getShape());
	const hkpHeightFieldShape* hfShape      = static_cast<const hkpHeightFieldShape*>(hfBody.getShape());

	// Get the relative transform for the two bodies for the collision detector
	hkTransform bTa; bTa.setMulInverseMul( hfBody.getTransform(), csBody.getTransform() );

	const int numSpheres = csShape->getNumCollisionSpheres( );
	hkSphere* sphereBuffer = hkAllocateStack<hkSphere>(numSpheres);


	  //
	  //	Get the collision spheres in CollisionSphereSpace and transform them into heightfield space
	  //
	{
		HK_TIMER_SPLIT_LIST("getSpheres");
		const hkSphere* spheres = csShape->getCollisionSpheres( &sphereBuffer[0] );

		HK_TIMER_SPLIT_LIST("transform");
		hkVector4Util::transformSpheres( bTa, &spheres[0].getPosition(), numSpheres, reinterpret_cast<hkVector4*>(&sphereBuffer[0]) );
	}

		//
		// collide
		//
	HK_TIMER_SPLIT_LIST("collide");
	hkpHeightFieldShape::SphereCollisionOutput* out2 = hkAllocateStack<hkpHeightFieldShape::SphereCollisionOutput>(numSpheres);
	{
		hkpHeightFieldShape::CollideSpheresInput in2;
		in2.m_spheres = &sphereBuffer[0];
		in2.m_numSpheres = numSpheres;
		in2.m_tolerance = input.getTolerance();

		hfShape->collideSpheres( in2, &out2[0] );
	}

		//
		//	examine results
		//
	HK_TIMER_SPLIT_LIST("examine");
	{
		hkpHeightFieldShape::SphereCollisionOutput* outP = &out2[0];
		hkSphere*   sphereP = &sphereBuffer[0];

		hkSimdReal tolerance; tolerance.load<1>(&(input.m_tolerance.ref()));
		for (int i = numSpheres-1; i>=0; i--)
		{
			if ( outP[0].getW().isLessEqual(tolerance) )
			{
				hkVector4 position; position.setAddMul( sphereP->getPosition(), outP[0], -sphereP->getRadiusSimdReal() - outP[0].getW() );

				hkContactPoint contact;
				hkVector4 cpPos; cpPos._setTransformedPos( hfBody.getTransform(), position );
				contact.setPosition(cpPos);
				hkVector4 cpN; cpN._setRotatedDir( hfBody.getTransform().getRotation(), outP[0] );
				contact.setSeparatingNormal(cpN, outP[0].getW() );

				hkpCdPoint point( csBody, hfBody, contact );
				collector.addCdPoint( point );
			}
			sphereP++;
			outP++;
		}
	}
	hkDeallocateStack<hkpHeightFieldShape::SphereCollisionOutput>(out2, numSpheres);
	hkDeallocateStack<hkSphere>(sphereBuffer, numSpheres);
	HK_TIMER_END_LIST();

}

class hkHeightFieldRayForwardingCollector : public hkpRayHitCollector
{
	public:
		virtual void addRayHit( const hkpCdBody& cdBody, const hkpShapeRayCastCollectorOutput& hitInfo )
		{
			hkContactPoint contact;
			hkVector4 cpPos; cpPos.setAddMul( m_currentFrom, m_path, hkSimdReal::fromFloat(hitInfo.m_hitFraction) );
			hkVector4 cpN; cpN._setRotatedDir( cdBody.getTransform().getRotation(), hitInfo.m_normal );
			cpPos.addMul( hkSimdReal::fromFloat(-m_currentRadius), cpN);
			contact.setPosition(cpPos);
			contact.setSeparatingNormal(cpN);
			contact.setDistance( hitInfo.m_hitFraction );

			hkpCdPoint point( m_csBody, cdBody, contact );
			m_collector.addCdPoint( point );

			m_earlyOutHitFraction = hkMath::min2( m_collector.getEarlyOutDistance(), m_earlyOutHitFraction);
		}

		hkHeightFieldRayForwardingCollector( const hkpCdBody& csBody, const hkVector4& path, hkpCdPointCollector& collector )
			: m_path(path), m_csBody( csBody), m_collector( collector )
		{

		}

		hkVector4 m_currentFrom;
		hkReal    m_currentRadius;
		hkVector4 m_path;

		const hkpCdBody&  m_csBody;
		hkpCdPointCollector& m_collector;

};


void hkpHeightFieldAgent::staticLinearCast( const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_TIMER_BEGIN_LIST( "HeightField", "ClosestPoints" );
	if ( startCollector )
	{
		staticGetClosestPoints( csBody, hfBody, input, *startCollector );
	}

	HK_TIMER_SPLIT_LIST("GetSpheres");

	const hkpSphereRepShape* csShape			= static_cast<const hkpSphereRepShape*>(csBody.getShape());
	const hkpHeightFieldShape* hfShape       = static_cast<const hkpHeightFieldShape*>(hfBody.getShape());

	// Get the relative transform for the two bodies for the collision detector
	hkTransform bTa; bTa.setMulInverseMul( hfBody.getTransform(), csBody.getTransform() );

	const int numSpheres = csShape->getNumCollisionSpheres( );
	hkSphere* sphereBuffer = hkAllocateStack<hkSphere>(numSpheres);

	hkVector4 pathLocal; pathLocal._setRotatedInverseDir( hfBody.getTransform().getRotation(), input.m_path );

	  //
	  //	Get the collision spheres in CollisionSphereSpace and transform them into heightfield space
	  //
	const hkSphere* spheres = csShape->getCollisionSpheres( &sphereBuffer[0] );

		// Cast each sphere
	{
		HK_TIMER_SPLIT_LIST("CastSpheres");
		hkpHeightFieldShape::hkpSphereCastInput ray;
		ray.m_maxExtraPenetration = input.m_maxExtraPenetration;

		hkHeightFieldRayForwardingCollector rayCollector( csBody, input.m_path, collector );

		for (int i = 0; i < numSpheres; i++ )
		{
			ray.m_from._setTransformedPos( bTa, spheres->getPosition() );
			ray.m_radius = spheres->getRadius();
			ray.m_to.setAdd( ray.m_from, pathLocal );

			// The adapter needs these values to work out the actual position
			rayCollector.m_currentFrom._setTransformedPos( csBody.getTransform(), spheres->getPosition());
			rayCollector.m_currentRadius = spheres->getRadius();

			hfShape->castSphere( ray, hfBody, rayCollector );
			spheres++;
		}
	}

	hkDeallocateStack<hkSphere>(sphereBuffer, numSpheres);
	HK_TIMER_END_LIST();
}

void hkpHeightFieldAgent::linearCast( const hkpCdBody& csBody, const hkpCdBody& hfBody, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	staticLinearCast( csBody, hfBody, input, collector, startCollector );
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
