/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>


#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>

#include <Physics2012/Collide/Util/hkpCollideTriangleUtil.h>
#include <Physics2012/Collide/Util/hkpCollideCapsuleUtil.h>
#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>

#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent3/CapsuleTriangle/hkpCapsuleTriangleAgent3.h>

#define HK_THIS_AGENT_SIZE HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpCapsuleTriangleCache3) )
HK_COMPILE_TIME_ASSERT(HK_THIS_AGENT_SIZE <= hkAgent3::MAX_NET_SIZE);

void hkCapsuleTriangleAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = sepNormal;
	f.m_cleanupFunc  = cleanup;
	f.m_removePointFunc  = removePoint;
	f.m_commitPotentialFunc  = commitPotential;
	f.m_createZombieFunc  = createZombie;
	f.m_updateFilterFunc = HK_NULL;
	//invalidateTim
	//warpTime
	f.m_destroyFunc  = destroy;
	f.m_isPredictive = false;

	dispatcher->registerAgent3( f, hkcdShapeType::CAPSULE, hkcdShapeType::TRIANGLE );

}


hkpAgentData* hkCapsuleTriangleAgent3::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );

	//const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(input.m_bodyA->getShape());
	const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(input.m_bodyB->getShape());

	entry->m_numContactPoints = 0;

	capsTriCache->m_contactPointId[0] = HK_INVALID_CONTACT_POINT;
	capsTriCache->m_contactPointId[1] = HK_INVALID_CONTACT_POINT;
	capsTriCache->m_contactPointId[2] = HK_INVALID_CONTACT_POINT;

	const hkpTriangleShape* triB = static_cast<const hkpTriangleShape*>(shapeB);
	hkpCollideTriangleUtil::setupPointTriangleDistanceCache( triB->getVertices(), capsTriCache->m_triangleCache );

	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}

HK_DISABLE_OPTIMIZATION_VS2008_X64
void hkCapsuleTriangleAgent3::sepNormal( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4& separatingNormalOut )
{
	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );

	const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*> (input.m_bodyA->getShape());
	const hkpTriangleShape* triB    = static_cast<const hkpTriangleShape*>(input.m_bodyB->getShape());

	hkVector4 endPoints[2];
	hkVector4 triVertices[3];

	hkVector4Util::transformPoints( input.m_bodyA->getTransform(), capsuleA->getVertices(), 2, endPoints );
	hkVector4Util::transformPoints( input.m_bodyB->getTransform(), triB->getVertices(), 3, &triVertices[0]);

	hkContactPoint points[3];
	int searchManifold = false;

	hkCollideCapsuleUtilCapsVsTri( endPoints, capsuleA->getRadius(), triVertices, triB->getRadius(), capsTriCache->m_triangleCache, HK_REAL_MAX, searchManifold, points);

	const hkSimdReal dist0 = points[0].getDistanceSimdReal();
	const hkSimdReal dist1 = points[1].getDistanceSimdReal();
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	separatingNormalOut.setSelect(dist0.less(dist1), points[0].getSeparatingNormal(), points[1].getSeparatingNormal());
#else
	if ( dist0 < dist1 )
	{
		separatingNormalOut = points[0].getSeparatingNormal();
	}
	else
	{
		separatingNormalOut = points[1].getSeparatingNormal();
	}
#endif
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

hkpAgentData* hkCapsuleTriangleAgent3::cleanup ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );
	for (int i = 0; i < 3; i++ )
	{
		if(capsTriCache->m_contactPointId[i] != HK_INVALID_CONTACT_POINT)
		{
			mgr->removeContactPoint( capsTriCache->m_contactPointId[i], constraintOwner );
			capsTriCache->m_contactPointId[i] = HK_INVALID_CONTACT_POINT;
		}
	}
	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}

void    hkCapsuleTriangleAgent3::removePoint ( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToRemove )
{
	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );
	for ( int i = 0; i < 3; i++)
	{
		if ( capsTriCache->m_contactPointId[i] == idToRemove)
		{
			capsTriCache->m_contactPointId[i] = HK_INVALID_CONTACT_POINT;
			break;
		}
	}
}

void    hkCapsuleTriangleAgent3::commitPotential( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToCommit )
{
	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );
	for ( int i = 0; i < 3; i++)
	{
		if ( capsTriCache->m_contactPointId[i] == HK_INVALID_CONTACT_POINT)
		{
			capsTriCache->m_contactPointId[i] = idToCommit;
			break;
		}
	}
}

void	hkCapsuleTriangleAgent3::createZombie( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToConvert )
{
	return;
	//	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );
	//	for ( int i = 0; i < 3; i++)
	//	{
	//		if ( capsTriCache->m_contactPointId[i] == idToConvert)
	//		{
	//			cp.m_dimA = 0;
	//			cp.m_dimB = 0;
	//			break;
	//		}
	//	}
}


void  hkCapsuleTriangleAgent3::destroy ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher )
{
	cleanup(entry, agentData, mgr, constraintOwner );
}





//HK_COMPILE_TIME_ASSERT( sizeof(hkpCapsuleTriangleCache3) == 16*5 );

hkpAgentData* hkCapsuleTriangleAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_TIMER_BEGIN("CapsTri3", this );

	//
	//	Calc the relative movement for this timestep
	//
	//hkReal distAtT1 = input.m_distAtT1;
	hkpCapsuleTriangleCache3* capsTriCache = reinterpret_cast<hkpCapsuleTriangleCache3*>( agentData );	

	int numContacts = 0;

	const hkpCapsuleShape* capsuleA = static_cast<const hkpCapsuleShape*> (input.m_bodyA->getShape());
	const hkpTriangleShape* triB    = static_cast<const hkpTriangleShape*>(input.m_bodyB->getShape());

	hkVector4 endPoints[2];
	hkVector4 triVertices[3];

	hkVector4Util::transformPoints( input.m_bodyA->getTransform(), capsuleA->getVertices(), 2, endPoints );
	hkVector4Util::transformPoints( input.m_bodyB->getTransform(), triB->getVertices(), 3, &triVertices[0]);

	hkContactPoint points[3];
	int searchManifold = true;

	hkpFeatureOutput featuresOut[3];
	hkCollideCapsuleUtilCapsVsTri( endPoints, capsuleA->getRadius(), triVertices, triB->getRadius(), capsTriCache->m_triangleCache, HK_REAL_MAX, searchManifold, points , featuresOut);


	const hkSimdReal dist0 = points[0].getDistanceSimdReal();
	const hkSimdReal dist1 = points[1].getDistanceSimdReal();

	int referenceIndex = dist0.isLess(dist1) ? 0 : 1;

	{
		hkSimdReal inputTol; inputTol.load<1>(&(input.m_input->m_tolerance.ref()));
		for (int p = 0; p < 3; p++ )
		{
			hkContactPointId cpId = capsTriCache->m_contactPointId[p];
			if ( points[p].getDistanceSimdReal().isLess(inputTol) )
			{
				if( cpId == HK_INVALID_CONTACT_POINT)
				{
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
					if ( output.m_potentialContacts )
					{
						hkResult canReserve = input.m_contactMgr->reserveContactPoints(1);
						if ( canReserve == HK_SUCCESS )
						{
							hkpProcessCollisionOutput::ContactRef& cr = *(output.m_potentialContacts->m_firstFreePotentialContact++);
							cr.m_contactPoint = output.m_firstFreeContactPoint;
							cr.m_agentData = agentData;
							cr.m_agentEntry = entry;
							goto exportContactPoint;
						}
					}
					else
#endif
					{
						cpId = input.m_contactMgr->addContactPoint( *input.m_bodyA, *input.m_bodyB, *input.m_input, output, HK_NULL, points[p] );
					}
				}

				if ( cpId != HK_INVALID_CONTACT_POINT )
				{
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
	exportContactPoint:
#endif

					// weld collision point normal
					hkUint8 numFeaturePoints = featuresOut[p].numFeatures;
					if( numFeaturePoints > 0 )
					{
						hkVector4 weldedNormal = points[p].getSeparatingNormal();
						hkVector4 cpPos = points[p].getPosition();
						hkpConvexShape::WeldResult weldResult = (hkpConvexShape::WeldResult)triB->weldContactPoint(	featuresOut[p].featureIds , numFeaturePoints , cpPos , 
							&input.m_bodyB->getTransform(), capsuleA , &input.m_bodyA->getTransform(), weldedNormal );
						points[p].setPosition(cpPos);

						if (weldResult == hkpConvexShape::WELD_RESULT_REJECT_CONTACT_POINT)
						{	
							if(cpId != HK_INVALID_CONTACT_POINT)
							{
								input.m_contactMgr->removeContactPoint( cpId, *output.m_constraintOwner.val() );
								capsTriCache->m_contactPointId[p] = HK_INVALID_CONTACT_POINT;
							}
							continue;
						}
						else if(weldResult == hkpConvexShape::WELD_RESULT_ACCEPT_CONTACT_POINT_MODIFIED)
						{
							points[p].setNormalOnly(weldedNormal);
						}
					}

					numContacts++;
					hkpProcessCdPoint& point = *output.reserveContactPoints(1);
					output.commitContactPoints(1);
					point.m_contact.setPosition(points[p].getPosition());
					point.m_contact.setSeparatingNormal(points[p].getSeparatingNormal());
					point.m_contactPointId = cpId;
#if defined(HK_1N_MACHINE_SUPPORTS_WELDING)
					if ( p == referenceIndex && output.m_potentialContacts )
					{
						*(output.m_potentialContacts->m_firstFreeRepresentativeContact++) = &point;
					}
#endif
				}
			}
			else
			{
				if(cpId != HK_INVALID_CONTACT_POINT)
				{
					input.m_contactMgr->removeContactPoint( cpId, *output.m_constraintOwner.val() );
					cpId = HK_INVALID_CONTACT_POINT;
				}
			}
			capsTriCache->m_contactPointId[p] = cpId;
		}
	}

	separatingNormal[0] = points[referenceIndex].getSeparatingNormal();

	entry->m_numContactPoints = hkUchar(numContacts);
	HK_TIMER_END();

	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
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
