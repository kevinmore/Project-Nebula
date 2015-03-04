/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
//#include <hkmath/linear/hkSweptTransformUtil.h>
#include <Physics2012/Collide/BoxBox/hkpBoxBoxCollisionDetection.h>
#include <Physics2012/Collide/BoxBox/hkpBoxBoxContactPoint.h>

#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionOutput.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent3/BoxBox/hkpBoxBoxAgent3.h>
#if defined(HK_PLATFORM_SPU)
#	include <Physics2012/Collide/Agent/ContactMgr/hkpContactMgrSpu.inl> // include this after we include the actual contact manager!
#endif

#define HK_THIS_AGENT_SIZE HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hkpBoxBoxManifold) )
HK_COMPILE_TIME_ASSERT(HK_THIS_AGENT_SIZE <= hkAgent3::MAX_NET_SIZE);

void hkBoxBoxAgent3::initAgentFunc(hkpCollisionDispatcher::Agent3Funcs& f)
{
	f.m_createFunc   = create;
	f.m_processFunc  = process;
	f.m_sepNormalFunc = HK_NULL; //sepNormal;
	f.m_cleanupFunc  = cleanup;
#if !defined(HK_PLATFORM_SPU)
	f.m_removePointFunc  = removePoint;
	f.m_commitPotentialFunc  = commitPotential;
	f.m_createZombieFunc  = createZombie;
	f.m_updateFilterFunc = HK_NULL;
#endif
	f.m_destroyFunc  = destroy;
	f.m_isPredictive = false;
}

#if !defined(HK_PLATFORM_SPU)
void hkBoxBoxAgent3::registerAgent3(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::Agent3Funcs f;
	initAgentFunc(f);
	dispatcher->registerAgent3( f, hkcdShapeType::BOX, hkcdShapeType::BOX );
}
#endif

hkpAgentData* hkBoxBoxAgent3::create  ( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData )
{
	new (agentData) hkpBoxBoxManifold();
	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE);
}

/*
void hkBoxBoxAgent3::sepNormal( const hkpAgent3Input& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4& separatingNormalOut )
{
	HK_TIMER_BEGIN("BoxBox", this);
	const hkpBoxShape* boxA = static_cast<const hkpBoxShape*>(bodyA.getShape());
	const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(bodyB.getShape());

	hkVector4 rA; rA.setAll3( boxA->getRadius() );
	hkVector4 rA4; rA4.setAdd4( boxA->getHalfExtents(), rA );
	hkVector4 rB; rB.setAll3( boxB->getRadius() );
	hkVector4 rB4; rB4.setAdd4( boxB->getHalfExtents(), rB );

	hkpBoxBoxCollisionDetection detector( bodyA, bodyB, HK_NULL, HK_NULL, HK_NULL,
										 bodyA.getTransform(), rA4,
										 bodyB.getTransform(), rB4, input.getTolerance() );
	
	hkpCdPoint event( bodyA, bodyB );

	hkBool result = detector.calculateClosestPoint( event.m_contact );

	if (result)
	{ 
		collector.addCdPoint( event );
	}
	HK_TIMER_END();
	}
*/

hkpAgentData* hkBoxBoxAgent3::cleanup ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner )
{
	hkpBoxBoxManifold* manifold = static_cast<hkpBoxBoxManifold*>(agentData);
	for (int i = 0; i < manifold->getNumPoints(); i++)
	{
		if ( manifold->m_contactPoints[i].m_contactPointId != HK_INVALID_CONTACT_POINT )
		{
			mgr->removeContactPoint(manifold->m_contactPoints[i].m_contactPointId, constraintOwner );
		}
	}
	manifold->m_numPoints = 0;
	entry->m_numContactPoints = 0;
	return hkAddByteOffset( agentData, HK_THIS_AGENT_SIZE );
}

void    hkBoxBoxAgent3::removePoint ( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToRemove )
{
	hkpBoxBoxManifold* manifold = static_cast<hkpBoxBoxManifold*>(agentData);
	for ( int i = 0; i < manifold->getNumPoints(); i++)
	{
		if ( manifold->m_contactPoints[i].m_contactPointId  == idToRemove)
		{
			manifold->removePoint( i  );
			entry->m_numContactPoints--;
			break;
		}
	}
}

void hkBoxBoxAgent3::commitPotential( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToCommit )
{
	hkpBoxBoxManifold* manifold = static_cast<hkpBoxBoxManifold*>(agentData);
	for ( int i = 0; i < manifold->getNumPoints(); i++)
	{
		if ( manifold->m_contactPoints[i].m_contactPointId  == HK_INVALID_CONTACT_POINT)
		{
			manifold->m_contactPoints[i].m_contactPointId = idToCommit;
			break;
		}
	}
}

void	hkBoxBoxAgent3::createZombie( hkpAgentEntry* entry, hkpAgentData* agentData, hkContactPointId idToConvert )
{
	return;
}


void  hkBoxBoxAgent3::destroy ( hkpAgentEntry* entry, hkpAgentData* agentData, hkpContactMgr* mgr, hkCollisionConstraintOwner& constraintOwner, hkpCollisionDispatcher* dispatcher  )
{
	cleanup(entry, agentData, mgr, constraintOwner );
}


hkpAgentData* hkBoxBoxAgent3::process( const hkpAgent3ProcessInput& input, hkpAgentEntry* entry, hkpAgentData* agentData, hkVector4* separatingNormal, hkpProcessCollisionOutput& output)
{
	HK_TIMER_BEGIN("BoxBox3", this );

	hkpBoxBoxManifold* manifold = static_cast<hkpBoxBoxManifold*>(agentData);

	const hkpBoxShape* boxA = static_cast<const hkpBoxShape*>(input.m_bodyA->getShape());
	const hkpBoxShape* boxB = static_cast<const hkpBoxShape*>(input.m_bodyB->getShape());

	const hkVector4& extA = boxA->getHalfExtents();
	const hkVector4& extB = boxB->getHalfExtents();

	hkSimdReal rA; rA.load<1>(&boxA->getRadius());
	hkSimdReal rB; rB.load<1>(&boxB->getRadius());
	hkVector4 rA4; rA4.setAdd(extA, rA);
	hkVector4 rB4; rB4.setAdd(extB, rB);

	hkSimdReal tolerance = hkSimdReal::fromFloat(input.m_input->getTolerance());
	hkpBoxBoxCollisionDetection detector( *input.m_bodyA, *input.m_bodyB, input.m_input, input.m_contactMgr, &output,
										  input.m_aTb, input.m_bodyA->getTransform(), rA4, input.m_bodyB->getTransform(), rB4, tolerance );

	detector.calcManifold( *manifold );

	entry->m_numContactPoints = manifold->m_numPoints;
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
