/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Physics2012/Collide/Shape/Convex/hkpConvexShape.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpCollisionQualityInfo.h>

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskfAgent.h>
#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>

#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Internal/Collide/Gjk/Agent/hkpGskAgentUtil.h>

//HK_COMPILE_TIME_ASSERT( sizeof( hkpGskfAgent ) == 12/*base*/ + 20/*tim*/ + 16/*cache*/ + 64/*manifold*/ );


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hkpGskfAgent::initAgentFunc(hkpCollisionDispatcher::AgentFuncs& af)
{
	af.m_createFunc          = createGskfAgent;
	af.m_getPenetrationsFunc = staticGetPenetrations;
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc      = staticLinearCast;
	af.m_isFlipped           = false;
	af.m_isPredictive		 = false;
}

#else

void HK_CALL hkpGskfAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc      = staticLinearCast;
}

#endif


#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkpGskfAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::AgentFuncs af;
	initAgentFunc(af);
	dispatcher->registerCollisionAgent( af, hkcdShapeType::CONVEX, hkcdShapeType::CONVEX );
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpGskfAgent::hkpGskfAgent(	const hkpCdBody& bodyA,	const hkpCdBody& bodyB, hkpContactMgr* mgr ): hkpGskBaseAgent( bodyA, bodyB, mgr )
{
}
#endif


#if !defined(HK_PLATFORM_SPU)
hkpCollisionAgent* HK_CALL hkpGskfAgent::createGskfAgent(const 	hkpCdBody& bodyA, const hkpCdBody& bodyB, 
																			const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpGskBaseAgent* agent;
	if ( mgr )
	{
		agent = new hkpGskfAgent(bodyA, bodyB, mgr);
	}
	else
	{
		agent = new hkpGskBaseAgent( bodyA, bodyB, mgr );
	}
	return agent;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskfAgent::cleanup( hkCollisionConstraintOwner& constraintOwner )
{
	hkGskManifold_cleanup( m_manifold, m_contactMgr, constraintOwner );
	delete this;
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskfAgent::removePoint( hkContactPointId idToRemove )
{
	for ( int i = 0; i < m_manifold.m_numContactPoints; i++)
	{
		if ( m_manifold.m_contactPoints[i].m_id == idToRemove)
		{
			hkGskManifold_removePoint( m_manifold, i );
			break;
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskfAgent::commitPotential( hkContactPointId idToCommit )
{
	for ( int i = 0; i < m_manifold.m_numContactPoints; i++)
	{
		if ( m_manifold.m_contactPoints[i].m_id == HK_INVALID_CONTACT_POINT)
		{
			m_manifold.m_contactPoints[i].m_id = idToCommit;
			break;
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskfAgent::createZombie( hkContactPointId idTobecomeZombie )
{
	for ( int i = 0; i < m_manifold.m_numContactPoints; i++)
	{
		hkpGskManifold::ContactPoint& cp = m_manifold.m_contactPoints[i];
		if ( cp.m_id == idTobecomeZombie)
		{
			cp.m_dimA = 0;
			cp.m_dimB = 0;
			break;
		}
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)

#if defined HK_COMPILER_MSVC
	// C4701: local variable 'output' may be used without having been initialized
#	pragma warning(disable: 4701)
#endif

void hkpGskfAgent::processCollisionNoTim(const hkpCdBody& bodyA,	const hkpCdBody& bodyB, 
										const hkpProcessCollisionInput& input, int explicitlyAllowNewPoint, hkpProcessCollisionOutput& output)
{
	hkpAgent3ProcessInput in3;
	{
		in3.m_bodyA = &bodyA;
		in3.m_bodyB = &bodyB;
		in3.m_contactMgr = m_contactMgr;
		in3.m_input = &input;

		const hkMotionState* msA = bodyA.getMotionState();
		const hkMotionState* msB = bodyB.getMotionState();
		//hkSweptTransformUtil::calcTimInfo( *msA, *msB, in3.m_linearTimInfo);
		in3.m_aTb.setMulInverseMul(msA->getTransform(), msB->getTransform());
	}
	hkpAgentEntry dummyEntry;
	hkpGskCache dummyAgentData;
	hkGskAgentUtil_processCollisionNoTim( in3, &dummyEntry, &dummyAgentData, m_cache, m_manifold, m_separatingNormal, explicitlyAllowNewPoint, output );
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpGskfAgent::processCollision(const hkpCdBody& bodyA,		const hkpCdBody& bodyB, 
								const hkpProcessCollisionInput& input,		hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x57213df1,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN_LIST( "GskAgent", "Tim" );

	//
	//	Get the relative linear movement (xyz) and the worst case angular movement (w)
	//
	if ( ! (m_timeOfSeparatingNormal == input.m_stepInfo.m_startTime) )
	{
		hkVector4 timInfo;
		hkSweptTransformUtil::calcTimInfo( *bodyA.getMotionState(), *bodyB.getMotionState(), input.m_stepInfo.m_deltaTime, timInfo);

		hkSimdReal t = m_separatingNormal.getW();
		const hkSimdReal tol = hkSimdReal::fromFloat(input.getTolerance());
		if ( t.isGreater(tol) )
		{
			t.sub( timInfo.dot4xyz1( m_separatingNormal ) );
			m_separatingNormal.setW(t);
			if ( t.isGreater(tol) )
			{
				goto END;
			}
		}
	}
	HK_TIMER_SPLIT_LIST( "Gsk" );
	{
		m_timeOfSeparatingNormal = input.m_stepInfo.m_endTime;
		int explicitlyAllowNewPoint = 0;
		processCollisionNoTim( bodyA, bodyB, input, explicitlyAllowNewPoint, result);
	}
END:;
	HK_TIMER_END_LIST();
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
