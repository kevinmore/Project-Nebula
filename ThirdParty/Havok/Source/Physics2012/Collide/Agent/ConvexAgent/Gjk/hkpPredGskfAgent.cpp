/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>

#include <Physics2012/Collide/Agent/hkpCollisionAgentConfig.h>
#include <Physics2012/Collide/Agent/hkpCollisionInput.h>
#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpPredGskfAgent.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>
#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifoldUtil.h>
#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>
#include <Physics2012/Collide/Agent3/PredGskAgent3/hkpPredGskAgent3.h>
#include <Physics2012/Internal/Collide/Gjk/Continuous/hkpContinuousGsk.h>

void HK_CALL hkpPredGskfAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createPredGskfAgent;
		af.m_getPenetrationsFunc  = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive        = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX, hkcdShapeType::CONVEX );
	}

}


hkpCollisionAgent* HK_CALL hkpPredGskfAgent::createPredGskfAgent( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr )
{
	hkpGskBaseAgent* agent;
	if ( mgr )
	{
		agent = new hkpPredGskfAgent(bodyA, bodyB, input, mgr);
	}
	else
	{
		agent = new hkpGskBaseAgent( bodyA, bodyB, mgr );
	}
	return agent;
}


void hkpPredGskfAgent::processCollision( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpProcessCollisionInput& input, hkpProcessCollisionOutput& output )
{
	HK_TIME_CODE_BLOCK("Gsk", HK_NULL);
	HK_INTERNAL_TIMER_BEGIN_LIST("PredGskf","init");

	char names[2][128];
#	if defined HK_DEBUG_TOI
	{
		hkpWorldObject* sA = static_cast<hkpWorldObject*>( bodyA.getRootCollidable()->getOwner() );
		hkpWorldObject* sB = static_cast<hkpWorldObject*>( bodyB.getRootCollidable()->getOwner() );
		char* nameA; 
		char* nameB;
		if ( sA ){ nameA = sA->getName(); }else{ hkString::sprintf(names[0], "bodyA" ); nameA = names[0]; } 
		if ( sB ){ nameB = sB->getName(); }else{ hkString::sprintf(names[1], "bodyB" ); nameB = names[1]; } 
		hkToiPrintf("Tst", "#    Tst    %-6s %-6s\n", nameA, nameB );
	}
#endif

	int explicitlyAllowNewPoint = 0;

	{

		//
		//	validate separating plane
		//
		if ( ! (m_timeOfSeparatingNormal == input.m_stepInfo.m_startTime) )
		{
			if ( !input.m_collisionQualityInfo->m_useContinuousPhysics.val() )
			{
				m_timeOfSeparatingNormal = input.m_stepInfo.m_endTime;
				goto PROCESS_AT_T1;
			}

			HK_INTERNAL_TIMER_BEGIN("recalcT0", this);
			hkTransform tA;
			hkTransform tB;
			hkpCdBody bA( &bodyA, &tA ); bA.setShape( bodyA.getShape(), bodyA.getShapeKey());
			hkpCdBody bB( &bodyB, &tB ); bB.setShape( bodyB.getShape(), bodyB.getShapeKey());
			hkSweptTransformUtil::lerp2( bodyA.getMotionState()->getSweptTransform(), input.m_stepInfo.m_startTime, tA );
			hkSweptTransformUtil::lerp2( bodyB.getMotionState()->getSweptTransform(), input.m_stepInfo.m_startTime, tB );
		
			const hkpConvexShape* shapeA = static_cast<const hkpConvexShape*>(bodyA.getShape());
			const hkpConvexShape* shapeB = static_cast<const hkpConvexShape*>(bodyB.getShape());

			hkpGsk gsk;
			gsk.init( shapeA, shapeB, m_cache );
			calcSeparatingNormal( bA, bB, input.m_collisionQualityInfo->m_keepContact, gsk, m_separatingNormal );
			gsk.checkForChangesAndUpdateCache( m_cache );
			HK_INTERNAL_TIMER_END();
		}

			// optimistically set the separatingNormal time to the end of the step
		m_timeOfSeparatingNormal = input.m_stepInfo.m_endTime;
		

		const hkMotionState* msA = bodyA.getMotionState();
		const hkMotionState* msB = bodyB.getMotionState();

		//
		//	Calc the relative movement for this timestep
		//
		hkVector4 timInfo; 	hkSweptTransformUtil::calcTimInfo( *msA, *msB, input.m_stepInfo.m_deltaTime, timInfo);

		hkSimdReal realProjectedLinearDelta  = timInfo.dot<3>( m_separatingNormal );
		hkSimdReal distAtT1 = m_separatingNormal.getW() - realProjectedLinearDelta - timInfo.getW();

		//
		//	Check for traditional tims
		//
		hkSimdReal allowedPenetration; allowedPenetration.load<1>(&m_allowedPenetration);
		if ( distAtT1.isGreater(hkSimdReal::fromFloat(input.m_collisionQualityInfo->m_keepContact)) && distAtT1.isGreater(hkSimdReal_Half * allowedPenetration) )
		{
			hkToiPrintf("Tim", "#    Tim    %-6s %-6s        dist:%2.4f  \n", names[0], names[1], distAtT1.getReal() );
			HK_INTERNAL_TIMER_SPLIT_LIST("tim");

			m_separatingNormal.setW(distAtT1);
			if ( m_manifold.m_numContactPoints )
			{
				hkGskManifold_cleanup( m_manifold, m_contactMgr, *output.m_constraintOwner.val() );
			}
			goto END_OF_FUNCTION;
		}

			//
			//  Check for normal operation
			//
		if ( input.m_collisionQualityInfo->m_useContinuousPhysics.val() )
		{
			//
			//	Advance time using safe time steps
			//
			HK_INTERNAL_TIMER_SPLIT_LIST("toi");
	
			hkpAgent3ProcessInput in3;
			in3.m_bodyA = & bodyA;
			in3.m_bodyB = & bodyB;
			in3.m_input = &input;
			in3.m_contactMgr = m_contactMgr;
			distAtT1.store<1>(&(in3.m_distAtT1.ref()));
			in3.m_linearTimInfo = timInfo;

			hkpCollisionQualityInfo& qi = *input.m_collisionQualityInfo;
			const hkSimdReal distance = m_separatingNormal.getW();
			hkSimdReal minSep; minSep.load<1>(&qi.m_minSeparation);
			hkSimdReal minExtraSep; minExtraSep.load<1>(&qi.m_minExtraSeparation);
			hkSimdReal minSeparation; minSeparation.setMin( minSep * allowedPenetration, distance + minExtraSep * allowedPenetration );
			if (distAtT1.isGreaterEqual(minSeparation))
			{
				goto QUICK_VERIFY_MANIFOLD;
			}
			hkSimdReal toiSep; toiSep.load<1>(&qi.m_toiSeparation);
			hkSimdReal toiExtraSep; toiExtraSep.load<1>(&qi.m_toiExtraSeparation);
			hkSimdReal toiSeparation; toiSeparation.setMin( toiSep * allowedPenetration, distance + toiExtraSep * allowedPenetration );

			hk4dGskCollideCalcToi( in3, allowedPenetration, minSeparation, toiSeparation, m_cache, m_separatingNormal, output );
		}
		else
		{
QUICK_VERIFY_MANIFOLD:
			// tim early out for manifolds
			if ( distAtT1.isGreater(hkSimdReal::fromFloat(input.m_collisionQualityInfo->m_manifoldTimDistance)) )
			{
				hkToiPrintf("Pts", "#    Pts    %-6s %-6s        dist:%2.4f  \n", names[0], names[1], distAtT1.getReal() );
				HK_INTERNAL_TIMER_SPLIT_LIST("getPoints");

				m_separatingNormal.setW(distAtT1);

				hkpGskManifoldWork work;
				hkGskManifold_init( m_manifold, m_separatingNormal, bodyA, bodyB, input.getTolerance(), work );
				explicitlyAllowNewPoint |= hkGskManifold_verifyAndGetPoints( m_manifold, work, 0, output, m_contactMgr ); 

				if (0 == explicitlyAllowNewPoint || !(m_cache.m_gskFlags & hkpGskCache::GSK_FLAGS_ALLOW_QUICKER_CONTACT_POINT_RECREATION))
				{
					goto END_OF_FUNCTION;
				}
				else
				{
					// abort all confirmed points
					output.uncommitContactPoints(m_manifold.m_numContactPoints);
				}
			}
		}

	}

PROCESS_AT_T1:
	HK_INTERNAL_TIMER_SPLIT_LIST("process");
	{
		//hkToiPrintf("Gsk", "#    Gsk    %-6s %-6s        dist:%2.4f  \n", names[0], names[1], distAtT1 );
		hkpGskfAgent::processCollisionNoTim( bodyA, bodyB, input, explicitlyAllowNewPoint, output );
	}
END_OF_FUNCTION:;

	HK_INTERNAL_TIMER_END_LIST();
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
