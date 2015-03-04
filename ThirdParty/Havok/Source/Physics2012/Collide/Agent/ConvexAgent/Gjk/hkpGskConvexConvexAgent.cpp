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

#include <Physics2012/Collide/Agent/ConvexAgent/Gjk/hkpGskConvexConvexAgent.h>

//HK_COMPILE_TIME_ASSERT( sizeof( hkpGskConvexConvexAgent ) == 12/*base*/ + 20/*tim*/ + 16/*cache*/ + 4*48 );


hkpGskConvexConvexAgent::hkpGskConvexConvexAgent(	const hkpCdBody& bodyA,	const hkpCdBody& bodyB, hkpContactMgr* mgr ): hkpGskBaseAgent( bodyA, bodyB, mgr )
{
	m_numContactPoints = 0;
	m_numContactPoints = 0;
}

hkpCollisionAgent* HK_CALL hkpGskConvexConvexAgent::createGskConvexConvexAgent(const 	hkpCdBody& bodyA, const hkpCdBody& bodyB,
																			const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	hkpGskBaseAgent* agent;
	if ( mgr )
	{
		agent = new hkpGskConvexConvexAgent(bodyA, bodyB, mgr);
	}
	else
	{
		agent = new hkpGskBaseAgent( bodyA, bodyB, mgr );
	}
	return agent;
}

void HK_CALL hkpGskConvexConvexAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	hkpCollisionDispatcher::AgentFuncs af;
	af.m_createFunc          = createGskConvexConvexAgent;
	af.m_getPenetrationsFunc = staticGetPenetrations;
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc      = staticLinearCast;
	af.m_isFlipped           = false;
	af.m_isPredictive		 = false;

	dispatcher->registerCollisionAgent( af, hkcdShapeType::CONVEX, hkcdShapeType::CONVEX);
}




void hkpGskConvexConvexAgent::cleanup( hkCollisionConstraintOwner& info )
{
	hkpClosestPointManifold::cleanup( m_contactPoints, m_numContactPoints, m_contactMgr, info );
	delete this;
}

#if defined HK_COMPILER_MSVC
	// C4701: local variable 'output' may be used without having been initialized
#	pragma warning(disable: 4701)
#endif



void hkpGskConvexConvexAgent::processCollision(const hkpCdBody& bodyA,		const hkpCdBody& bodyB, 
								const hkpProcessCollisionInput& input,		hkpProcessCollisionOutput& result)
{
	HK_ASSERT2(0x57213df1,  m_contactMgr, HK_MISSING_CONTACT_MANAGER_ERROR_TEXT );

	HK_TIMER_BEGIN_LIST( "Gsk", "Tim" );


	if ( ! (m_timeOfSeparatingNormal == input.m_stepInfo.m_startTime) )
	{
		hkVector4 timInfo;
		hkSweptTransformUtil::calcTimInfo( *bodyA.getMotionState(), *bodyB.getMotionState(), input.m_stepInfo.m_deltaTime, timInfo);
		if ( m_separatingNormal(3) > input.getTolerance() )
		{
			m_separatingNormal(3) -= timInfo.dot4xyz1( m_separatingNormal ).getReal();
			if ( m_separatingNormal(3) > input.getTolerance() )
			{
				goto END;
			}
		}
	}

	{
		hkpExtendedGskOut output;
		HK_TIMER_SPLIT_LIST( "Gsk" );
		bool hasPoint = getClosestPoint(bodyA,bodyB, input, output);
		m_separatingNormal = output.m_normalInWorld;
		m_separatingNormal(3) = output.m_distance;
		m_timeOfSeparatingNormal = input.m_stepInfo.m_endTime;

		if( hasPoint )
		{
			HK_INTERNAL_TIMER_SPLIT_LIST("addPoint");
			hkpCollisionQualityInfo& sq = *input.m_collisionQualityInfo;
			int dim = m_cache.m_dimA + m_cache.m_dimB;
			hkReal createContactRangeMax = (dim==4)? sq.m_create4dContact: sq.m_createContact;

			hkpClosestPointManifold::addPoint(bodyA, bodyB, input, result, output, createContactRangeMax, m_contactMgr, *result.m_constraintOwner.val(), m_contactPoints, m_numContactPoints);

			const hkReal epsilon = .001f;

			HK_INTERNAL_TIMER_SPLIT_LIST("getPoints");
			hkpClosestPointManifold::getPoints( bodyA, bodyB, input, output.m_distance - epsilon, m_contactPoints, m_numContactPoints, result, m_contactMgr, *result.m_constraintOwner.val()); 
		}
		else
		{
			hkpClosestPointManifold::cleanup( m_contactPoints, m_numContactPoints, m_contactMgr, *result.m_constraintOwner.val() );
		}
	}
END:

	HK_TIMER_END_LIST();
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
