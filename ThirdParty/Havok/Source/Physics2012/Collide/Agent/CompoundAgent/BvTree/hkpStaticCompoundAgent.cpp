/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpStaticCompoundAgent.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>
#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/Agent/Collidable/hkpCdBody.h>
#if defined(HK_PLATFORM_SPU)
	#include <Physics2012/Collide/Filter/Spu/hkpSpuCollisionFilterUtil.h>
#endif


// Macros to access the right dispatcher regardless of the platform
#if defined(HK_PLATFORM_SPU)
	#define GET_LINEAR_CAST_FUNC									input.m_queryDispatcher->getLinearCastFunc	
	typedef hkpSpuCollisionQueryDispatcher::LinearCastFunc			LinearCastFunc;
#else	
	#define GET_LINEAR_CAST_FUNC									input.m_dispatcher->getLinearCastFunc	
	typedef hkpCollisionDispatcher::LinearCastFunc					LinearCastFunc;
#endif


#if !defined(HK_PLATFORM_SPU)

hkpStaticCompoundAgent::hkpStaticCompoundAgent(hkpContactMgr* mgr) : hkpBvTreeAgent( mgr ) {}


/////////////////////////////////////////////////
// Agent registration in hkpCollisionDispatcher

void HK_CALL hkpStaticCompoundAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{	
	// Register functions for collisions between static compound and any shape
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createStaticCompoundVsShapeAgent;
		af.m_getPenetrationsFunc = hkpSymmetricAgent<hkpStaticCompoundAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpStaticCompoundAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpStaticCompoundAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::STATIC_COMPOUND, hkcdShapeType::ALL_SHAPE_TYPES);
	}

	// Register functions for collisions between any shape and static compound
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createShapeVsStaticCompoundAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::STATIC_COMPOUND);
	}

	// Register functions for collisions between two static compounds
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createStaticCompoundVsStaticCompoundAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::STATIC_COMPOUND, hkcdShapeType::STATIC_COMPOUND);
	}
}


//////////////////////////////
// Agent creation functions

hkpCollisionAgent* HK_CALL hkpStaticCompoundAgent::createShapeVsStaticCompoundAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpStaticCompoundAgent(mgr);
}

hkpCollisionAgent* HK_CALL hkpStaticCompoundAgent::createStaticCompoundVsShapeAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpStaticCompoundAgent(mgr);
}

hkpCollisionAgent* HK_CALL hkpStaticCompoundAgent::createStaticCompoundVsStaticCompoundAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpStaticCompoundAgent(mgr);
}

#else


/////////////////////////////////////////////////////////
// Agent registration in hkpSpuCollisionQueryDispatcher

void HK_CALL hkpStaticCompoundAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc = staticLinearCast;
}

void HK_CALL hkpStaticCompoundAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = hkpSymmetricAgent<hkpStaticCompoundAgent>::staticGetClosestPoints;
	af.m_linearCastFunc = hkpSymmetricAgent<hkpStaticCompoundAgent>::staticLinearCast;
}

#endif // !defined(HK_PLATFORM_SPU)


/////////////////////////////////
// hkpCollisionAgent interface	

void hkpStaticCompoundAgent::linearCast(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, 
										hkpCdPointCollector& collector, hkpCdPointCollector* startCollector)
{
	staticLinearCast(bodyA, bodyB, input, collector, startCollector);
}

void hkpStaticCompoundAgent::staticLinearCast(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
											  const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, 
											  hkpCdPointCollector* startCollector)
{	
	// Calculate transform from body A to body B
	hkTransform AtoB;
	hkVector4 toInB;
	{
		const hkTransform& AtoW = bodyA.getTransform();
		hkTransform WtoB; WtoB.setInverse(bodyB.getTransform());
		AtoB.setMul(WtoB, bodyA.getTransform());
		toInB.setAdd(AtoW.getTranslation(), input.m_path);
		toInB._setTransformedPos(WtoB, toInB);
	}

	// Cast shape A aabb into the compound shape
	const hkpStaticCompoundShape* compound = static_cast<const hkpStaticCompoundShape*>(bodyB.getShape());
	hkpBvTreeAgent::LinearCastAabbCastCollector aabbCastCollector(bodyA, bodyB, input, collector, startCollector);
	compound->castAabb(bodyA.getShape(), AtoB, toInB, aabbCastCollector, input.getTolerance());
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
