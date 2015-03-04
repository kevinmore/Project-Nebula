/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpBvCompressedMeshAgent.h>
#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShape.h>
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

hkpBvCompressedMeshAgent::hkpBvCompressedMeshAgent(hkpContactMgr* mgr) : hkpBvTreeAgent( mgr ) {}


/////////////////////////////////////////////////
// Agent registration in hkpCollisionDispatcher

void HK_CALL hkpBvCompressedMeshAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{	
	// Register functions for collisions between static mesh and any shape
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createStaticMeshVsShapeAgent;
		af.m_getPenetrationsFunc = hkpSymmetricAgent<hkpBvCompressedMeshAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpSymmetricAgent<hkpBvCompressedMeshAgent>::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpSymmetricAgent<hkpBvCompressedMeshAgent>::staticLinearCast;
		af.m_isFlipped           = true;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_COMPRESSED_MESH, hkcdShapeType::ALL_SHAPE_TYPES);
	}

	// Register functions for collisions between any shape and static mesh
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createShapeVsStaticMeshAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::ALL_SHAPE_TYPES, hkcdShapeType::BV_COMPRESSED_MESH);
	}

	// Register functions for collisions between two static meshes
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = createStaticMeshVsStaticMeshAgent;
		af.m_getPenetrationsFunc = staticGetPenetrations;
		af.m_getClosestPointFunc = staticGetClosestPoints;
		af.m_linearCastFunc      = staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::BV_COMPRESSED_MESH, hkcdShapeType::BV_COMPRESSED_MESH);
	}
}


//////////////////////////////
// Agent creation functions

hkpCollisionAgent* HK_CALL hkpBvCompressedMeshAgent::createShapeVsStaticMeshAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpBvCompressedMeshAgent(mgr);
}

hkpCollisionAgent* HK_CALL hkpBvCompressedMeshAgent::createStaticMeshVsShapeAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpContactMgr* mgr)
{
	return new hkpBvCompressedMeshAgent(mgr);
}

hkpCollisionAgent* HK_CALL hkpBvCompressedMeshAgent::createStaticMeshVsStaticMeshAgent(const hkpCdBody& bodyA, 
																				 const hkpCdBody& bodyB,
																				 const hkpCollisionInput& input,
																				 hkpContactMgr* mgr)
{
	// Collide the smallest mesh in number of primitives against the biggest one. hkpBvCompressedMeshShape has an
	// optimized implementation of getNumShildShapes that allows to do this check cheaply.
	const hkpBvCompressedMeshShape* meshA = static_cast<const hkpBvCompressedMeshShape*>(bodyA.getShape());
	const hkpBvCompressedMeshShape* meshB = static_cast<const hkpBvCompressedMeshShape*>(bodyB.getShape());	
	if ( meshA->getNumChildShapes() < meshB->getNumChildShapes() )
	{
		hkpBvCompressedMeshAgent* agent = new hkpBvCompressedMeshAgent(mgr);
		return agent;
	}
	else
	{
		hkpBvCompressedMeshAgent* agent = new hkpSymmetricAgent<hkpBvCompressedMeshAgent>(mgr);
		return agent;
	}
}

#else


/////////////////////////////////////////////////////////
// Agent registration in hkpSpuCollisionQueryDispatcher

void HK_CALL hkpBvCompressedMeshAgent::initAgentFunc(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = staticGetClosestPoints;
	af.m_linearCastFunc = staticLinearCast;
}

void HK_CALL hkpBvCompressedMeshAgent::initAgentFuncInverse(hkpSpuCollisionQueryDispatcher::AgentFuncs& af)
{
	af.m_getClosestPointFunc = hkpSymmetricAgent<hkpBvCompressedMeshAgent>::staticGetClosestPoints;
	af.m_linearCastFunc = hkpSymmetricAgent<hkpBvCompressedMeshAgent>::staticLinearCast;
}

#endif // !defined(HK_PLATFORM_SPU)


/////////////////////////////////
// hkpCollisionAgent interface	

void hkpBvCompressedMeshAgent::linearCast(const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, 
									hkpCdPointCollector& collector, hkpCdPointCollector* startCollector)
{
	staticLinearCast(bodyA, bodyB, input, collector, startCollector);
}

void hkpBvCompressedMeshAgent::staticLinearCast(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
												const hkpLinearCastCollisionInput& input, hkpCdPointCollector& castCollector, 
												hkpCdPointCollector* startCollector)
{
	// Calculate initial bodyA aabb and final position in bodyB space
	hkAabb aabbFrom;
	hkVector4 to;
	{		
		// Calculate transform from bodyA to bodyB
		hkTransform aToB;
		const hkTransform& bToW = bodyB.getTransform();
		const hkTransform& aToW = bodyA.getTransform();		
		aToB.setMulInverseMul(bToW, aToW);	

		// Get bodyA aabb in bodyB space
		bodyA.getShape()->getAabb(aToB, input.m_tolerance, aabbFrom);
	
		// Calculate final bodyA position in bodyB space	
		hkVector4 bodyAPos;
		aabbFrom.getCenter(bodyAPos);
		to._setRotatedInverseDir(bToW.getRotation(), input.m_path);
		to.add(bodyAPos);
	}
		
	// Obtain bodyB mesh shape	
	const hkpBvCompressedMeshShape* mesh = static_cast<const hkpBvCompressedMeshShape*>(bodyB.getShape());

	// Perform linear cast against the tree
	hkpBvTreeAgent::LinearCastAabbCastCollector aabbCastCollector(bodyA, bodyB, input, castCollector, startCollector);
	mesh->castAabb(aabbFrom, to, aabbCastCollector);
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
