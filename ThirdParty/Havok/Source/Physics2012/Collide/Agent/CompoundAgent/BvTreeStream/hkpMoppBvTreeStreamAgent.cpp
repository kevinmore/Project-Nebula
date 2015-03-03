/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>


#include <Physics2012/Collide/Agent/CompoundAgent/BvTree/hkpMoppAgent.h>
#include <Physics2012/Collide/Agent/CompoundAgent/BvTreeStream/hkpMoppBvTreeStreamAgent.h>

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppAabbCastVirtualMachine.h>

#include <Physics2012/Collide/Dispatch/hkpAgentDispatchUtil.h>

#ifdef HK_MOPP_DEBUGGER_ENABLED
#	include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#endif

#include <Common/Base/Algorithm/Sort/hkSort.h>


void HK_CALL hkpMoppBvTreeStreamAgent::registerAgent(hkpCollisionDispatcher* dispatcher)
{
	// register symmetric version
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc           = hkpBvTreeStreamAgent::createBvTreeShapeAgent;
		af.m_getPenetrationsFunc  = hkpSymmetricAgent<hkpMoppAgent>::staticGetPenetrations;
		af.m_getClosestPointFunc  = hkpSymmetricAgent<hkpMoppAgent>::staticGetClosestPoints;
		af.m_linearCastFunc       = hkpSymmetricAgent<hkpMoppAgent>::staticLinearCast;
		af.m_isFlipped            = true;
		af.m_isPredictive		  = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MOPP, hkcdShapeType::CONVEX );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          =  hkpBvTreeStreamAgent::createShapeBvAgent;
		af.m_getPenetrationsFunc =  hkpMoppAgent::staticGetPenetrations;
		af.m_getClosestPointFunc =  hkpMoppAgent::staticGetClosestPoints;
		af.m_linearCastFunc      =  hkpMoppAgent::staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::CONVEX, hkcdShapeType::MOPP );
	}
	{
		hkpCollisionDispatcher::AgentFuncs af;
		af.m_createFunc          = hkpMoppAgent::createBvBvAgent;
		af.m_getPenetrationsFunc = hkpMoppAgent::staticGetPenetrations;
		af.m_getClosestPointFunc = hkpMoppAgent::staticGetClosestPoints;
		af.m_linearCastFunc      = hkpMoppAgent::staticLinearCast;
		af.m_isFlipped           = false;
		af.m_isPredictive		 = true;
		dispatcher->registerCollisionAgent(af, hkcdShapeType::MOPP, hkcdShapeType::MOPP );
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
