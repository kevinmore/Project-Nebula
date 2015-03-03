/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Agent/Util/Null/hkpNullAgent.h>
#include <Physics2012/Collide/Dispatch/hkpAgentRegisterUtil.h>


//#include <hkcollide/dispatch/contactmgr/hkpNullContactMgrFactory.h>

//static hkpNullContactMgr hkpNullContactMgr;
static hkpNullAgent hkNullAgentInstance;

hkpNullAgent::hkpNullAgent()
:	hkpCollisionAgent( HK_NULL )
{
}

void HK_CALL hkpNullAgent::staticGetClosestPoints( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, class hkpCdPointCollector& collector  )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( bodyA.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( bodyB.getShape()->getType() ) );
	HK_WARN_ONCE(0x3ad17e8b,  "Have you called hkpAgentRegisterUtil::registerAllAgents?\n" \
								"Do not know how to get closest points between " << typeA << " and " << typeB << " types.");
}

void HK_CALL hkpNullAgent::staticGetPenetrations(const  hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpCollisionInput& input, hkpCdBodyPairCollector& collector )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( bodyA.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( bodyB.getShape()->getType() ) );
	HK_WARN_ONCE(0x3ad17e8c,  "Have you called hkpAgentRegisterUtil::registerAllAgents?\n" \
								"Do not know how to get penetrations for " << typeA << " and " << typeB << " types.");
}

void HK_CALL hkpNullAgent::staticLinearCast( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpLinearCastCollisionInput& input, hkpCdPointCollector& collector, hkpCdPointCollector* startCollector )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( bodyA.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( bodyB.getShape()->getType() ) );
	HK_WARN_ONCE(0x3ad17e8d,  "Have you called hkpAgentRegisterUtil::registerAllAgents?\n" \
								"Do not know how to make linear casting between " << typeA << " and " << typeB << " types.");
}

hkpCollisionAgent* HK_CALL hkpNullAgent::createNullAgent(const hkpCdBody& bodyA, const hkpCdBody& bodyB, 
												const hkpCollisionInput& input, hkpContactMgr* mgr )
{
	HK_ON_DEBUG( const char* typeA = hkGetShapeTypeName( bodyA.getShape()->getType() ) );
	HK_ON_DEBUG( const char* typeB = hkGetShapeTypeName( bodyB.getShape()->getType() ) );
	HK_WARN_ONCE(0x3ad17e8a,  "Have you called hkpAgentRegisterUtil::registerAllAgents?\n" \
								"Do not know how to dispatch types " << typeA << " vs " << typeB);
	return &hkNullAgentInstance;
}

hkpNullAgent* HK_CALL hkpNullAgent::getNullAgent()
{
	return &hkNullAgentInstance;
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
