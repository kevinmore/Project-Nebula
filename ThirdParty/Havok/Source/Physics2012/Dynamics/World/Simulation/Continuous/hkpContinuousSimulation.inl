/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#if !defined(HK_PLATFORM_SPU)
void hkpContinuousSimulation::waitForSolverExport(hkChar* exportFinished)
{
	// wait until solver export (either on spu or ppu) has finished
	//		extended note: this has to be done even if no add/remove pairs is about to be performed as the 'atomic' spu dma function setting this flag is verifying that
	//		the value hasn't changed and the taskHeader (where this flag is located in) is deleted once we return from here; this could lead to the flag's memory already
	//		being purged before this dma check has been performed
	if ( exportFinished && (*exportFinished != hkChar(1) || *exportFinished != hkChar(2)) )
	{
		HK_TIME_CODE_BLOCK("WaitForExport", HK_NULL);
		volatile hkChar* flag = exportFinished;
		while ( (*flag != hkChar(1)) && (*flag != hkChar(2)) ) { };
	}
}
#endif


#if !defined(HK_PLATFORM_SPU)
void hkpContinuousSimulation::removeAndAddPairs(hkpWorld* world, hkpEntity** entities, hkArray<hkpBroadPhaseHandlePair>& delPairs, hkArray<hkpBroadPhaseHandlePair>& newPairs)
{
#if !defined(HK_ENABLE_DETERMINISM_CHECKS)
	if ( newPairs.getSize() + delPairs.getSize() > 0)
#endif
	{
		hkCheckDeterminismUtil::checkMt( 0xf00001e1, hkUint32(0xadadadad) );

		HK_TIMER_SPLIT_LIST("RemoveAgt");
		world->lockIslandForConstraintUpdate( entities[0]->getSimulationIsland() );
		CHECK_TRACK(entities[0]->getSimulationIsland()->m_narrowphaseAgentTrack);
		CHECK_TRACK(entities[0]->getSimulationIsland()->m_midphaseAgentTrack);
		world->m_broadPhaseDispatcher->removePairs( static_cast<hkpTypedBroadPhaseHandlePair*>(delPairs.begin()), delPairs.getSize() );
		hkCheckDeterminismUtil::checkMt( 0xf00001e2, hkUint32(0xdbdbdbdb) );

		// check the memory limit

		if ( !hkHasMemoryAvailable(28, newPairs.getSize() * 1024) )
		{
			world->unlockIslandForConstraintUpdate( entities[0]->getSimulationIsland() );
			return;
		}

		HK_TIMER_SPLIT_LIST("AddAgt");
		world->m_broadPhaseDispatcher->addPairs( static_cast<hkpTypedBroadPhaseHandlePair*>(newPairs.begin()), newPairs.getSize(), world->getCollisionFilter() );
		CHECK_TRACK(entities[0]->getSimulationIsland()->m_narrowphaseAgentTrack);
		CHECK_TRACK(entities[0]->getSimulationIsland()->m_midphaseAgentTrack);
		world->unlockIslandForConstraintUpdate( entities[0]->getSimulationIsland() );
		hkCheckDeterminismUtil::checkMt( 0xf00001e3, hkUint32(0xacacacac) );
	}
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
