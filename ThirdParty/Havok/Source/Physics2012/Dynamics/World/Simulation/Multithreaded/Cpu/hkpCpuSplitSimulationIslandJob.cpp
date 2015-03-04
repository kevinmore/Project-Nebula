/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSplitSimulationIslandJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

#if defined (HK_ENABLE_INTERNAL_DATA_RANDOMIZATION)
static hkPseudoRandomGenerator prng(1);
#endif

void HK_CALL hkCpuSplitSimulationIslandJobImpl( hkpSimulationIsland*				island,
												hkArray<hkpSimulationIsland*>&	newSplitIslands )
{
	{
		island->m_splitCheckRequested = false;

		 // we create a new array here, as we want to bypass the destructor,
		// basically we want to keep the original array,
		// the old entities array will be freed in the hkBroadphaseJob::popJobTask
		int buffer[ sizeof(hkArray<hkpEntity*>)/4];
		hkArray<hkpEntity*>& oldEntities = *new(buffer) hkArray<hkpEntity*>;

		island->markForWrite();
		HK_ASSERT2(0xad342938, newSplitIslands.isEmpty(), "newSplitIslands is expected to be empty.");
		hkpWorldOperationUtil::splitSimulationIsland( island, island->getWorld(), newSplitIslands, &oldEntities );

#if defined (HK_ENABLE_INTERNAL_DATA_RANDOMIZATION)
		// Randomize order of new islands
		for (int ii = 0; ii < newSplitIslands.getSize(); ii++)
		{
			int newIndex = int(prng.getRandRange(0.0f, hkReal(newSplitIslands.getSize())-0.01f));
			hkAlgorithm::swap(newSplitIslands[ii], newSplitIslands[newIndex]);
			newSplitIslands[ii]->m_storageIndex = hkObjectIndex(ii);
			newSplitIslands[newIndex]->m_storageIndex = hkObjectIndex(newIndex);
		}
#endif

#ifdef HK_DEBUG_MULTI_THREADING
		{ for (int i =0; i < newSplitIslands.getSize(); i++){ newSplitIslands[i]->unmarkForWrite(); } }
#endif
		island->unmarkForWrite();
	}
}


hkJobQueue::JobStatus HK_CALL hkCpuSplitSimulationIslandJob(	hkpMtThreadStructure&		tl,
																hkJobQueue&					jobQueue,
																hkJobQueue::JobQueueEntry&	jobInOut )
{
	HK_TIMER_BEGIN("SplitIsle", HK_NULL)

	const hkpSplitSimulationIslandJob& job = reinterpret_cast<hkpSplitSimulationIslandJob&>(jobInOut);

	hkCpuSplitSimulationIslandJobImpl( job.m_island, job.m_taskHeader->m_newSplitIslands );

	HK_TIMER_END();
	return jobQueue.finishJobAndGetNextJob( (const hkJobQueue::JobQueueEntry*)&job, jobInOut );
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
