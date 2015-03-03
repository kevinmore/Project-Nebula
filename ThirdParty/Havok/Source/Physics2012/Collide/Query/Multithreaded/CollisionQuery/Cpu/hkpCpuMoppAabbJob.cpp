/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuMoppAabbJob.h>

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppObbVirtualMachine.h>


HK_FORCE_INLINE void HK_CALL hkCpuMoppAabbJobProcessCommand(hkpMoppObbVirtualMachine& query, hkpMoppCode* rootCodePtr, const hkpMoppAabbCommand& moppAabbCommand)
{
	//
	// Create a local hkArray using the supplied output buffer as m_data. The array is initialized as 'empty' but with a capacity of MAX_OUTPUT_KEYS_PER_QUERY.
	//
	hkArray<hkpMoppPrimitiveInfo> output( (hkpMoppPrimitiveInfo *)moppAabbCommand.m_results, 0, hkpMoppAabbCommand::MAX_OUTPUT_KEYS_PER_QUERY );

	//
	// Do the query
	//
	query.queryAabb( rootCodePtr, moppAabbCommand.m_aabbInput, &output );

	//
	// add an 'end' marker to results
	//
	{
		hkpMoppPrimitiveInfo info;
		info.ID = HK_INVALID_SHAPE_KEY;
		output.pushBack( info );
	}
}


hkJobQueue::JobStatus HK_CALL hkCpuMoppAabbQueryJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_CHECK_FLUSH_DENORMALS();
	HK_TIMER_BEGIN("CollQueryMoppAabbQuery", HK_NULL);

	const hkpMoppAabbJob& moppAabbJob = reinterpret_cast<hkpMoppAabbJob&>(nextJobOut);

	// Init the query
	hkpMoppObbVirtualMachine query;

	HK_DECLARE_ALIGNED_LOCAL_PTR( hkpMoppCode, rootCodePtr, 16 );
	rootCodePtr->initialize( moppAabbJob.m_moppCodeInfo, moppAabbJob.m_moppCodeData, HK_MOPP_CHUNK_SIZE );

	//
	// perform all queries
	//
	{
		for (int i = 0; i < moppAabbJob.m_numCommands; i++)
		{
			hkCpuMoppAabbJobProcessCommand(query, rootCodePtr, moppAabbJob.m_commandArray[i]);
		}
	}

	HK_TIMER_END();

	return jobQueue.finishJobAndGetNextJob( &nextJobOut, nextJobOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
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
