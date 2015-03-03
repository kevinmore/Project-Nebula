/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobQueueUtils.h>

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairGetClosestPointsJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldGetClosestPointsJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuMoppAabbJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuPairLinearCastJob.h>
#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/Cpu/hkpCpuWorldLinearCastJob.h>

#ifdef HK_PLATFORM_HAS_SPU
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

//
// POP COLLISION JOB
//

hkJobQueue::JobPopFuncResult HK_CALL hkpCollisionQueryJobQueueUtils::popCollisionJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut )
{
	hkString::memCpy16NonEmpty(&jobOut, &jobIn, sizeof(hkJobQueue::JobQueueEntry)>>4);

	hkpCollisionQueryJob& job = reinterpret_cast<hkpCollisionQueryJob&>(jobIn);

	//
	// Handle the different collision query jobs.
	//
	switch( job.m_jobSubType )
	{
		case hkpCollisionQueryJob::COLLISION_QUERY_PAIR_LINEAR_CAST:
			{
				hkpPairLinearCastJob& onQueue = static_cast     <hkpPairLinearCastJob&>(job);
				hkpPairLinearCastJob& out     = reinterpret_cast<hkpPairLinearCastJob&>(jobOut);
				return onQueue.popJobTask(out);
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_WORLD_LINEAR_CAST:
			{
				hkpWorldLinearCastJob& onQueue = static_cast     <hkpWorldLinearCastJob&>(job);
				hkpWorldLinearCastJob& out     = reinterpret_cast<hkpWorldLinearCastJob&>(jobOut);
				return onQueue.popJobTask(out);
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_MOPP_AABB:
			{
				hkpMoppAabbJob& onQueue = static_cast     <hkpMoppAabbJob&>(job);
				hkpMoppAabbJob& out     = reinterpret_cast<hkpMoppAabbJob&>(jobOut);
				return onQueue.popJobTask(out);
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS:
			{
				hkpPairGetClosestPointsJob& onQueue = static_cast     <hkpPairGetClosestPointsJob&>(job);
				hkpPairGetClosestPointsJob& out     = reinterpret_cast<hkpPairGetClosestPointsJob&>(jobOut);
				return onQueue.popJobTask(out);
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS:
			{
				hkpWorldGetClosestPointsJob& onQueue = static_cast     <hkpWorldGetClosestPointsJob&>(job);
				hkpWorldGetClosestPointsJob& out     = reinterpret_cast<hkpWorldGetClosestPointsJob&>(jobOut);
				return onQueue.popJobTask(out);
			}

		default:
			{
				break;
			}
	}

	return hkJobQueue::POP_QUEUE_ENTRY;
}


//
// FINISH COLLISION JOB
//
#ifdef HK_PLATFORM_SPU

namespace
{
	static HK_LOCAL_INLINE const hkpCollisionQueryJobHeader* getJobHeaderFromPpu(const hkpCollisionQueryJobHeader* jobHeaderInMainMemory, hkpCollisionQueryJobHeader* buffer)
	{
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( buffer, jobHeaderInMainMemory, sizeof(hkpCollisionQueryJobHeader), hkSpuDmaManager::READ_COPY );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS( jobHeaderInMainMemory, buffer, sizeof(hkpCollisionQueryJobHeader) );
		return buffer;
	}

	static HK_LOCAL_INLINE void putOpenJobsVariableToPpu( const hkpCollisionQueryJobHeader* localJobHeader, const hkpCollisionQueryJobHeader* jobHeaderInMainMemory )
	{
		HK_CPU_PTR(hkpCollisionQueryJobHeader*) dest = const_cast<HK_CPU_PTR(hkpCollisionQueryJobHeader*)>(jobHeaderInMainMemory);
		hkSpuDmaManager::putToMainMemorySmallAndWaitForCompletion( &dest->m_openJobs, &localJobHeader->m_openJobs, sizeof(int), hkSpuDmaManager::WRITE_NEW );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS                          ( &dest->m_openJobs, &localJobHeader->m_openJobs, sizeof(int) );
	}

}

#endif

hkJobQueue::JobCreationStatus HK_CALL hkpCollisionQueryJobQueueUtils::finishCollisionJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreated )
{
	const hkpCollisionQueryJob& job = reinterpret_cast<const hkpCollisionQueryJob&>(jobIn);

	//
	// Bring in the jobHeader. This will be done by either DMA (if we are on SPU) or by simply redirecting (if we are on PPU/CPU).
	//
#if defined (HK_PLATFORM_SPU)
	HK_ALIGN16( char jobHeaderBufferOnSpu[sizeof(hkpCollisionQueryJobHeader)] );
	const hkpCollisionQueryJobHeader* localJobHeader = HK_NULL;
	if ( job.m_sharedJobHeaderOnPpu )
	{
		localJobHeader = getJobHeaderFromPpu(job.m_sharedJobHeaderOnPpu, reinterpret_cast<hkpCollisionQueryJobHeader*>(jobHeaderBufferOnSpu));
	}
#else
	const hkpCollisionQueryJobHeader* localJobHeader = job.m_sharedJobHeaderOnPpu;
#endif

	//
	// Handle the different collision query jobs.
	//
	switch( job.m_jobSubType )
	{
		case hkpCollisionQueryJob::COLLISION_QUERY_PAIR_LINEAR_CAST:
			{
				const hkpPairLinearCastJob& pairLinearCastJob = reinterpret_cast<const hkpPairLinearCastJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					pairLinearCastJob.atomicIncrementAndReleaseSemaphore();
				}
				break;
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_WORLD_LINEAR_CAST:
			{
				const hkpWorldLinearCastJob& worldLinearCastJob = reinterpret_cast<const hkpWorldLinearCastJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					worldLinearCastJob.atomicIncrementAndReleaseSemaphore();
				}
				break;
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_MOPP_AABB :
			{
				const hkpMoppAabbJob& moppAabbQueryJob =  reinterpret_cast<const hkpMoppAabbJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					moppAabbQueryJob.atomicIncrementAndReleaseSemaphore();
				}
				break;
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS:
			{
				const hkpPairGetClosestPointsJob& pairGetClosestPointsJob = reinterpret_cast<const hkpPairGetClosestPointsJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					pairGetClosestPointsJob.atomicIncrementAndReleaseSemaphore();
				}
				break;
			}

		case hkpCollisionQueryJob::COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS:
			{
				const hkpWorldGetClosestPointsJob& worldGetClosestPointsJob = reinterpret_cast<const hkpWorldGetClosestPointsJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					worldGetClosestPointsJob.atomicIncrementAndReleaseSemaphore();
				}
				break;
			}

		default:
			{
				HK_ASSERT2(0x1f5199b6, false, "Unknown job type in hkpCollisionQueryJobQueueUtils::finishCollisionJob");
				break;
			}
	}

#if defined (HK_PLATFORM_SPU)
	//
	// write back the modified m_openJobs variable to PPU.
	//
	if ( job.m_sharedJobHeaderOnPpu )
	{
		putOpenJobsVariableToPpu( localJobHeader, job.m_sharedJobHeaderOnPpu );
	}
#endif

	return hkJobQueue::NO_JOB_CREATED;
}


HK_COMPILE_TIME_ASSERT( sizeof( hkpMoppAabbJob )				<= sizeof( hkJobQueue::JobQueueEntry ) );
HK_COMPILE_TIME_ASSERT( sizeof( hkpPairGetClosestPointsJob )	<= sizeof( hkJobQueue::JobQueueEntry ) );
HK_COMPILE_TIME_ASSERT( sizeof( hkpWorldGetClosestPointsJob )	<= sizeof( hkJobQueue::JobQueueEntry ) );
HK_COMPILE_TIME_ASSERT( sizeof( hkpPairLinearCastJob )			<= sizeof( hkJobQueue::JobQueueEntry ) );
HK_COMPILE_TIME_ASSERT( sizeof( hkpWorldLinearCastJob )			<= sizeof( hkJobQueue::JobQueueEntry ) );

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
