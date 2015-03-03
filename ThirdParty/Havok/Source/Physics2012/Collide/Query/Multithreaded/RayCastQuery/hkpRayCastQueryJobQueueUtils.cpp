/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobQueueUtils.h>

#ifdef HK_PLATFORM_SPU
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h> 
#endif

#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuShapeRaycastJob.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/Cpu/hkpCpuWorldRaycastJob.h>


HK_COMPILE_TIME_ASSERT( sizeof( hkpShapeRayCastJob )				<= sizeof( hkJobQueue::JobQueueEntry ) );
HK_COMPILE_TIME_ASSERT( sizeof( hkpWorldRayCastJob )				<= sizeof( hkJobQueue::JobQueueEntry ) );


//
// POP COLLISION JOB
//

hkJobQueue::JobPopFuncResult HK_CALL hkpRayCastQueryJobQueueUtils::popRayCastQueryJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut )
{
	hkString::memCpy16NonEmpty(&jobOut, &jobIn, sizeof(hkJobQueue::JobQueueEntry)>>4);

	hkpRayCastQueryJob& job = reinterpret_cast<hkpRayCastQueryJob&>(jobIn);

	//
	// Handle the different collision query jobs.
	//
	switch( job.m_jobSubType )
	{
		case hkpRayCastQueryJob::RAYCAST_QUERY_SHAPE_RAYCAST:
			{
				hkpShapeRayCastJob& onQueue = static_cast     <hkpShapeRayCastJob&>(job);
				hkpShapeRayCastJob& out     = reinterpret_cast<hkpShapeRayCastJob&>(jobOut);
				return onQueue.popJobTask(out);
			}

		case hkpRayCastQueryJob::RAYCAST_QUERY_WORLD_RAYCAST:
			{
				hkpWorldRayCastJob& onQueue = static_cast     <hkpWorldRayCastJob&>(job);
				hkpWorldRayCastJob& out     = reinterpret_cast<hkpWorldRayCastJob&>(jobOut);
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
	static HK_LOCAL_INLINE const hkpRayCastQueryJobHeader* getJobHeaderFromPpu(const hkpRayCastQueryJobHeader* jobHeaderInMainMemory, hkpRayCastQueryJobHeader* buffer)
	{
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( buffer, jobHeaderInMainMemory, sizeof(hkpRayCastQueryJobHeader), hkSpuDmaManager::READ_COPY );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS( jobHeaderInMainMemory, buffer, sizeof(hkpRayCastQueryJobHeader) );
		return buffer;
	}

	static HK_LOCAL_INLINE void putOpenJobsVariableToPpu( const hkpRayCastQueryJobHeader* localJobHeader, const hkpRayCastQueryJobHeader* jobHeaderInMainMemory )
	{
		HK_CPU_PTR(hkpRayCastQueryJobHeader*) dest = const_cast<HK_CPU_PTR(hkpRayCastQueryJobHeader*)>(jobHeaderInMainMemory);
		hkSpuDmaManager::putToMainMemorySmallAndWaitForCompletion( &dest->m_openJobs, &localJobHeader->m_openJobs, sizeof(int), hkSpuDmaManager::WRITE_NEW );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS                          ( &dest->m_openJobs, &localJobHeader->m_openJobs, sizeof(int) );
	}

}
#endif

hkJobQueue::JobCreationStatus HK_CALL hkpRayCastQueryJobQueueUtils::finishRayCastQueryJob( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreated )
{
	const hkpRayCastQueryJob& job = reinterpret_cast<const hkpRayCastQueryJob&>(jobIn);

	//
	// Bring in the jobHeader. This will be done by either DMA (if we are on SPU) or by simply redirecting (if we are on PPU/CPU).
	//
#if defined (HK_PLATFORM_SPU)
	HK_ALIGN16( char jobHeaderBufferOnSpu[sizeof(hkpRayCastQueryJobHeader)] );
	const hkpRayCastQueryJobHeader* localJobHeader = HK_NULL;
	if ( job.m_sharedJobHeaderOnPpu )
	{
		localJobHeader = getJobHeaderFromPpu(job.m_sharedJobHeaderOnPpu, reinterpret_cast<hkpRayCastQueryJobHeader*>(jobHeaderBufferOnSpu));
	}
#else
	const hkpRayCastQueryJobHeader* localJobHeader = job.m_sharedJobHeaderOnPpu;
#endif

	//
	// Handle the different collision query jobs.
	//
	switch( job.m_jobSubType )
	{
		case hkpRayCastQueryJob::RAYCAST_QUERY_SHAPE_RAYCAST:
			{
				const hkpShapeRayCastJob& raycastJob = reinterpret_cast<const hkpShapeRayCastJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					raycastJob.atomicIncrementAndReleaseSemaphore();

				}
				break;
			}

		case hkpRayCastQueryJob::RAYCAST_QUERY_WORLD_RAYCAST:
			{
				const hkpWorldRayCastJob& worldRayCastJob = reinterpret_cast<const hkpWorldRayCastJob&>(job);

				localJobHeader->m_openJobs--;
				if ( localJobHeader->m_openJobs == 0 )
				{
					// Release the semaphore to indicate the job is complete
					worldRayCastJob.atomicIncrementAndReleaseSemaphore();

				}
				break;
			}
		default:
			{
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
