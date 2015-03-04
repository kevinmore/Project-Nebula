/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobs.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobQueueUtils.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyIntegrateJob.h>

#ifdef HK_PLATFORM_HAS_SPU
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

HK_COMPILE_TIME_ASSERT( sizeof( hkpCharacterProxyJob ) <= sizeof( hkJobQueue::JobQueueEntry ) );

hkJobQueue::JobPopFuncResult HK_CALL hkpCharacterProxyJobQueueUtils::popCharacterProxyJob  ( hkJobQueue& queue,
							hkJobQueue::DynamicData* data,
							hkJobQueue::JobQueueEntry& jobIn,
							hkJobQueue::JobQueueEntry& jobOut  )
{

	hkString::memCpy16NonEmpty(&jobOut, &jobIn, sizeof(hkJobQueue::JobQueueEntry)>>4);

	hkpCharacterProxyJob& job = reinterpret_cast<hkpCharacterProxyJob&>(jobIn);

	//
	// Handle the different character proxy jobs.
	//
	switch( job.m_jobSubType )
	{
		case hkpCharacterProxyJob::CHARACTER_PROXY_JOB_INTEGRATE:
		{
			hkpCharacterProxyIntegrateJob& onQueue = static_cast     <hkpCharacterProxyIntegrateJob&>(job);
			hkpCharacterProxyIntegrateJob& out     = reinterpret_cast<hkpCharacterProxyIntegrateJob&>(jobOut);
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
// FINISH CHARACTER PROXY JOB
//
#ifdef HK_PLATFORM_SPU

namespace
{
	static HK_LOCAL_INLINE const hkpCharacterProxyJobHeader* getJobHeaderFromPpu(const hkpCharacterProxyJobHeader* jobHeaderInMainMemory, hkpCharacterProxyJobHeader* buffer)
	{
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( buffer, jobHeaderInMainMemory, sizeof(hkpCharacterProxyJobHeader), hkSpuDmaManager::READ_COPY );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS( jobHeaderInMainMemory, buffer, sizeof(hkpCharacterProxyJobHeader) );
		return buffer;
	}

	static HK_LOCAL_INLINE void putOpenJobsVariableToPpu( const hkpCharacterProxyJobHeader* localJobHeader, const hkpCharacterProxyJobHeader* jobHeaderInMainMemory )
	{
		HK_CPU_PTR(hkpCharacterProxyJobHeader*) dest = const_cast<HK_CPU_PTR(hkpCharacterProxyJobHeader*)>(jobHeaderInMainMemory);
		hkSpuDmaManager::putToMainMemorySmallAndWaitForCompletion( &dest->m_openJobs, &localJobHeader->m_openJobs, sizeof(int), hkSpuDmaManager::WRITE_NEW );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS                          ( &dest->m_openJobs, &localJobHeader->m_openJobs, sizeof(int) );
	}

}

#endif


hkJobQueue::JobCreationStatus HK_CALL hkpCharacterProxyJobQueueUtils::finishCharacterProxyJob( hkJobQueue& queue,
											hkJobQueue::DynamicData* data,
											const hkJobQueue::JobQueueEntry& jobIn,
											hkJobQueue::JobQueueEntryInput& newJobCreated )
{
	const hkpCharacterProxyJob& job = reinterpret_cast<const hkpCharacterProxyJob&>(jobIn);
	
	//
	// Bring in the jobHeader. This will be done by either DMA (if we are on SPU) or by simply redirecting (if we are on PPU/CPU).
	// 
#if defined (HK_PLATFORM_SPU)
	HK_ALIGN16( char jobHeaderBufferOnSpu[sizeof(hkpCharacterProxyJobHeader)] );
	const hkpCharacterProxyJobHeader* localJobHeader = HK_NULL;
	if ( job.m_sharedJobHeaderOnPpu )
	{
		localJobHeader = getJobHeaderFromPpu(job.m_sharedJobHeaderOnPpu, reinterpret_cast<hkpCharacterProxyJobHeader*>(jobHeaderBufferOnSpu));
	}
#else
	const hkpCharacterProxyJobHeader* localJobHeader = job.m_sharedJobHeaderOnPpu;
#endif
	
	switch( job.m_jobSubType )
	{
		case hkpCharacterProxyJob::CHARACTER_PROXY_JOB_INTEGRATE:
		{
			const hkpCharacterProxyIntegrateJob& characterProxyJob = reinterpret_cast<const hkpCharacterProxyIntegrateJob&>(job);
			
			localJobHeader->m_openJobs--;
			if ( localJobHeader->m_openJobs == 0 )
			{
				// Release the semaphore to indicate the job is complete
				characterProxyJob.atomicIncrementAndReleaseSemaphore();
			}
			break;
		}
		default:
		{
			HK_ASSERT2(0x1f5199b6, false, "Unknown job type in hkpCharacterProxyJobQueueUtils::finishCollisionJob");
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
 * Havok SDK - Base file, BUILD(#20130912)
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
