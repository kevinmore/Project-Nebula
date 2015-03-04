/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include  <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>
#include  <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobQueueUtils.h>
#include  <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleIntegrateJob.h>

HK_COMPILE_TIME_ASSERT( sizeof( hkpVehicleJob ) <= sizeof( hkJobQueue::JobQueueEntry ) );

hkJobQueue::JobPopFuncResult HK_CALL hkpVehicleJobQueueUtils::popVehicleJob  ( hkJobQueue& queue,
																				hkJobQueue::DynamicData* data,
																				hkJobQueue::JobQueueEntry& jobIn,
																				hkJobQueue::JobQueueEntry& jobOut  )
{
	hkString::memCpy16NonEmpty( &jobOut, &jobIn, sizeof( hkJobQueue::JobQueueEntry ) >> 4 );

	hkpVehicleJob& job = reinterpret_cast< hkpVehicleJob& >( jobIn );

	//
	// Handle the different jobs.
	//
	switch( job.m_jobSubType )
	{
		case hkpVehicleJob::VEHICLE_JOB_INTEGRATE:
		{
			hkpVehicleIntegrateJob& onQueue = static_cast     < hkpVehicleIntegrateJob& >( job );
			hkpVehicleIntegrateJob& out     = reinterpret_cast< hkpVehicleIntegrateJob& >( jobOut );
			return onQueue.popJobTask( out );
		}
		default:
		{
			break;
		}
	}

	return hkJobQueue::POP_QUEUE_ENTRY;
}

hkJobQueue::JobCreationStatus HK_CALL hkpVehicleJobQueueUtils::finishVehicleJob( hkJobQueue& queue,
																					hkJobQueue::DynamicData* data,
																					const hkJobQueue::JobQueueEntry& jobIn,
																					hkJobQueue::JobQueueEntryInput& newJobCreated )
{
	return hkJobQueue::NO_JOB_CREATED;
}

#ifndef HK_PLATFORM_SPU

static hkJobQueue::ProcessJobFunc s_vehicleProcessFuncs[ hkpVehicleJob::VEHICLE_JOB_END ];


void hkpVehicleJobQueueUtils::registerWithJobQueue( hkJobQueue* jobQueue )
{
#if defined( HK_PLATFORM_MULTI_THREAD ) && ( HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED )

	hkJobQueue::hkJobHandlerFuncs jobHandlerFuncs;
	jobHandlerFuncs.m_popJobFunc = popVehicleJob;
	jobHandlerFuncs.m_finishJobFunc = finishVehicleJob;

	jobHandlerFuncs.initProcessJobFuncs( s_vehicleProcessFuncs, HK_COUNT_OF( s_vehicleProcessFuncs ) ) ;

	jobHandlerFuncs.registerProcessJobFunc( hkpVehicleJob::VEHICLE_JOB_INTEGRATE, hkVehicleIntegrateJob );

	jobQueue->registerJobHandler( HK_JOB_TYPE_VEHICLE, jobHandlerFuncs );

#if defined(HK_PLATFORM_HAS_SPU)

	HK_WARN_ONCE( 0x3058cb47, "This job has only been implememented on the PPU. Performance loss likely." );

#endif

#endif

}
#endif // ifndef HK_PLATFORM_SPU

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
