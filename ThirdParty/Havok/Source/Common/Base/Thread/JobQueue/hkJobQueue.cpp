/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#if defined (HK_PLATFORM_SPU)
#	include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#	include <Common/Base/Spu/Printf/hkSpuPrintfUtil.h>
#endif

#if defined (HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Spu/Util/hkSpuUtil.h>
#endif

#if defined (HK_PLATFORM_PS3_SPU)
#	include <cell/spurs.h>
uint64_t hclSpuOverlayTable;
#endif

#include <Common/Base/Thread/JobQueue/hkJobQueue.h>
#include <Common/Base/System/hkBaseSystem.h>

#if defined (HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Thread/Job/ThreadPool/Spu/hkSpuJobThreadPool.h>
#endif

HK_COMPILE_TIME_ASSERT(sizeof(hkJobQueue::JobQueueEntry) == hkJobQueue::JOB_QUEUE_ENTRY_SIZE);


//#define JOB_QUEUE_PRINTF(A) printf("Thread: %d: ", HK_THREAD_LOCAL_GET(hkThreadNumber)); printf(A)

//#include <stdio.h>
//void hkPrintJobQueue( hkJobQueue* queue, hkJobQueue::DynamicData* data )
//{
//#if defined HK_PLATFORM_SPU
//	printf("SPU %d\n", hkThreadNumber );
//#else
//	printf("Thread %d", HK_THREAD_LOCAL_GET(hkThreadNumber) );
//
//	for (int i = 0; i < queue->m_numJobQueues; ++i)
//	{
//		printf("\nQ%d\t", i);
//		for (int j = 0; j < data->m_jobQueue[i].getSize(); ++j)
//		{
//			hkJobQueue::JobQueueEntry entry;
//			data->m_jobQueue[i].peek( entry );
//			printf("Type: %d SubType: %d SPU: %d\n",entry.m_jobType, entry.m_jobSubType, entry.m_jobSpuType ) ;
//		}
//	}
//#endif
//}

#if defined(HK_PLATFORM_SPU)
	bool hkJobQueue::m_outOfMemory = false;
	extern hkSpuTaskParams g_taskParams;
#endif


#if !defined(HK_PLATFORM_SPU)

hkJobQueueHwSetup::hkJobQueueHwSetup() 
{ 
	m_cellRules = PPU_CANNOT_TAKE_SPU_TASKS; 
	hkHardwareInfo info;
	hkGetHardwareInfo(info);
	m_numCpuThreads = info.m_numThreads;
	m_spuSchedulePolicy = SEMAPHORE_WAIT_OR_SWITCH_ELF;
#ifdef HK_PLATFORM_HAS_SPU
	m_noSpu = false;
#else
	m_noSpu = true;
#endif
}


hkJobQueue::JobPopFuncResult HK_CALL defaultPopDispatchFunc( hkJobQueue& queue, hkJobQueue::DynamicData* data, hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntry& jobOut )
{
	return queue.m_jobFuncs[jobIn.m_jobType].m_popJobFunc(queue, data, jobIn, jobOut);
}

hkJobQueue::JobCreationStatus HK_CALL defaultFinishDispatchFunc( hkJobQueue& queue, hkJobQueue::DynamicData* data, const hkJobQueue::JobQueueEntry& jobIn, hkJobQueue::JobQueueEntryInput& newJobCreatedOut )
{
	return queue.m_jobFuncs[jobIn.m_jobType].m_finishJobFunc( queue, data, jobIn, newJobCreatedOut );
}


static hkJobQueue::JobPopFuncResult  HK_CALL hkDefaultPopJobFunc(	hkJobQueue& queue, 
																 hkJobQueue::DynamicData* data, 
																 hkJobQueue::JobQueueEntry& jobIn, 
																 hkJobQueue::JobQueueEntry& jobOut ) 
{ 
	HK_ASSERT3(0x9762fe35, 0, "Unregistered pop job function for job type " << jobIn.m_jobType );
	return hkJobQueue::POP_QUEUE_ENTRY; 
}

static hkJobQueue::JobCreationStatus HK_CALL hkDefaultFinishJobFunc(	hkJobQueue& queue, 
																	hkJobQueue::DynamicData* data, 
																	const hkJobQueue::JobQueueEntry& jobIn, 
																	hkJobQueue::JobQueueEntryInput& newJobCreatedOut ) 
{ 
	HK_ASSERT3(0x9762fe36, 0, "Unregistered finish job function for job type " << jobIn.m_jobType );
	return hkJobQueue::NO_JOB_CREATED; 
}

#if defined(HK_PLATFORM_HAS_SPU)
hkJobQueue* hkJobQueue::s_instance = 0;
#endif

hkJobQueue::hkJobQueue( const hkJobQueueCinfo& cinfo )
:	m_criticalSection(0),
	m_numJobTypes( cinfo.m_maxNumJobTypes ),
	m_externalJobProfiler(HK_NULL)
{
	HK_THREAD_LOCAL_SET( hkThreadNumber, 0);

	m_data = new DynamicData();
	m_queryRulesAreUpdated = false;
	m_data->m_outOfMemory = false;

	HK_ASSERT2(0x05342310, MAX_NUM_CPU_THREADS >= cinfo.m_jobQueueHwSetup.m_numCpuThreads, "Too many threads specified for platform");

	m_data->m_waitPolicy = WAIT_UNTIL_ALL_WORK_COMPLETE;

	m_hwSetup = cinfo.m_jobQueueHwSetup;

#if defined(HK_PLATFORM_HAS_SPU)
	m_data->m_numRunningSpus = 0;

	if ( s_instance )
	{
		HK_WARN( 0xf034de78, "You have multiple job queues, this means if you are running out of memory (hkFreelistAllocator::softLimit), the spus won't get signalled" );
	}
	s_instance = this;
#endif


	// Every time the queue is resized from the PPU, it will make sure enough free slots are available (which SPUs can then fill)
	// This capacity could be dynamically resized based on the number of elements in the SPU queue

	m_data->m_masterThreadFinishingFlags = 0;
	for (int i=0; i< m_numJobTypes; i++)
	{
		m_jobFuncs[i].m_numProcessJobFuncs = 0;
		m_jobFuncs[i].m_processJobFuncs = HK_NULL; // = hkDefaultProcessJobFunc;
		m_jobFuncs[i].m_popJobFunc     = hkDefaultPopJobFunc;
		m_jobFuncs[i].m_finishJobFunc  = hkDefaultFinishJobFunc;

		m_data->m_numActiveJobs[i] = 0;

		m_data->m_masterThreadFinishingFlags |= 1 << i;
	}

#if defined(HK_PLATFORM_HAS_SPU)
	hkString::memSet(m_spuElfs, -1, sizeof(m_spuElfs));
#endif

	m_popJobFunc = defaultPopDispatchFunc;
	m_finishJobFunc = defaultFinishDispatchFunc;

	m_numQueueSemaphores = 0;

	m_threadPool = HK_NULL;
	
	for (int i = 0 ; i < MAX_NUM_THREAD_TYPES; i++)
	{
		m_queueSemaphores[i] = HK_NULL;
	}

	// Initialize all values in m_cpuThreadIndexToSemaphoreIndex
	updateJobQueryRules( );

#if defined(HK_PLATFORM_PS3_PPU)
	// Initialize overlay table pointers
	for (int i = 0; i<HK_JOB_TYPE_MAX; ++i)
	{
		m_spuOverlays[i] = HK_NULL;
	}
#endif
}

hkJobQueue::~hkJobQueue()
{
	delete m_data;

#if defined(HK_PLATFORM_HAS_SPU)
	s_instance = HK_NULL;
#endif

	if (m_queryRulesAreUpdated)
	{
		for (int i = 0; i < MAX_NUM_THREAD_TYPES; i++)
		{
			delete m_queueSemaphores[i];
		}
	}

#if defined(HK_PLATFORM_PS3_PPU)
	// Deallocate any allocated overlay tables
	for (int i = 0; i<HK_JOB_TYPE_MAX; ++i)
	{
		if (m_spuOverlays[i]) hkAlignedDeallocate<char>((char*)m_spuOverlays[i]);
	}
#endif
}




void hkJobQueue::registerJobWithCpuThread( hkJobType jobType, hkJobSubType subType, int threadId )
{
	// only add the custom job if it differs from the existing ones
	if ( m_customJobSetup.getSize() >= 1 )
	{
		for ( int i = 0; i < m_customJobSetup.getSize(); i++ )
		{
			if ( ( m_customJobSetup[i].m_jobType == jobType ) &&
				( m_customJobSetup[i].m_jobSubType == subType ) &&
				( m_customJobSetup[i].m_threadId == threadId ) )
			{
				return;
			}
		}
	}

	CustomJobTypeSetup& setup = m_customJobSetup.expandOne();
	setup.m_jobType = jobType;
	setup.m_jobSubType = subType;
	setup.m_threadId = threadId;
	updateJobQueryRules();
}

void hkJobQueue::updateJobQueryRules()
{
	int numSharedCaches = m_hwSetup.m_threadIdsSharingCaches.getSize() == 0 ? 1 : m_hwSetup.m_threadIdsSharingCaches.getSize();
	m_numCustomJobs = m_customJobSetup.getSize();

	if ( !m_hwSetup.m_noSpu )
	{
		m_cpuCacheQueuesBegin = m_numJobTypes;
	}
	else
	{
		m_cpuCacheQueuesBegin = 0;
	}
	m_cpuCustomQueuesBegin = m_cpuCacheQueuesBegin + numSharedCaches;
	m_cpuTypesQueuesBegin = m_cpuCustomQueuesBegin + m_customJobSetup.getSize();
	m_numJobQueues = m_cpuTypesQueuesBegin + m_numJobTypes;

	// Set capacity for standard CPU queues
	for (int  qIdx = m_cpuCacheQueuesBegin ; qIdx < m_cpuCustomQueuesBegin ; qIdx++)
	{
		m_data->m_jobQueue[qIdx].setCapacity( 128 );
	}


	// Duplicate value - only here for readability
	m_cpuSemaphoreBegin = m_cpuCacheQueuesBegin;

	m_directMapSemaphoreEnd = m_cpuTypesQueuesBegin;

	//
	// First setup the SPU queues
	//
#if defined HK_PLATFORM_HAS_SPU
	for (int i = 0; i < m_numJobTypes; ++i)
	{
		hkInt8* table = m_nextQueueToGet[i];		
		if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF )
		{
			// SPU version - take physics jobs first, then own queue, then the other queues
			*(table++) = HK_JOB_TYPE_DYNAMICS;
			*(table++) = HK_JOB_TYPE_COLLIDE;
			*(table++) = HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND;
		}
		*(table++) = (hkUint8)i;
		for (int j = 0; j < m_numJobTypes; ++j)
		{
			if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF && 
				( j == HK_JOB_TYPE_COLLIDE || j == HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND || j == HK_JOB_TYPE_DYNAMICS ) )
			{
				continue;
			}
			if ( ( i != j ) )
			{
				*(table++) = (hkUint8)j;
			}
		}
		// Place marker at end
		*(table++) = -1;
	}
#endif

	//
	// Next, setup the CPU queues, using specified cache groupings, if present
	//

#if defined HK_PLATFORM_WIN32

	if ( m_hwSetup.m_threadIdsSharingCaches.getSize() > 0)
	{
		for (int i = 0; i < m_hwSetup.m_numCpuThreads; ++i)
		{
			m_cpuThreadIndexToSemaphoreIndex[i] = -1;
		}
		for ( int i = 0; i < m_hwSetup.m_threadIdsSharingCaches.getSize(); ++i )
		{
			for ( int j = 0; j < m_hwSetup.m_threadIdsSharingCaches[i].getSize(); ++j )
			{
				m_cpuThreadIndexToSemaphoreIndex[ m_hwSetup.m_threadIdsSharingCaches[i][j] ] = hkUint8(m_cpuCacheQueuesBegin + i);
			}
		}
		for (int i = 0; i < m_hwSetup.m_numCpuThreads; ++i)
		{
			HK_ASSERT2(0x278ff346, m_cpuThreadIndexToSemaphoreIndex[i] != -1, "Incomplete thread cache specification. You must specify caches all threads in a contiguous block up to m_hwSetup.m_numCpuThreads" );
		}
	}
#endif

	if ( m_hwSetup.m_threadIdsSharingCaches.getSize() == 0)
	{
		for (int i = 0; i < m_hwSetup.m_numCpuThreads; ++i)
		{
			m_cpuThreadIndexToSemaphoreIndex[i] = hkUint8(m_cpuCacheQueuesBegin);
		}
	}



	for (int i = 0; i < numSharedCaches; ++i)
	{
		hkInt8* table = m_nextQueueToGet[i + m_cpuCacheQueuesBegin];
		*(table++) = (hkUint8)(i + m_cpuCacheQueuesBegin);

		#if defined HK_PLATFORM_WIN32
			// Next, assign a thread to the shared cache
			for (int j = 0; j < m_hwSetup.m_threadIdsSharingCaches.getSize(); ++j)
			{
				// NOTE: This is not always the best thing to do - it might be better to take jobs from type queues first
				if (i != j)
				{
					*(table++) = hkUint8(j + m_cpuCacheQueuesBegin);
				}
			}
		#endif

		// Next look in all CPU type queues starting from the first type
		for (int j = 0; j < m_numJobTypes; ++j)
		{
			*(table++) = hkUint8(j + m_cpuTypesQueuesBegin);
		}
		if ( m_hwSetup.m_cellRules == hkJobQueueHwSetup::PPU_CAN_TAKE_SPU_TASKS )
		{
			for (int j = 0; j < m_numJobTypes; ++j)
			{
				*(table++) = hkUint8(j);
			}
		}

		// Finally look in the custom job queues 
		for ( int j = 0; j < m_numCustomJobs; ++j)
		{
			*(table++) = hkUint8(j + m_cpuCustomQueuesBegin);
		}


		// Place marker at end
		*(table++) = -1;
	}

	//
	// Handle custom jobs
	//
	for ( int i = 0; i < m_numCustomJobs; ++i)
	{
		m_customJobs[i].m_jobType = m_customJobSetup[i].m_jobType;
		m_customJobs[i].m_jobSubType = m_customJobSetup[i].m_jobSubType;
		m_customJobs[i].m_queueId = hkUint8( i + m_cpuCustomQueuesBegin );
		HK_ASSERT2(0x27836482, m_customJobSetup[i].m_threadId <= m_hwSetup.m_numCpuThreads, "You cannot register a job with a thread with an id greater than the job queue is set up to handle.");
		HK_ASSERT2(0x27836434, m_cpuThreadIndexToSemaphoreIndex[ m_customJobSetup[i].m_threadId ] < m_cpuCustomQueuesBegin, "Multiple custom jobs per thread not currently supported");
		
		int originalSemaphoreIndex = m_cpuThreadIndexToSemaphoreIndex[m_customJobSetup[i].m_threadId];
		m_cpuThreadIndexToSemaphoreIndex[ m_customJobSetup[i].m_threadId ] = m_customJobs[i].m_queueId;

		// Fill out the table entry for this custom job
		hkInt8* table = m_nextQueueToGet[i + m_cpuCustomQueuesBegin];
		*(table++) = (hkUint8)(i + m_cpuCustomQueuesBegin);

		// Copy values from original list
		for ( int j = 0; m_nextQueueToGet[originalSemaphoreIndex][j] != -1; ++j )
		{
			if (m_nextQueueToGet[originalSemaphoreIndex][j] != m_customJobs[i].m_queueId)
				*(table++) = m_nextQueueToGet[originalSemaphoreIndex][j];
		}
		*(table++) = -1;

	}

	// Setup master thread semaphore

	int numSemaphores = m_cpuTypesQueuesBegin;

	m_masterThreadQueue = m_cpuThreadIndexToSemaphoreIndex[0];
	bool newSemaphoreNeeded = false;
	for (int i = 1; i < m_hwSetup.m_numCpuThreads; ++i)
	{
		if ( m_cpuThreadIndexToSemaphoreIndex[i] == m_cpuThreadIndexToSemaphoreIndex[0] )
		{
			newSemaphoreNeeded = true;
			break;
		}
	}
	if (newSemaphoreNeeded)
	{
		m_cpuThreadIndexToSemaphoreIndex[0] = (hkUint8)numSemaphores;
		numSemaphores++;
	}

	// Note : Sempahores are allocated and deallocate here for CPU threads as these change dynamically
	// Semaphore allocation for SPU (one per job type) are allocated when jobs are registered for each SPU
	if (m_queryRulesAreUpdated)
	{
		// deallocate cpu and custom job semaphores
		for (int i = m_cpuCacheQueuesBegin ; i < m_numQueueSemaphores; i++)
		{
			delete m_queueSemaphores[i];
			m_queueSemaphores[i] = HK_NULL;
		}
	}

	m_numQueueSemaphores = numSemaphores;
	HK_ASSERT2(0x44443331, m_numQueueSemaphores < MAX_NUM_THREAD_TYPES, "Max num thread types exceeded");

	for (int i =0; i < m_numQueueSemaphores; i++)
	{
		m_data->m_numThreadsWaiting[i] = 0;

		// Only allocate semaphores for CPU queues 
		if ( i >= m_cpuCacheQueuesBegin)
		{
			m_queueSemaphores[i] = new hkSemaphoreBusyWait( 0,1000 );
		}
	}

	m_queryRulesAreUpdated = true;
}


static hkJobQueue::JobStatus HK_CALL hkDefaultProcessJobFunc( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& jobInOut )
{
	HK_ASSERT2(0x9762fe34, 0, "Unregistered process job function" );
	return jobQueue.finishJobAndGetNextJob( &jobInOut, jobInOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
}

void hkJobQueue::setQueueCapacityForJobType(hkJobType jobType, int queueCapacity )
{
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );

#if defined HK_PLATFORM_HAS_SPU
	data->m_jobQueue[jobType].setCapacity(queueCapacity);
#endif
	data->m_jobQueue[jobType + m_cpuTypesQueuesBegin].setCapacity(queueCapacity);

	// This is a hack, but needed (this is only used for physics broad phase anyway)
	for (int i = 0; i < m_numCustomJobs; ++i)
	{
		data->m_jobQueue[m_customJobs[i].m_queueId].setCapacity(queueCapacity);
	}

	// Also resize for post collide jobs
	for (int qIdx = m_cpuCacheQueuesBegin; qIdx < m_cpuCustomQueuesBegin; qIdx++ )
	{
		data->m_jobQueue[ qIdx ].setCapacity( queueCapacity );
	}

	unlockQueue( data );
}

void hkJobQueue::setQueueCapacityForCpuCache( int queueCapacity )
{
#if defined HK_PLATFORM_HAS_SPU
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );
	data->m_jobQueue[m_cpuCacheQueuesBegin].setCapacity(queueCapacity);
	unlockQueue( data );
#endif
}

void hkJobQueue::setExternalProfiler(hkExternalJobProfiler* p)
{
	m_externalJobProfiler = p; 
}

hkJobQueue::JobStatus hkJobQueue::processAllJobs( bool addTimers )
{
	hkJobQueue::JobQueueEntry job;


	hkJobQueue::JobStatus jobStatus = getNextJob( job);
#define MONITOR_COMMAND_TIMER_BEGIN "Tt"
	const char* timerName = MONITOR_COMMAND_TIMER_BEGIN"Unknown";
	while ( jobStatus == hkJobQueue::GOT_NEXT_JOB )
	{
		hkJob& typedJob = reinterpret_cast<hkJob&>(job);

		// Cache job type because it may get overwritten in the m_processJobFuncs call
		const hkJobType jobType = typedJob.m_jobType;

		HK_ASSERT2(0xafe1a255, jobType < m_numJobTypes, "Invalid job type. Type exceeds allowed m_numJobTypes.");
		HK_ASSERT2(0xafe1a256, typedJob.m_jobSubType < m_jobFuncs[jobType].m_numProcessJobFuncs, "Invalid job type. No function registered");

		switch (jobType)
		{
		case HK_JOB_TYPE_DYNAMICS:			timerName = MONITOR_COMMAND_TIMER_BEGIN"Physics 2012";			break;
		case HK_JOB_TYPE_COLLIDE_STATIC_COMPOUND:
		case HK_JOB_TYPE_COLLIDE:			timerName = MONITOR_COMMAND_TIMER_BEGIN"Physics 2012";			break;
		case HK_JOB_TYPE_COLLISION_QUERY:	timerName = MONITOR_COMMAND_TIMER_BEGIN"Collision Query";		break;
		case HK_JOB_TYPE_RAYCAST_QUERY:		timerName = MONITOR_COMMAND_TIMER_BEGIN"RayCast Query";			break;
		case HK_JOB_TYPE_ANIMATION_SAMPLE_AND_COMBINE:	timerName = MONITOR_COMMAND_TIMER_BEGIN"Animation Sample and Combine";	break;
		case HK_JOB_TYPE_ANIMATION_SAMPLE_AND_BLEND:timerName = MONITOR_COMMAND_TIMER_BEGIN"Animation Sample and Blend";	break;
		case HK_JOB_TYPE_ANIMATION_MAPPING:	timerName = MONITOR_COMMAND_TIMER_BEGIN"Animation Mapping";		break;
		case HK_JOB_TYPE_BEHAVIOR:			timerName = MONITOR_COMMAND_TIMER_BEGIN"Behavior";			break;
		case HK_JOB_TYPE_CLOTH:				timerName = MONITOR_COMMAND_TIMER_BEGIN"Cloth";				break;
		case HK_JOB_TYPE_AI_PATHFINDING:	timerName = MONITOR_COMMAND_TIMER_BEGIN"Pathfinding Jobs";	break;
		case HK_JOB_TYPE_AI_VOLUME_PATHFINDING:	timerName = MONITOR_COMMAND_TIMER_BEGIN"Pathfinding Jobs";	break;
		case HK_JOB_TYPE_AI_DYNAMIC:		timerName = MONITOR_COMMAND_TIMER_BEGIN"AI Dynamic Jobs";	break;
		case HK_JOB_TYPE_AI_LOCAL_STEERING:	timerName = MONITOR_COMMAND_TIMER_BEGIN"LocalSteering Jobs";break;
		case HK_JOB_TYPE_AI_GENERATION:		timerName = MONITOR_COMMAND_TIMER_BEGIN"AI Generation";		break;
		case HK_JOB_TYPE_DESTRUCTION:		timerName = MONITOR_COMMAND_TIMER_BEGIN"Destruction";		break;
		case HK_JOB_TYPE_CHARACTER_PROXY:	timerName = MONITOR_COMMAND_TIMER_BEGIN"Character Proxy";	break;
		case HK_JOB_TYPE_VEHICLE:			timerName = MONITOR_COMMAND_TIMER_BEGIN"Vehicle";			break;
		case HK_JOB_TYPE_USER_0:			timerName = MONITOR_COMMAND_TIMER_BEGIN"UserJob";			break;
		default:							timerName = MONITOR_COMMAND_TIMER_BEGIN"Other";				break;
		}

		HK_ON_MONITORS_ENABLED(
			hkMonitorStream& mStream = hkMonitorStream::getInstance();
			if ( addTimers && mStream.memoryAvailable() )
			{
				hkMonitorStream::TimerCommand* h = reinterpret_cast<hkMonitorStream::TimerCommand*>(mStream.getEnd());
				h->m_commandAndMonitor = timerName;
				h->setTime();											
				mStream.setEnd( (char*)(h+1) );		
			}
		);

		if (m_externalJobProfiler) m_externalJobProfiler->onStartJob(jobType, typedJob.m_jobSubType );

		jobStatus = m_jobFuncs[jobType].m_processJobFuncs[typedJob.m_jobSubType]( *this, job );

		if (m_externalJobProfiler) m_externalJobProfiler->onEndJob(jobType);

		HK_ON_MONITORS_ENABLED(
			if ( addTimers )
			{
				HK_TIMER_END2(mStream);
			}
		)
		// Call finish and get next here, don't get the process functions to do it
		// Need to clean up logic with finish
	}

	return jobStatus;
}

#if defined(HK_PLATFORM_HAS_SPU)

void hkJobQueue::registerSpuElf( hkJobType jobType, void* spuElf )
{
	// If the queue isn't configured to run SPU jobs, do nothing when registering an ELF.
	if (m_hwSetup.m_noSpu)
	{
		return;
	}

	m_spuElfs[jobType] = spuElf;

	// Set the queue capacity for registered spu jobs
	// SPU cannot resize the queue when they add jobs
	setQueueCapacityForJobType( jobType, 128 );

	// Allocate semaphore for SPU job type
	if ( !m_queueSemaphores[ jobType ] )
	{
		m_queueSemaphores[ jobType ] = new hkSemaphoreBusyWait( 0,1000 );
	}
}

void hkJobQueue::registerSpuOverlayTable( hkJobType jobType, void* overlayTable )
{
	if (m_spuOverlays[jobType] != HK_NULL)
	{
		hkAlignedDeallocate<char>((char*)m_spuOverlays[jobType]);
	}
	m_spuOverlays[jobType] = overlayTable;
}


void* hkJobQueue::getInitialElf()
{
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	JobQueueEntry entry;

	DynamicData* data = lockQueue( dynamicDataStorage );

	for (int i = 0; i < m_numJobTypes; ++i)
	{
		if (!data->m_jobQueue[i].isEmpty())
		{
			HK_ASSERT2(0x987eee56, m_spuElfs[i] != (void*)-1, "Job on queue that no elf is registered to handle");
			unlockQueue( data );
			return m_spuElfs[i];
		}
	}

	// No jobs on any queues, just return the first registered elf we find.
	for (int i = 0; i < m_numJobTypes; ++i)
	{
		if ( m_spuElfs[i] != (void*)-1 )
		{
			unlockQueue( data );
			return m_spuElfs[i];
		}
	}
	HK_ASSERT2(0x987eee52, 0, "No elfs registered.");
	return HK_NULL;
}

void* hkJobQueue::getRegisteredSpuElf( hkJobType jobType )
{
	return m_spuElfs[ jobType ];
}

#endif // defined(HK_PLATFORM_HAS_SPU)



#endif // !defined(HK_PLATFORM_SPU) 


hkJobQueue::QueueIndex hkJobQueue::getQueueIndexForJob( const hkJob& job )
{
	// First check if its a job for a special queue
	for (int i = 0; i < m_numCustomJobs; ++i)
	{
		// TODO - One compare
		if ( ( job.m_jobType == m_customJobs[i].m_jobType)
			&& ( job.m_jobSubType == m_customJobs[i].m_jobSubType) )
		{
			return m_customJobs[i].m_queueId;
		}
	}
#if defined HK_PLATFORM_WIN32
	// PC only
	if (job.m_threadAffinity != -1)
	{
		// TODO
		//return m_cpuThreadIndexToSemaphoreIndex(job.m_threadAffinity); // The semaphore is also the index of the first queue
	}
#endif

#if defined HK_PLATFORM_HAS_SPU
	HK_ASSERT(0x7d240ae, job.m_jobSpuType != HK_JOB_SPU_TYPE_INVALID);
	if (job.m_jobSpuType == HK_JOB_SPU_TYPE_ENABLED )
	{
		return job.m_jobType;
	}
#endif

#if defined HK_PLATFORM_SPU
	// This allows us not to DMA the full dynamic data onto the SPU - just the first 256 bytes. The CPU types queues
	// are never referenced by the SPU because of this - when the SPU creates a job it always puts it onto the cache queue.
	return m_cpuCacheQueuesBegin;
#else
	return m_cpuTypesQueuesBegin + job.m_jobType;
#endif
}

hkJobQueue::DynamicData* hkJobQueue::lockQueue( char* dynamicDataStorage )
{
	//HK_TIME_CODE_BLOCK("Lock Q", HK_NULL);

#if !defined(HK_PLATFORM_SPU)
	// HK_ASSERT( 0xf03ef576, !m_criticalSection.haveEntered() );
	m_criticalSection.enter();
	return m_data;
#else
	HK_CHECK_ALIGN16( dynamicDataStorage );
	hkCriticalSection::enter( m_criticalSection.m_this );
	hkSpuDmaManager::getFromMainMemory( dynamicDataStorage, m_data, sizeof(DynamicData), hkSpuDmaManager::READ_WRITE);
	// we cannot combine this call with the above function, as we are waiting for all dmas
	hkSpuDmaManager::waitForAllDmaCompletion();
	DynamicData* dd = (DynamicData*)dynamicDataStorage;
	m_outOfMemory = dd->m_outOfMemory;
	return dd;
#endif
}

void hkJobQueue::unlockQueue( DynamicData* dynamicDataStorage )
{
	HK_CHECK_ALIGN16( dynamicDataStorage );

	//HK_TIME_CODE_BLOCK("Unlock Q", HK_NULL);

#if !defined (HK_PLATFORM_SPU)
	m_criticalSection.leave();
#else
	hkSpuDmaManager::putToMainMemoryAndWaitForCompletion( m_data, dynamicDataStorage, sizeof(DynamicData), hkSpuDmaManager::WRITE_BACK);
	HK_SPU_DMA_PERFORM_FINAL_CHECKS                     ( m_data, dynamicDataStorage, sizeof(DynamicData));
	hkCriticalSection::leave( m_criticalSection.m_this );
#endif
}


hkJobQueue::JobStatus hkJobQueue::getNextJob( JobQueueEntry& job, WaitStatus waitStatus )
{ 
	return finishJobAndGetNextJob( HK_NULL, job, waitStatus); 
}

void hkJobQueue::releaseOneWaitingThread( DynamicData* data )
{
	// wake up ppu threads first
	int semaphoreIndex;
	{
		bool cpuJobAvailable = false;
		int i = m_cpuSemaphoreBegin; // The start of CPU semaphores

		// First check the custom queues and the shared cache queues
		for (; i < m_directMapSemaphoreEnd; ++i)
		{
			// Each of these queues has an associated semaphore, and threads waiting on those semaphores
			// will want jobs from those queues first
			if ( !data->m_jobQueue[i].isEmpty() )
			{
				cpuJobAvailable = true;
				if( data->m_numThreadsWaiting[i] )
				{
					semaphoreIndex = i;
					goto releaseSemaphore;
				}
			}
		}
		// Next check the rest of the queues (the CPU types)
		for(; i < m_numJobQueues; ++i)
		{
			if ( !data->m_jobQueue[i].isEmpty() )
			{
				cpuJobAvailable = true;
				break;
			}
		}

		if ( cpuJobAvailable )
		{
			for (semaphoreIndex = m_cpuSemaphoreBegin; semaphoreIndex < m_numQueueSemaphores; semaphoreIndex++)
			{
				if ( data->m_numThreadsWaiting[semaphoreIndex])
				{
					goto releaseSemaphore;
				}
			}
		}
	}

#if defined HK_PLATFORM_HAS_SPU	
	// Wake up SPU
	{
		int spuWaiting = -1;
		bool spuJobAvailable = false;
		for (int i = 0; i < m_cpuSemaphoreBegin; i++)
		{
			spuJobAvailable |= (!data->m_jobQueue[i].isEmpty());
			if ( data->m_numThreadsWaiting[i] )
			{
				spuWaiting = i;
				if ( !data->m_jobQueue[i].isEmpty() )
				{
					semaphoreIndex = i;
					goto releaseSemaphore;
				}
			}
		}
		if (spuJobAvailable && spuWaiting != -1)
		{
			semaphoreIndex = spuWaiting;
			goto releaseSemaphore;
		}
	}

#endif // HK_PLATFORM_HAS_SPU

	return;

releaseSemaphore:
	data->m_numThreadsWaiting[semaphoreIndex]--;
	hkSemaphoreBusyWait::release( m_queueSemaphores[semaphoreIndex] );
}

void hkJobQueue::checkQueueAndReleaseOneWaitingThread( QueueIndex queueIndex, DynamicData* data )
{
	HK_ON_DEBUG( Queue* queue = &data->m_jobQueue[queueIndex] );
	HK_ASSERT( 0xf0323454, !queue->isEmpty() );

	// There is a (kind of) one to one mapping between semaphores and queues. There are more queues than semaphores
	// but they are later in the list of queues. A thread waiting on a semaphore will always look first in a queue
	// with the same index as the semaphore.
	if ( (queueIndex < m_numQueueSemaphores) && (data->m_numThreadsWaiting[queueIndex] > 0) )
	{		
		data->m_numThreadsWaiting[queueIndex]--;
		hkSemaphoreBusyWait::release( m_queueSemaphores[queueIndex] );
	}
	else
	{
		releaseOneWaitingThread( data );
	}
}

void hkJobQueue::addJobQueueLocked( DynamicData* data, const JobQueueEntry& job, JobPriority priority )
{
	QueueIndex queueIndex = getQueueIndexForJob( (hkJob&)job );
	HK_ASSERT(0xf032e454, queueIndex >= 0 && queueIndex < MAX_NUM_QUEUES );

	// Add the jobEntry to the queue
	Queue& queue = data->m_jobQueue[queueIndex];
	if ( priority == JOB_HIGH_PRIORITY )
	{
		queue.enqueueInFront( job );
	}
	else
	{
		queue.enqueue( job );
	}
	checkQueueAndReleaseOneWaitingThread( queueIndex, data );
}

void hkJobQueue::addJob( JobQueueEntry& job, JobPriority priority )
{
#if !defined HK_PLATFORM_SPU
	HK_ASSERT2(0x67556565, HK_THREAD_LOCAL_GET(hkThreadNumber) < m_hwSetup.m_numCpuThreads, "More thread using job queue than Job queue was initialized to handle");
#endif

	// Temporary storage used on SPU to DMA the DynamicData into it
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );
	addJobQueueLocked( data, job, priority );
	unlockQueue( data );
}

void hkJobQueue::addJob( hkJob& job, JobPriority priority )
{
	// XXX temp - change hkQueue to take a size. 
	JobQueueEntry entry;
	hkString::memCpy(&entry, &job, job.m_size );
	addJob( entry, priority );
}

#if !defined (HK_PLATFORM_SPU)
void hkJobQueue::addJobBatch( const hkArrayBase<hkJob*>& jobs, JobPriority priority )
{
#if !defined HK_PLATFORM_SPU
	HK_ASSERT2(0x67556565, HK_THREAD_LOCAL_GET(hkThreadNumber) < m_hwSetup.m_numCpuThreads, "More thread using job queue than Job queue was initialized to handle");
#endif
	HK_TIME_CODE_BLOCK("AddJobBatch", HK_NULL);

	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );
	{
		for(int i = 0; i < jobs.getSize(); i++)
		{
			JobQueueEntry entry;
			hkString::memCpy(&entry, jobs[i], jobs[i]->m_size );

			QueueIndex queueIndex = getQueueIndexForJob( entry );

			// Add the jobEntry to the queue
			if ( priority == JOB_HIGH_PRIORITY )
			{
				data->m_jobQueue[queueIndex].enqueueInFront( entry );
			}
			else
			{
				data->m_jobQueue[queueIndex].enqueue( entry );
			}

			checkQueueAndReleaseOneWaitingThread( queueIndex, data );
		}
	}
	unlockQueue( data );
}


void hkJobQueue::setWaitPolicy( WaitPolicy waitPolicy )
{
	HK_ASSERT2(0x5454dd52, HK_THREAD_LOCAL_GET(hkThreadNumber) == 0, "Only the master thread may call this function");
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );
	m_data->m_waitPolicy = waitPolicy;
	if (waitPolicy == WAIT_UNTIL_ALL_WORK_COMPLETE)
	{
		releaseWaitingThreads( data );
	}
	unlockQueue( data );
}

/// Get the wait policy. See comments for setWaitPolicy for details.
hkJobQueue::WaitPolicy hkJobQueue::getWaitPolicy()
{
	HK_ASSERT2(0x5454dd52, HK_THREAD_LOCAL_GET(hkThreadNumber) == 0, "Only the master thread may call this function");
	return m_data->m_waitPolicy;
}

int hkJobQueue::getMasterThreadFinishingFlags()
{
	HK_ASSERT2(0x5454dd52, HK_THREAD_LOCAL_GET(hkThreadNumber) == 0, "Only the master thread may call this function");
	return m_data->m_masterThreadFinishingFlags;
}

void hkJobQueue::setMasterThreadFinishingFlags( int flags )
{
	HK_ASSERT2(0x5454dd52, HK_THREAD_LOCAL_GET(hkThreadNumber) == 0, "Only the master thread may call this function");
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );

	m_data->m_masterThreadFinishingFlags = flags;
	if ( data->m_numThreadsWaiting[m_cpuThreadIndexToSemaphoreIndex[0]] > 0 )
	{
		// Release the one master thread.		
		data->m_numThreadsWaiting[m_cpuThreadIndexToSemaphoreIndex[0]]--;
		hkSemaphoreBusyWait::release( m_queueSemaphores[m_cpuThreadIndexToSemaphoreIndex[0]] );
	}

	unlockQueue( data );
}


#endif

HK_FORCE_INLINE hkJobQueue::QueueIndex hkJobQueue::findNextJob( JobQueueEntry& jobOut, DynamicData* data )
{
	// WARNING: THIS FUNCTION MUST ALWAYS BE CALLED WHEN THE MT CRITICAL SECTION IS LOCKED
	// TODO - add isLocked to critical section and add assert
	// 
	//	check queues based on rules
	// 
	QueueIndex queueIndex;
	Queue* queue;
	{
#if !defined (HK_PLATFORM_SPU)
		// This is necessary because the thread 0 can be the master thread, in which case it may point to a different queue than
		// the semaphore. This if statement saves us having a map for all threads.
		int index = HK_THREAD_LOCAL_GET(hkThreadNumber) == 0 ? m_masterThreadQueue : getSemaphoreIndex(HK_THREAD_LOCAL_GET(hkThreadNumber));
#else
		int index = getSemaphoreIndex(HK_THREAD_LOCAL_GET(hkThreadNumber));
#endif
		hkInt8* queueIndices = m_nextQueueToGet[ index ];
		while ( ( queueIndex = queueIndices[0] ) >=0 )
		{
			queue = &data->m_jobQueue[ queueIndex ];
			queueIndices++;
			if ( !queue->isEmpty() )
			{
				goto GOT_JOB;
			}
		}
	}
	return -1;

GOT_JOB:
#if defined HK_PLATFORM_SPU
	const int thisQueueIndex = HK_THREAD_LOCAL_GET(hkThreadNumber);
	if (queueIndex != thisQueueIndex)
	{
		HK_ASSERT2(0x981735f4, (int)m_spuElfs[queueIndex] != -1, "Job found for unregistered elf" );
		if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::RULE_BASED_QUIT_OR_TIMEOUT )
		{
			if ( data->m_spuStats[queueIndex].m_runningTasks == 0 && data->m_numRunningSpus == g_taskParams.m_maxNumSpus )
			{
				return -2;
			}
			return -1;
		}
		return -2;
	}

#endif
	{
		HK_ALIGN16(JobQueueEntry job);
		queue->dequeue(job);
		
		if ( m_popJobFunc(*this, data, job, jobOut) == DO_NOT_POP_QUEUE_ENTRY )
		{
			queue->enqueueInFront(job);
			checkQueueAndReleaseOneWaitingThread( queueIndex,data );
		}
		data->m_numActiveJobs[job.m_jobType]++;
	}
	return queueIndex;			
}


HK_FORCE_INLINE hkBool hkJobQueue::allQueuesEmpty( hkJobQueue::DynamicData* data )
{
	int numJobs = 0;
	for (int i = 0; i < m_numJobQueues; i++)
	{
		numJobs += data->m_jobQueue[i].getSize();
	}
	return numJobs == 0;
}



HK_FORCE_INLINE int hkJobQueue::getSemaphoreIndex( int threadNumber )
{
#if defined HK_PLATFORM_SPU
	return threadNumber;
#else
	return m_cpuThreadIndexToSemaphoreIndex[ threadNumber ];
#endif
}

#if defined(HK_PLATFORM_PS3_SPU) || defined(HK_PLATFORM_PS3_PPU)

void hkJobQueue::spawnSpuTasks( DynamicData* data )
{	
#if defined HK_PLATFORM_PS3_PPU
	if ( m_threadPool == HK_NULL )
	{
		return;
	}
	int maxNumSpus = m_threadPool->getNumThreads();
#endif

#if defined HK_PLATFORM_PS3_SPU	
	int maxNumSpus = (int)g_taskParams.m_maxNumSpus;
#endif

	// try to create a new task	
	
	for ( int i = 0; i < HK_JOB_TYPE_MAX && data->m_numRunningSpus < maxNumSpus; ++i )
	{
		JobTypeSpuStats& stats = data->m_spuStats[i];
		stats.m_timeSinceLastTask++;

		if ( data->m_jobQueue[i].isEmpty() )
		{
			continue;
		}

		bool startCondition = false;
		if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::RULE_BASED_QUIT_OR_TIMEOUT )
		{
			startCondition = ( stats.m_waitingTasks == 0 && stats.m_timeSinceLastTask > data->m_numRunningSpus );
		}

		
#if defined HK_PLATFORM_PS3_PPU
		int tid;
		if ( startCondition && ( tid = m_threadPool->startSpursTask( (hkJobType)i, this ) ) >= 0 )
		{
#endif
#if defined HK_PLATFORM_PS3_SPU
		CellSpursTaskId tid;
		int ret;
		if ( startCondition && ( ret = cellSpursCreateTask2( cellSpursGetTasksetAddress(), &tid, (hkUlong)m_spuElfs[i], (qword&)g_taskParams, HK_NULL ) ) == 0 )
		{
#endif
			data->m_numRunningSpus++;
			stats.m_runningTasks++;
			stats.m_timeSinceLastTask = 0;
			break;
		}
	}
}

#endif

// assumes a locked queue
hkJobQueue::JobStatus hkJobQueue::findJobInternal( QueueIndex queueIndexOfNewJob, DynamicData* data, WaitStatus waitStatus, JobQueueEntry& jobOut )
{	
	// Try to find another job from available job queues 
	QueueIndex queueIndexOfFoundJob = findNextJob( jobOut, data );

#if defined(HK_PLATFORM_PS3_SPU) || defined(HK_PLATFORM_PS3_PPU)
	if ( m_hwSetup.m_spuSchedulePolicy != hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF )
	{
		spawnSpuTasks( data );
	}
#endif

	// If we have just added a job prior to calling this function (from addJob... or finishJob...)
	// then release a thread. Note that if we have just got a job from that queue then we can skip this release.
	// If the job we got actually got split, then the release happens from findNextJob().
	if ( (queueIndexOfNewJob != -1) && (queueIndexOfNewJob != queueIndexOfFoundJob) )
	{
		checkQueueAndReleaseOneWaitingThread( queueIndexOfNewJob, data );
	}

	if ( queueIndexOfFoundJob >= 0 )
	{
		unlockQueue( data );
		return GOT_NEXT_JOB;
	}

#if defined HK_PLATFORM_SPU
	if (queueIndexOfFoundJob == -2 )
	{
		unlockQueue( data );
		return JOBS_AVAILABLE_BUT_NOT_FOR_CURRENT_ELF;
	}
#endif

	//
	// Finish the job
	//

	bool masterThreadShouldFinish = false;
	bool allWorkComplete = false;
	{
		// First check if the cpu cache queues are empty
		int numCpuCacheJobsOnQueues = 0;
		for (int i = m_cpuSemaphoreBegin; i < m_directMapSemaphoreEnd; ++i)
		{
			numCpuCacheJobsOnQueues += data->m_jobQueue[i].getSize();
		}

		// If they are, then check the type queues - check one (or two on PlayStation(R)3) queues and threads waiting for each type
		if (numCpuCacheJobsOnQueues == 0)
		{
			// Create a bitfield for all the job types, where a 1 means the job type is finished
			int currentFlags = 0;
			for (int i = 0; i < m_numJobTypes; ++i)
			{
				// Note: This requires SPU to have all queues
				int typeActive = (data->m_numActiveJobs[i] > 0) || !data->m_jobQueue[m_cpuTypesQueuesBegin + i].isEmpty();
	#if defined HK_PLATFORM_HAS_SPU
				typeActive |= !data->m_jobQueue[i].isEmpty();
	#endif
				currentFlags |= typeActive << i;
			}
			masterThreadShouldFinish = (currentFlags & data->m_masterThreadFinishingFlags) == 0;
			allWorkComplete = (currentFlags == 0);
		}
	}

	if (masterThreadShouldFinish)
	{
		#if !defined HK_PLATFORM_SPU
			if ( HK_THREAD_LOCAL_GET(hkThreadNumber) == 0 )
			{
				if (allWorkComplete)
				{
					releaseWaitingThreads(data);
				}
				unlockQueue( data );
				return ALL_JOBS_FINISHED;
			}
			else
		#endif
			if ( data->m_numThreadsWaiting[m_cpuThreadIndexToSemaphoreIndex[0]] > 0 )
			{
				// Release the one master thread.				
				data->m_numThreadsWaiting[m_cpuThreadIndexToSemaphoreIndex[0]]--;
				hkSemaphoreBusyWait::release( m_queueSemaphores[m_cpuThreadIndexToSemaphoreIndex[0]] );
			}
	}

	if (allWorkComplete)
	{
		//printf("All work complete %d\n", HK_THREAD_LOCAL_GET(hkThreadNumber) );
		if ( data->m_waitPolicy != WAIT_INDEFINITELY )
		{
			releaseWaitingThreads(data);
			unlockQueue( data );
			return ALL_JOBS_FINISHED;
		}
	}

	if ( waitStatus == DO_NOT_WAIT_FOR_NEXT_JOB )
	{
		unlockQueue( data );
		return NO_JOBS_AVAILABLE;
	}

#ifdef HK_PLATFORM_PS3_SPU
	JobTypeSpuStats& stats = data->m_spuStats[HK_THREAD_LOCAL_GET(hkThreadNumber)];
	if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::RULE_BASED_QUIT_OR_TIMEOUT )
	{
		const int activeTasks = stats.m_runningTasks - stats.m_waitingTasks;
		if ( activeTasks == 0 && stats.m_runningTasks > 1 ) 
		{
			return SPU_QUIT_ELF;
		}
		return JOB_INVALID;
		// we leave the queue unlocked as it will be locked right before sleeping/quitting
	}
#endif

	//
	//	Wait for a semaphore
	//
	{
		int mySemaphoreIndex = getSemaphoreIndex(HK_THREAD_LOCAL_GET(hkThreadNumber));
		data->m_numThreadsWaiting[mySemaphoreIndex]++;
		unlockQueue( data );
		HK_TIMER_BEGIN("NoJobAvailable",HK_NULL);
		hkSemaphoreBusyWait::acquire( m_queueSemaphores[mySemaphoreIndex] );
		HK_TIMER_END();
	}

	return JOB_INVALID;
}


void hkJobQueue::finishJob( const JobQueueEntry* oldJob, FinishJobFlag flag )
{
#if !defined HK_PLATFORM_SPU
	HK_ASSERT2(0x67556565, HK_THREAD_LOCAL_GET(hkThreadNumber) < m_hwSetup.m_numCpuThreads, "More thread using job queue than Job queue was initialized to handle");
#endif


#if defined(HK_PLATFORM_SPU)
	hkSpuMonitorCache::dmaMonitorDataToMainMemorySpu(); // empty inline function if monitors are disabled
#endif

//#if !defined(HK_PLATFORM_SPU)
	HK_TIME_CODE_BLOCK("finishJob", HK_NULL);
//#endif
	HK_ASSERT(0x975efae9, oldJob != HK_NULL );

	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );

	hkBool jobCreated = false;
	JobQueueEntryInput createdJob;
		
	QueueIndex queueIndexOfNewJob = -1;

	// If we have an old job, we need to check whether this old job just triggers a new job
	if ( m_finishJobFunc( *this, data, *oldJob, createdJob ) == JOB_CREATED )
	{
		queueIndexOfNewJob = getQueueIndexForJob( (hkJob&)createdJob.m_job );

		// Add the job to the queue
		if ( createdJob.m_jobPriority == JOB_HIGH_PRIORITY )
		{
			data->m_jobQueue[queueIndexOfNewJob].enqueueInFront( (const JobQueueEntry&)createdJob.m_job );
		}
		else
		{
			data->m_jobQueue[queueIndexOfNewJob].enqueue( (const JobQueueEntry&)createdJob.m_job );
		}
		checkQueueAndReleaseOneWaitingThread( queueIndexOfNewJob, data );
	}


	if (flag == FINISH_FLAG_NORMAL)
	{
		data->m_numActiveJobs[oldJob->m_jobType]--;
	}
	unlockQueue( data );

}

#if defined(HK_PLATFORM_SPU)

void* hkJobQueue::getNextElfToLoad()
{
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );

	hkInt8* queueIndices = m_nextQueueToGet[getSemaphoreIndex(HK_THREAD_LOCAL_GET(hkThreadNumber))];
	while ( *queueIndices >=0 )
	{
		if ( !data->m_jobQueue[ *queueIndices ].isEmpty() )
		{
			HK_ASSERT2(0x981735f4, m_spuElfs[*queueIndices] != (void*)-1, "Job found for unregistered elf" );
			//HK_SPU_DEBUG_PRINTF(("Thread %d switching to ELF %d, job for it\n",(HK_THREAD_LOCAL_GET(hkThreadNumber)),*queueIndices));
			unlockQueue( data );
			return m_spuElfs[*queueIndices];
		}
		queueIndices++;
	} 

	unlockQueue( data );

	// COM-925 - We couldn't find a job to hint us which ELF to pick. Pick the first available one.
	queueIndices = m_nextQueueToGet[getSemaphoreIndex(HK_THREAD_LOCAL_GET(hkThreadNumber))];
	while ( *queueIndices >=0 )
	{
		const hkInt8 elfId = *queueIndices;
		if (m_spuElfs[elfId] != (void*)-1 )
		{
			//HK_SPU_DEBUG_PRINTF(("Arbitrarily switching from ELF %d to ELF %d. SPU ELF address is %u\n", (HK_THREAD_LOCAL_GET(hkThreadNumber)), elfId, m_spuElfs[elfId]));
			return m_spuElfs[elfId];
		}
		queueIndices++;
	}

	HK_CRITICAL_ASSERT2(0x8731a73c, 0, "Can't find any ELF registered.");

	return HK_NULL;
}
#endif

hkJobQueue::JobStatus hkJobQueue::finishJobAndGetNextJob( const JobQueueEntry* oldJob, JobQueueEntry& jobOut, WaitStatus waitStatus )
{
#if !defined HK_PLATFORM_SPU
	HK_ASSERT2(0x67556565, HK_THREAD_LOCAL_GET(hkThreadNumber) < m_hwSetup.m_numCpuThreads, "More thread using job queue than Job queue was initialized to handle");
#endif

#if defined(HK_PLATFORM_SPU)
	hkSpuMonitorCache::dmaMonitorDataToMainMemorySpu(); // empty inline function if monitors are disabled
#endif

	//#if !defined(HK_PLATFORM_SPU)
	HK_TIME_CODE_BLOCK("GetNextJob", HK_NULL);
	//#endif

#ifdef HK_PLATFORM_PS3_SPU
	int counter = 0;
#endif
	while(1)
	{
		HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
		DynamicData* data = lockQueue( dynamicDataStorage );

		QueueIndex queueIndexOfNewJob = -1;
		JobQueueEntryInput createdJob; 
		// If we have an old job, we need to check whether this old job just triggers a new job		

		if (oldJob)
		{			
			if (m_finishJobFunc( *this, data, *oldJob, createdJob ) == JOB_CREATED )
			{
				// Add the job to the queue
				queueIndexOfNewJob = getQueueIndexForJob( (hkJob&)createdJob.m_job );
				Queue& queue = data->m_jobQueue[queueIndexOfNewJob];

				if ( createdJob.m_jobPriority == JOB_HIGH_PRIORITY )
				{
					queue.enqueueInFront( (const JobQueueEntry&)createdJob.m_job );
				}
				else
				{
					queue.enqueue( (const JobQueueEntry&)createdJob.m_job );
				}
			}

			data->m_numActiveJobs[oldJob->m_jobType]--;
			oldJob = 0;
		}		

		hkJobQueue::JobStatus status = findJobInternal( queueIndexOfNewJob, data, waitStatus, jobOut );
		if (status != JOB_INVALID)
		{			
#ifdef HK_PLATFORM_PS3_SPU
			if ( status == SPU_QUIT_ELF )
			{
				// the queue is still locked; do not quit right after the first iteration
				if ( counter > 0 )
				{
					data->m_spuStats[HK_THREAD_LOCAL_GET(hkThreadNumber)].m_waitingTasks--;
					unlockQueue( data );
					return status;
				}
				// else go on to the timeout part
			}
			else
			{
				if ( counter > 0 )
				{
					
					data = lockQueue( dynamicDataStorage );
					data->m_spuStats[HK_THREAD_LOCAL_GET(hkThreadNumber)].m_waitingTasks--;
					unlockQueue( data );
				}
				return status;
			}
#else
			return status;
#endif
		}

#ifdef HK_PLATFORM_PS3_SPU

		if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF )
		{
			continue;
		}

		counter++;
		if ( counter > 5 )
		{
			data->m_spuStats[HK_THREAD_LOCAL_GET(hkThreadNumber)].m_waitingTasks--;
			unlockQueue( data );
			return status;
		}
		
		if ( counter == 1 )
		{
			data->m_spuStats[HK_THREAD_LOCAL_GET(hkThreadNumber)].m_waitingTasks++;
		}
		unlockQueue( data );

		// sleep a bit
		int sleepTime = spu_read_decrementer();
		while ( sleepTime - spu_read_decrementer() < 2000 ) ;
#endif
	}
}

void hkJobQueue::registerJobHandler(hkJobType jobId, hkJobHandlerFuncs jobHandlerFuncs )
{
	HK_ASSERT2(0xaf3526ea, jobId < int(m_numJobTypes), "You can only register a maximum of m_numJobTypes.");

	m_jobFuncs[jobId] = jobHandlerFuncs;
}

#if !defined(HK_PLATFORM_SPU)
hkJobQueue::JobStatus hkJobQueue::finishAddAndGetNextJob( hkJobType oldJobType, JobPriority priority, JobQueueEntry& jobInOut, WaitStatus waitStatus )
{
#if !defined HK_PLATFORM_SPU
	HK_ASSERT2(0x67556565, HK_THREAD_LOCAL_GET(hkThreadNumber) < m_hwSetup.m_numCpuThreads, "More thread using job queue than Job queue was initialized to handle");
#endif

	HK_TIME_CODE_BLOCK("GetNextJob", HK_NULL);

	bool firstTime = true;
	while (1)
	{
		HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
		DynamicData* data = lockQueue( dynamicDataStorage );
		QueueIndex queueIndexOfNewJob = -1;
		if (firstTime)
		{
			// Finish the old job
			data->m_numActiveJobs[oldJobType]--;

			queueIndexOfNewJob = getQueueIndexForJob( (hkJob&)jobInOut );
			Queue& queue = data->m_jobQueue[queueIndexOfNewJob];

			// Add the job to the queue
			if ( priority == JOB_HIGH_PRIORITY )
			{
				queue.enqueueInFront( jobInOut );
			}
			else
			{ 
				queue.enqueue( jobInOut );
			}
			firstTime = false;
		}
		hkJobQueue::JobStatus status = findJobInternal(queueIndexOfNewJob, data, waitStatus, jobInOut );
		if (status != JOB_INVALID )
		{
			return status;
		}

	}
}
#endif

void hkJobQueue::releaseWaitingThreads(DynamicData* data)
{
#if defined(HK_DEBUG)
	//{for (int i=0; i< m_numJobQueues; i++){ 	HK_ASSERT2( 0xf032de21, data->m_jobQueue[i].isEmpty(), "Queues not empty" ); }}
#endif
	for (int i = 0; i < m_numQueueSemaphores; i++ )
	{
		int numThreadsWaiting = data->m_numThreadsWaiting[i];
		data->m_numThreadsWaiting[i] = 0;

		for ( ;numThreadsWaiting > 0; numThreadsWaiting--)
		{
			hkSemaphoreBusyWait* semaphore = m_queueSemaphores[i];
			hkSemaphoreBusyWait::release(semaphore);
		}
	}
}

void hkJobQueue::setNumRunningSpus( int numSpus )
{
#ifdef HK_PLATFORM_HAS_SPU
	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );

	data->m_numRunningSpus = numSpus;

	unlockQueue( data );
#endif
}

hkJobQueueHwSetup::SpuSchedulePolicy hkJobQueue::setSpuSchedulingPolicy( hkJobQueueHwSetup::SpuSchedulePolicy policy )
{
	hkJobQueueHwSetup::SpuSchedulePolicy prevPolicy = m_hwSetup.m_spuSchedulePolicy;
#ifdef HK_PLATFORM_HAS_SPU
	// nothing to do if we're not changing the policy or using the old semaphore one
	if ( m_hwSetup.m_spuSchedulePolicy == policy || policy == hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF )
	{
		m_hwSetup.m_spuSchedulePolicy = policy;
		return prevPolicy;
	}

	HK_ALIGN16(char dynamicDataStorage[sizeof(DynamicData)]);
	DynamicData* data = lockQueue( dynamicDataStorage );

	if ( m_hwSetup.m_spuSchedulePolicy == hkJobQueueHwSetup::SEMAPHORE_WAIT_OR_SWITCH_ELF )
	{
		releaseWaitingThreads( data );
	}

	data->m_numRunningSpus = 0;
	for ( int i = 0; i < HK_JOB_TYPE_MAX; ++i )
	{
		hkJobQueue::JobTypeSpuStats& stats = data->m_spuStats[i];
		// clear SPU statistics
		stats.m_runningTasks = 0;
		stats.m_waitingTasks = 0;
	}

	m_hwSetup.m_spuSchedulePolicy = policy;

	unlockQueue( data );
#endif
	return prevPolicy;
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
