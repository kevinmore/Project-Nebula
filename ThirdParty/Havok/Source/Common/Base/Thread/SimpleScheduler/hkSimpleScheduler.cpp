/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/SimpleScheduler/hkSimpleScheduler.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif


#if !defined(HK_PLATFORM_SPU)


HK_FORCE_INLINE void hkSimpleSchedulerTaskBuilder::TaskHeader::addDependentTask(TaskId id )
{
	HK_ASSERT2(0xf034def0, m_numDependencies < m_capacity, "Adding too many dependencies to this task, increase the capacity when adding the base task" );
	m_dependentTasks[ m_numDependencies ] = id;
	m_numDependencies++;
}


hkSimpleSchedulerTaskBuilder::hkSimpleSchedulerTaskBuilder(int taskCapacity)
{
	m_finalized = false;
	m_lockCount = 0;
	m_numActiveThreads = 0;
	m_numFinishedTasks = 0;
	m_initialReferenceCounts.reserve(taskCapacity );
	m_taskData.reserve (taskCapacity * (sizeof(TaskHeader) + 4) );


	// create dummy task with no dependencies and reference count of 1
	// this task is used to get the first task
	hkUchar taskData[4];
	TaskId firstTask = addTask(taskData, 4, 0 );
	m_initialReferenceCounts[firstTask.value()].m_referenceCount = 1;	// can never be taken
}

hkSimpleSchedulerTaskBuilder::~hkSimpleSchedulerTaskBuilder()
{

}


hkSimpleSchedulerTaskBuilder::TaskId hkSimpleSchedulerTaskBuilder::addTask(const void* taskDataIn, int numBytesTaskData, int dependentTasksCapacity )
{
	HK_ASSERT2(0xf0434fde, (numBytesTaskData&3) == 0, "Your task data size must be a multiple of 4");
	HK_ASSERT2(0xf0434fdf, dependentTasksCapacity < 255, "Your maximum number of dependencies is 255, use proxy tasks to have more dependent tasks");
	HK_ASSERT2(0xf034dfdc, !m_finalized && m_lockCount==0, "Cannot add tasks or dependecies after finalize()" );
	int capacity = dependentTasksCapacity + (1-(dependentTasksCapacity&1));	// make uneven

	int size = sizeof(TaskHeader) - sizeof(TaskId) + capacity * sizeof(TaskId) + numBytesTaskData;

	int taskOffset = m_taskData.getSize();

	// allocate space for task header and task data
	TaskHeader* header = (TaskHeader*)m_taskData.expandBy(size);

	header->m_capacity = hkUchar(capacity);
	header->m_numDependencies = 0;

	// append task data to the task header
	hkUchar* taskData = header->getTaskData();
	hkString::memCpy4(taskData, taskDataIn, numBytesTaskData>>2 );

	TaskId taskId = TaskId(m_initialReferenceCounts.getSize() );
	TaskInfo& ol = m_initialReferenceCounts.expandOne();

	ol.m_referenceCount = 0;
	ol.m_taskOffset = hkUint16(taskOffset);

	return taskId;
}

void hkSimpleSchedulerTaskBuilder::addDependency(TaskId baseTask, TaskId dependentTask )
{
	HK_ASSERT2(0xf034dfde, !m_finalized && m_lockCount==0, "Cannot add tasks or dependecies after finalize()" );

	m_initialReferenceCounts[ dependentTask.value() ].m_referenceCount++;
	getTaskHeader(baseTask).addDependentTask(dependentTask );
}

void hkSimpleSchedulerTaskBuilder::finalize()
{
	HK_ASSERT2(0xf034dfde, !m_finalized && m_lockCount==0, "Already finalized or threads are working on this data" );
	m_finishedTasks.clear();
	int numFinishedTasks = HK_NEXT_MULTIPLE_OF( 16/sizeof(TaskId), m_initialReferenceCounts.getSize()+ MAX_NUM_THREADS );
	m_finishedTasks.setSize(numFinishedTasks, TaskId::invalid() );
	m_finalized = true;
}

void hkSimpleSchedulerTaskBuilder::resetRuntimeData()
{
	HK_ASSERT2(0xf034dfde, !m_finalized && m_lockCount==0, "Already finalized" );
	m_numFinishedTasks = 0;
}
#endif

void hkSimpleScheduler::resetTasksRuntimeData(hkSimpleSchedulerTaskBuilder* schedulerPpu)
{

#if !defined(HK_PLATFORM_SPU)
	HK_ASSERT2(0xf03dd445, schedulerPpu->m_numActiveThreads == 0, "There are still active threads");

	// clear the finished list
	static HK_ALIGN16( int minusOne[4] ) = { -1, -1,-1,-1 };
	hkString::memSet16( schedulerPpu->m_finishedTasks.begin(), minusOne, m_finishTasksByteSize>>4 );
	schedulerPpu->m_numFinishedTasks = 0;
#else
	hkSpuDmaManager::putToMainMemory( m_finishedTasksPpu, m_finishedTasks, m_finishTasksByteSize, hkSpuDmaManager::WRITE_NEW );	// no wait here, will be done in setInt32InMainMemory()
	hkSpuDmaUtils::setInt32InMainMemory( (hkInt32*)&schedulerPpu->m_numFinishedTasks, 0 );
	hkSpuDmaManager::performFinalChecks( m_finishedTasksPpu, m_finishedTasks, m_finishTasksByteSize );
#endif
}

void hkSimpleScheduler::resetScheduler()
{
	m_numActiveTasks = 0;
	m_numProcessedFinishedTask = 0;
	m_numActiveThreadsDetected = 0;
	m_numTasksRemovedFromTheActiveList = 0;

	//
	//	Make a copy of the shared initial reference counts to my local thread
	//
#if !defined(HK_PLATFORM_SPU)
	m_referenceCounts = m_schedulerPpu->m_initialReferenceCounts;
#else
	int transferSizeReferenceCounts = HK_NEXT_MULTIPLE_OF( 16, sizeof(hkSimpleSchedulerTaskBuilder::TaskInfo) * m_numTotalTasks );
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( m_referenceCounts, m_referenceCountsPpu, transferSizeReferenceCounts, hkSpuDmaManager::READ_COPY );
	hkSpuDmaManager::performFinalChecks( m_referenceCountsPpu, m_referenceCounts, transferSizeReferenceCounts);

	// clear the local copy of the finish list 
	// (not necessary on ppu as this list is shared)
	static HK_ALIGN16( int minusOne[4] ) = { -1, -1,-1,-1 };
	hkString::memSet16( m_finishedTasks, minusOne, m_finishTasksByteSize>>4 );
#endif

	// collect all tasks with reference count 0 
	{
		for (int i=m_numTotalTasks-1; i >0; i-- )	// Note skip 0
		{
			if (m_referenceCounts[i].m_referenceCount == 0 )
			{
				m_activeTasks[ m_numActiveTasks ] = TaskId(i);
				m_numActiveTasks = m_numActiveTasks+1;
			}
		}
	}
	HK_ASSERT2( 0xf0344556, m_numActiveTasks < MAX_ACTIVE_TASKS, "You cannot have a tasklist which generates too many active tasks");
}
void hkSimpleScheduler::initScheduler(const hkSimpleSchedulerTaskBuilder* tasksPpu )
{
	m_schedulerPpu = tasksPpu;
	m_numFinishedTasks  = &tasksPpu->m_numFinishedTasks;

#if !defined(HK_PLATFORM_SPU)
	m_numTotalTasks		= tasksPpu->m_initialReferenceCounts.getSize();
	m_finishTasksByteSize = tasksPpu->m_finishedTasks.getSize() * sizeof(TaskId::Type);

	// check Spu Ppu consistency
	HK_ASSERT( 0xf0344565, hkUint32(m_finishTasksByteSize) == HK_NEXT_MULTIPLE_OF( 16, sizeof(TaskId::Type) * HK_HINT_SIZE16(m_numTotalTasks+hkSimpleSchedulerTaskBuilder::MAX_NUM_THREADS) ) );
	m_taskData = tasksPpu->m_taskData.begin();
	m_finishedTasks = (TaskId::Type*)tasksPpu->m_finishedTasks.begin();
	HK_ASSERT(0xf03dfdf6, tasksPpu->m_finalized == true );
#else
	{
		// get the task builder onto the SPU
		const int taskBuilderSize  = HK_NEXT_MULTIPLE_OF(16, sizeof(hkSimpleSchedulerTaskBuilder) );
		HK_ALIGN16(char builderBuffer[taskBuilderSize]);
		const hkSimpleSchedulerTaskBuilder* taskBuilder = (const hkSimpleSchedulerTaskBuilder*)&builderBuffer;
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( builderBuffer, tasksPpu, taskBuilderSize, hkSpuDmaManager::READ_COPY );

		m_numTotalTasks		= taskBuilder->m_initialReferenceCounts.getSize();

		// get the task data
		{
			int taskTransferSize = HK_NEXT_MULTIPLE_OF(16, taskBuilder->m_taskData.getSize() );
			int taskDataSize = HK_NEXT_MULTIPLE_OF(128, taskTransferSize );
			m_taskData = (hkUchar*)hkSpuStack::getInstance().allocateStack(taskDataSize, "hkSimpleScheduler::TaskData");
			m_taskDataSize = taskDataSize;
			hkSpuDmaManager::getFromMainMemory( (void*)m_taskData.val(), taskBuilder->m_taskData.begin(), taskTransferSize, hkSpuDmaManager::READ_COPY );
			hkSpuDmaManager::deferFinalChecksUntilWait( taskBuilder->m_taskData.begin(), m_taskData.val(), taskTransferSize);
			// no dma wait here, will be done in resetScheduler
		}

		// allocate the reference count list (will be transfered to spu in resetScheduler
		{
			int refCountSize = HK_NEXT_MULTIPLE_OF( 128, sizeof(hkSimpleSchedulerTaskBuilder::TaskInfo) * m_numTotalTasks );
			m_referenceCounts = (hkSimpleSchedulerTaskBuilder::TaskInfo*)hkSpuStack::getInstance().allocateStack(refCountSize, "hkSimpleScheduler::ReferenceCounts");
			m_referenceCountsPpu = taskBuilder->m_initialReferenceCounts.begin();
		}

		// allocate the finished list (will be set to invalid in resetScheduler)
		{
			m_finishTasksByteSize = HK_NEXT_MULTIPLE_OF( 16, sizeof(TaskId::Type) * HK_HINT_SIZE16(m_numTotalTasks+hkSimpleSchedulerTaskBuilder::MAX_NUM_THREADS) );
			int finishSize = HK_NEXT_MULTIPLE_OF( 128, m_finishTasksByteSize );
			m_finishedTasks = (TaskId::Type*)hkSpuStack::getInstance().allocateStack(finishSize, "hkSimpleScheduler::FinishCache");
			m_finishedTasksPpu = (TaskId::Type*)taskBuilder->m_finishedTasks.begin();
		}
		HK_ASSERT(0xf03dfdf6, taskBuilder->m_finalized == true );
		hkSpuDmaManager::performFinalChecks( tasksPpu, builderBuffer, taskBuilderSize);
	}
#endif
	HK_ON_DEBUG(hkDmaManager::atomicExchangeAdd(&tasksPpu->m_lockCount, 1 ));
	resetScheduler();
}

void hkSimpleScheduler::exitScheduler(const hkSimpleSchedulerTaskBuilder* schedulerPpu)
{
	HK_ON_DEBUG( hkDmaManager::atomicExchangeAdd(&schedulerPpu->m_lockCount, -1 ) );

#if !defined(HK_PLATFORM_SPU)
	HK_ASSERT2(0xf03dd445, schedulerPpu->m_numActiveThreads == 0, "There are still active threads");
#else
	int finishSize = HK_NEXT_MULTIPLE_OF( 128, m_finishTasksByteSize );
	hkSpuStack::getInstance().deallocateStack( (void*)m_finishedTasks.val(), finishSize);

	int refCountSize = HK_NEXT_MULTIPLE_OF( 128, sizeof(hkSimpleSchedulerTaskBuilder::TaskInfo) * m_numTotalTasks );
	hkSpuStack::getInstance().deallocateStack( (void*)m_referenceCounts.val(), refCountSize);

	hkSpuStack::getInstance().deallocateStack( (void*)m_taskData.val(), m_taskDataSize);
#endif

}


static void insertTaskIntoArray(hkSimpleScheduler::TaskId* array, int sizeNotIncludingNewElement, hkSimpleScheduler::TaskId toInsert )
{
	int k;
	for (k= sizeNotIncludingNewElement; k>0; k-- )
	{
		hkSimpleScheduler::TaskId at = array[k-1];
		if (toInsert <= at )
		{
			array[k] = toInsert;
			break;
		}
		array[k] = at;
	}
	if (k == 0 )
	{
		array[0] = toInsert;
	}
#if defined(HK_DEBUG)
	{
		for (int i=0; i < sizeNotIncludingNewElement-1;i++)
		{
			HK_ASSERT(0xf04fdf5, array[i] > array[i+1]);
		}
	}
#endif
}

extern HK_THREAD_LOCAL(int ) hkThreadNumber;

#undef HK_REPORT
#define HK_REPORT(x)
#define HK_ON_REPORT(x)
//#define HK_ON_REPORT(x) x

hkSimpleScheduler::TaskId hkSimpleScheduler::finishTaskAndGetNextTask(TaskId previousTask, void** taskDataOut )
{
	HK_ON_DEBUG(if (previousTask == TaskId(0)) { hkDmaManager::atomicExchangeAdd(&m_schedulerPpu->m_numActiveThreads, +1); } );

	hkMonitorStream* timerStream = HK_NULL;
	HK_ON_REPORT(int threadIdx = HK_THREAD_LOCAL_GET(hkThreadNumber));
	TaskId myTaskId = TaskId::invalid();

	//
	//	finish my task,
	//
	int finishedTaskIndex = hkDmaManager::atomicExchangeAdd((hkUint32*)m_numFinishedTasks, 1 );
	HK_ASSERT( 0xf0344565, finishedTaskIndex < m_numTotalTasks + hkSimpleSchedulerTaskBuilder::MAX_NUM_THREADS);

	m_finishedTasks[finishedTaskIndex] = previousTask.value();

#if defined(HK_PLATFORM_SPU)
	// update master copy on the ppu as well
	hkSpuDmaUtils::setUint16InMainMemory( &m_finishedTasksPpu[finishedTaskIndex], previousTask.value() );
#endif

	int numTotalFinishedTasks = finishedTaskIndex+1;	// total tasks to finish
	int maxNumTasksToFinish = m_numTotalTasks + m_numActiveThreadsDetected - 1; // numTotalTasks plus other threads
	goto PROCESS_OLD_TASKS;
	//
	//	Iterate through the finished tasks and simulate what this and other tasks would do
	//
	{
OUTER_LOOP_AGAIN:
		while(m_numProcessedFinishedTask < numTotalFinishedTasks)
		{
			{
				TaskId::Type* pid = &m_finishedTasks[m_numProcessedFinishedTask];
				// we need to get the finished task id. Because write memory access might be reordered we are iterating until the changes have arrived.
				// So the waiting time should be a few cycles max.
#if !defined(HK_PLATFORM_SPU)
				TaskId::Type id   = hkDmaManager::waitForValueNotEqual(pid, TaskId::InvalidValue);
#else
				// get the finished data, if not there get it from ppu
				TaskId::Type id   = *pid;
				while ( id == TaskId::InvalidValue)
				{
					hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( m_finishedTasks, m_finishedTasksPpu, m_finishTasksByteSize, hkSpuDmaManager::READ_COPY_NON_DETERMINISTIC );
					hkSpuDmaManager::performFinalChecks( m_finishedTasksPpu, m_finishedTasks, m_finishTasksByteSize);
					id = *pid;
				}
#endif
				HK_ON_REPORT ({
					hkStringBuf msg;	char buffer[256];
					for (int i=0; i < m_numActiveTasks; i++)
					{
						hkString::sprintf(buffer, " T%i", m_activeTasks[i].value() );		msg.append(buffer );
					}
					HK_REPORT("Finish:" << threadIdx << "   T" << id << "[" << m_numProcessedFinishedTask << "](" <<
						m_numTasksRemovedFromTheActiveList<< "->" << finishedTaskIndex << ")\t\t" << msg.cString() );
				})




				//
				//  Process old tasks
				//	Decrement reference count on dependent tasks
				//
				if ( id != 0)
				{
					TaskHeader& header = getTaskHeader(TaskId(id) );
					int numDependencies = header.m_numDependencies;
					for (int k=0; k < numDependencies; k++)
					{
						TaskId depId = header.m_dependentTasks[k];
						hkSimpleSchedulerTaskBuilder::TaskInfo& ol = m_referenceCounts[depId.value()];
						ol.m_referenceCount--;
						if (ol.m_referenceCount == 0 )
						{	// insert this task into the active task array
							HK_REPORT("Activate:" << threadIdx << "   T" << depId.value() );
							insertTaskIntoArray( m_activeTasks, m_numActiveTasks, depId );
							m_numActiveTasks++;
						}
					}
				}
				else
				{
					maxNumTasksToFinish++;			// one more task to finish
					m_numActiveThreadsDetected = m_numActiveThreadsDetected + 1; // one more task detected
				}
				m_numProcessedFinishedTask++;	// we consumed the 0
			}

			//
			// Consume tasks for outstanding threads (including my thread)
			//
PROCESS_OLD_TASKS:
			HK_ON_REPORT( hkStringBuf msg;	char buffer[256]);
			while ( m_numTasksRemovedFromTheActiveList < m_numProcessedFinishedTask && m_numActiveTasks != 0 )
			{
				HK_ON_REPORT(hkString::sprintf(buffer, " T%i", m_activeTasks[ m_numActiveTasks-1].value() );		msg.append(buffer ));
				m_numActiveTasks--;	// task removed
				if (m_numTasksRemovedFromTheActiveList == finishedTaskIndex)	// now this thread owns the task
				{
					m_numTasksRemovedFromTheActiveList = m_numTasksRemovedFromTheActiveList+1;
					myTaskId = m_activeTasks[m_numActiveTasks];
					*taskDataOut = getTaskHeader(myTaskId).getTaskData();
					HK_REPORT("Got:" << threadIdx << "   Taken T" << myTaskId.value() << "   (" << finishedTaskIndex << ")  \t\tRemoved " << msg.cString() );
					goto FINISHED;	// found my task, don't continue
				}
				m_numTasksRemovedFromTheActiveList = m_numTasksRemovedFromTheActiveList+1;
			}
			HK_REPORT("Remove:" << threadIdx <<  "  \t\tRemoved " << msg.cString() );
		}

		//
		// If there are no more consumable tasks wait until more tasks appear on the queue
		//
		if (numTotalFinishedTasks < maxNumTasksToFinish )
		{
			if (!timerStream )	// insert timer command to track waiting time
			{
				timerStream = & hkMonitorStream::getInstance();
				HK_TIMER_BEGIN2((*timerStream), "WaitForOtherTasks", HK_NULL );
			}
			numTotalFinishedTasks = hkDmaManager::waitForValueNotEqual(m_numFinishedTasks, numTotalFinishedTasks );
			numTotalFinishedTasks = hkMath::min2( numTotalFinishedTasks, maxNumTasksToFinish );
			goto OUTER_LOOP_AGAIN;  // while(1)
		}
		// we cannot increase numTotalFinishedTasks, so we are done
	}

FINISHED:
	if (timerStream ){	HK_TIMER_END2( (*timerStream) );	}
	HK_ON_DEBUG(if (!myTaskId.isValid() ) { hkDmaManager::atomicExchangeAdd(&m_schedulerPpu->m_numActiveThreads, -1); } );
	return myTaskId;

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
