/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Task/hkTaskQueue.h>
#include <Common/Base/Thread/Task/hkTask.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>


// Internal task queue functionality.
class hkTaskQueueEx : public hkTaskQueue
{
	public:

		/// Adds to the queue all unprocessed tasks without unfulfilled dependencies in the given graph.
		void addAvailableTasksInGraph( hkTaskQueue::GraphId graphId, hkTaskQueue::LockMode lockMode );

		/// Implementation of finishTaskAndGetNext(), finishTaskAndGetNextForSpu() and finishTaskAndGetNextInGraph().
		template <int TASK_TYPE>
		HK_FORCE_INLINE GetNextTaskResult finishTaskAndGetNextImpl(
			PrioritizedTask* nextTask, WaitingMode waitingMode, GraphId graphId, const PrioritizedTask* finishedTask);
};

void hkTaskQueueEx::addAvailableTasksInGraph(hkTaskQueue::GraphId graphId, hkTaskQueue::LockMode lockMode)
{
	HK_TQ_TIMER_GET_MONITOR_STREAM();

	GraphInfo& graphInfo = m_graphInfos[graphId];
	hkTaskScheduler& scheduler = graphInfo.m_scheduler;
	int numAvailableTasks = scheduler.getNumAvailableTasks();

	if (numAvailableTasks)
	{
		if (lockMode == LOCK)
		{
			HK_TQ_TIMER_BEGIN_LIST("AddAvailableTasksInGraph", "LockQueue");
			graphInfo.checkThreadOwnership();
			m_queueLock.enter();
			HK_TQ_TIMER_SPLIT_LIST("AddTasks");
		}
		else
		{
			HK_TQ_TIMER_BEGIN_LIST("AddAvailableTasksInGraph", "AddTasks");
		}

		int numAvailableTasksCpu = 0;
		HK_ON_PLATFORM_HAS_SPU(int numAvailableTasksSpu = 0);

		// Add tasks to queue
		PrioritizedTask prioritizedTask;
		prioritizedTask.m_graphId = graphId;
		for (int i = 0; i < numAvailableTasks; ++i)
		{
			hkTaskScheduler::TaskIdAndMultiplicity taskIdAndMultiplicity = scheduler.getNextTask();
			const TaskId taskId(taskIdAndMultiplicity & 0xFFFF);
			HK_ASSERT(0x3f8c4a5a, taskId.isValid());
			prioritizedTask.setPriorityAndTaskId(graphInfo.m_priority, taskId);
			Multiplicity multiplicity = (Multiplicity)(taskIdAndMultiplicity >> 16);
			prioritizedTask.m_reaminingMultiplicity = multiplicity;

#if defined(HK_PLATFORM_HAS_SPU)
			hkTask* task = getTask(prioritizedTask);
			if (task->getElf() != HK_INVALID_ELF)
			{
				numAvailableTasksSpu += multiplicity;
				m_queueSpu.addEntry(prioritizedTask);
				HK_TQ_MONITOR_ADD_VALUE("TASK_ADDED_SPU", graphId.value() * 1000.0f + taskId.value(), HK_MONITOR_TYPE_INT);
			}
			else
#endif
			{
				numAvailableTasksCpu += multiplicity;
				m_queue.addEntry(prioritizedTask);
				HK_TQ_MONITOR_ADD_VALUE("TASK_ADDED", graphId.value() * 1000.0f + taskId.value(), HK_MONITOR_TYPE_INT);
			}
		}

		// Check if the main thread is waiting on the the graph signal
		bool releaseGraphSignal = false;
		if (numAvailableTasksCpu && graphInfo.m_ownerWaitingMode != DONT_WAIT)
		{
			graphInfo.m_ownerWaitingMode = DONT_WAIT;
			releaseGraphSignal = true;
			--numAvailableTasksCpu;
		}

		// Check if we need to wake up any threads waiting on the queue signal
		int numThreadsToWakeUp = 0;
		if (numAvailableTasksCpu && m_numThreadsWaiting)
		{
			numThreadsToWakeUp = hkMath::min2(m_numThreadsWaiting, numAvailableTasksCpu);
			m_numThreadsWaiting -= numThreadsToWakeUp;
		}

		// Check if the SPUs are waiting for tasks
#if defined(HK_PLATFORM_HAS_SPU)
		bool wakeUpSpu = false;
		if (numAvailableTasksSpu && m_isWaitingSpu)
		{
			m_isWaitingSpu = false;
			wakeUpSpu = true;
		}
#endif

		if (lockMode == LOCK)
		{
			m_queueLock.leave();
		}

		// Release graph signal. This is done outside the critical section when called from addGraph() but it is safe
		// because the only thread that can wait on the graph signal is the one that adds the graph.
		if (releaseGraphSignal)
		{
			HK_TQ_TIMER_BEGIN_RELEASE_SEMAPHORE(1);
			m_graphSignals[graphId.value()].release();
			HK_TQ_TIMER_END();
			HK_TQ_MONITOR_ADD_VALUE("GRAPH_SIGNAL_RELEASED", graphId.value(), HK_MONITOR_TYPE_INT);
		}

		// Wake up threads waiting in the queue signal
		if (numThreadsToWakeUp)
		{
			HK_TQ_TIMER_BEGIN_RELEASE_SEMAPHORE(numThreadsToWakeUp);
			m_taskAvailableSignal.release(numThreadsToWakeUp);
			HK_TQ_TIMER_END();
		}

#if defined(HK_PLATFORM_HAS_SPU)
		// Wake up PPU thread that launches SPU tasks
		if (wakeUpSpu)
		{
			HK_TQ_TIMER_BEGIN_RELEASE_SEMAPHORE(1);
			m_taskAvailableSignalSpu.release();
			HK_TQ_TIMER_END();
		}
#endif

		HK_TQ_TIMER_END_LIST();
	}
}

template <int TASK_TYPE>
HK_FORCE_INLINE hkTaskQueue::GetNextTaskResult hkTaskQueueEx::finishTaskAndGetNextImpl(
	PrioritizedTask* nextTask, WaitingMode waitingMode, GraphId groupId, const PrioritizedTask* finishedTask)
{
#if !defined(HK_ENABLE_TASK_QUEUE_TIMERS)
	HK_TIME_CODE_BLOCK("FinishTaskAndGetNext", HK_NULL);
#endif
	HK_TQ_TIMER_GET_MONITOR_STREAM();
	HK_TQ_TIMER_BEGIN_LIST("FinishTaskAndGetNext", "LockQueue");

	int numThreadsToWakeUp = 0;
	bool wakeUpSpu = false;
	if (finishedTask)
	{
		const GraphId finishedTaskGraphId = finishedTask->m_graphId;
		GraphInfo& finishedTaskGraphInfo = accessGraphInfo(finishedTaskGraphId);
		hkTaskScheduler& scheduler = finishedTaskGraphInfo.m_scheduler;
		TaskId finishedTaskId = finishedTask->getTaskId();

		m_queueLock.enter();

		// Finish task in the scheduler
		HK_TQ_TIMER_SPLIT_LIST("FinishTask");
		const hkBool32 isGroupFinished = scheduler.finishTask(finishedTaskId);
		HK_TQ_MONITOR_ADD_VALUE("TASK_FINISHED", finishedTaskGraphId.value() * 1000.0f + finishedTaskId.value(), HK_MONITOR_TYPE_INT);

		if (!isGroupFinished)
		{
			// Add any newly available tasks to queue
			HK_TQ_TIMER_SPLIT_LIST("AddAvailableTasks");
			int numAvailableTasks = scheduler.getNumAvailableTasks();
			if (numAvailableTasks)
			{
				int numAvailableTasksCpu = 0;
				HK_ON_PLATFORM_HAS_SPU(int numAvailableTasksSpu = 0);

				// Add tasks to queue
				PrioritizedTask prioritizedTask;
				prioritizedTask.m_graphId = finishedTaskGraphId;
				for (int i = 0; i < numAvailableTasks; ++i)
				{
					hkTaskScheduler::TaskIdAndMultiplicity taskIdAndMultiplicity = scheduler.getNextTask();
					const TaskId taskId(taskIdAndMultiplicity & 0xFFFF);
					HK_ASSERT(0x3f8c4a5a, taskId.isValid());
					prioritizedTask.setPriorityAndTaskId(finishedTaskGraphInfo.m_priority, taskId);
					Multiplicity multiplicity = (Multiplicity)(taskIdAndMultiplicity >> 16);
					prioritizedTask.m_reaminingMultiplicity = multiplicity;

#if defined(HK_PLATFORM_HAS_SPU)
					hkTask* task = getTask(prioritizedTask);
					if (task->getElf() != HK_INVALID_ELF)
					{
						numAvailableTasksSpu += multiplicity;
						m_queueSpu.addEntry(prioritizedTask);
						HK_TQ_MONITOR_ADD_VALUE("TASK_ADDED_SPU", finishedTaskGraphId.value() * 1000.0f + taskId.value(), HK_MONITOR_TYPE_INT);
					}
					else
#endif
					{
						numAvailableTasksCpu += multiplicity;
						m_queue.addEntry(prioritizedTask);
						HK_TQ_MONITOR_ADD_VALUE("TASK_ADDED", finishedTaskGraphId.value() * 1000.0f + taskId.value(), HK_MONITOR_TYPE_INT);
					}
				}

				// Wake up the group owner if there are at least 2 tasks available or just one but we are sure
				// we are not going to get it. Because the owner thread is waiting we know that there are no tasks
				// in the queue belonging to its group besides the ones we just added.
				if (finishedTaskGraphInfo.m_ownerWaitingMode != DONT_WAIT &&
					(numAvailableTasksCpu > 2 ||
					(numAvailableTasksCpu == 1 &&
					((TASK_TYPE == CPU_TASK && m_queue.getTop().m_graphId != finishedTaskGraphId) ||
					(TASK_TYPE == CPU_TASK_IN_GRAPH && groupId != finishedTaskGraphId) ||
					(TASK_TYPE == SPU_TASK)))))
				{
					// Release group signal. We can't do this outside the critical section because the group could
					// have been removed and the group info reused already.
					finishedTaskGraphInfo.m_ownerWaitingMode = DONT_WAIT;
					m_graphSignals[finishedTaskGraphId.value()].release();
					--numAvailableTasksCpu;
				}

				// Check if we need to wake any waiting threads up.
				if (m_numThreadsWaiting)
				{
					// We know the queue was empty before we added the new tasks because there are threads waiting, so
					// we check if we are going to take one of those new tasks to avoid waking up a thread for nothing.
					const int numTasksTaken = (TASK_TYPE == CPU_TASK || (TASK_TYPE == CPU_TASK_IN_GRAPH && groupId != finishedTaskGraphId)) ? 1 : 0;
					numThreadsToWakeUp = hkMath::min2(m_numThreadsWaiting, numAvailableTasksCpu - numTasksTaken);
					m_numThreadsWaiting -= numThreadsToWakeUp;
				}

				// Check if the SPUs are waiting for tasks
#if defined(HK_PLATFORM_HAS_SPU)
				if (numAvailableTasksSpu && m_isWaitingSpu)
				{
					m_isWaitingSpu = false;
					wakeUpSpu = true;
				}
#endif
			}
		}

		// We just finished the group
		else
		{
			// If this is the group that was passed as a parameter we can return now
			if (TASK_TYPE == CPU_TASK_IN_GRAPH && finishedTaskGraphId == groupId)
			{
				m_queueLock.leave();
				HK_TQ_TIMER_END_LIST();
				return GRAPH_FINISHED;
			}

			// Release group signal if required. We can't do this outside the critical section because the group could
			// have been removed and the group info reused already.
			if (finishedTaskGraphInfo.m_ownerWaitingMode != DONT_WAIT)
			{
				finishedTaskGraphInfo.m_ownerWaitingMode = DONT_WAIT;
				m_graphSignals[finishedTaskGraphId.value()].release();
			}
		}
	}

	// We did not finish a previous task
	else
	{
		m_queueLock.enter();
	}

	// If we are looking for tasks in a given group, get the group info. We can keep this reference even after unlocking
	// the queue because the group info free list array is not allowed to grow.
	hkTaskQueue::GraphInfo* HK_RESTRICT groupInfo = HK_NULL;
	if (TASK_TYPE == CPU_TASK_IN_GRAPH)
	{
		HK_ASSERT(0x114c87d4, groupId.isValid());
		groupInfo = &accessGraphInfo(groupId);
		groupInfo->checkThreadOwnership();
	}

	GetNextTaskResult result = TASK_OBTAINED;
	while (1)
	{
		HK_TQ_TIMER_SPLIT_LIST("GetNextTask");

		// No group specified, take the top task from the queue
		if (TASK_TYPE != CPU_TASK_IN_GRAPH)
		{
			// Return if there are no remaining tasks
			if (waitingMode == WAIT_UNTIL_NO_TASKS_REMAIN && m_numRemainingTasks == 0)
			{
				result = NO_TASK_AVAILABLE;
				break;
			}

			// Try to obtain a CPU task
			if (TASK_TYPE == CPU_TASK && !m_queue.isEmpty())
			{
				PrioritizedTask& topTask = m_queue.getEntry(0);
				*nextTask = topTask;
				if (topTask.m_reaminingMultiplicity == 1)
				{
					m_numRemainingTasks--;
					m_graphInfos[nextTask->m_graphId].m_numRemainingTasks--;
					m_queue.popTop();
				}
				else
				{
					topTask.m_reaminingMultiplicity--;
				}
				HK_TQ_MONITOR_ADD_VALUE("TASK_ACQUIRED", nextTask->m_graphId.value() * 1000.0f + nextTask->getTaskId().value(), HK_MONITOR_TYPE_INT);
				break;
			}

			// Try to obtain a SPU task
			if (TASK_TYPE == SPU_TASK && !m_queueSpu.isEmpty())
			{
				// We always pop the top task regardless of its multiplicity as the PPU thread should take care of
				// launching it as many times as necessary.
				const PrioritizedTask& topTask = m_queueSpu.getTop();
				*nextTask = topTask;
				m_numRemainingTasks--;
				m_graphInfos[topTask.m_graphId].m_numRemainingTasks--;
				m_queueSpu.popTop();
				HK_TQ_MONITOR_ADD_VALUE("TASK_ACQUIRED_SPU", nextTask->m_graphId.value() * 1000.0f + nextTask->getTaskId().value(), HK_MONITOR_TYPE_INT);
				break;
			}
		}

		// Take the first task belonging to the specified group
		if (TASK_TYPE == CPU_TASK_IN_GRAPH)
		{
			// Return if there are no remaining tasks in the group
			if (waitingMode == WAIT_UNTIL_NO_TASKS_REMAIN && groupInfo->m_numRemainingTasks == 0)
			{
				result = NO_TASK_AVAILABLE;
				break;
			}

			// Check if the group is finished
			if (!groupInfo->m_scheduler.getNumUnfinishedTasks())
			{
				result = hkTaskQueue::GRAPH_FINISHED;
				break;
			}

			// Get the first task in the queue from the group. Note that by iterating linearly over the min-heap
			// we may not get the task with the highest priority.
			const hkArray<hkTaskQueue::PrioritizedTask>& tasks = m_queue.getContents();
			const int numTasks = tasks.getSize();
			int entryIndex = 0;
			for (; (entryIndex < numTasks) && (tasks[entryIndex].m_graphId != groupId); ++entryIndex) {}
			if (entryIndex < numTasks)
			{
				PrioritizedTask& taskInGroup = m_queue.getEntry(entryIndex);
				*nextTask = taskInGroup;
				if (taskInGroup.m_reaminingMultiplicity == 1)
				{
					m_queue.removeEntry(entryIndex);
					m_numRemainingTasks--;
					groupInfo->m_numRemainingTasks--;
				}
				else
				{
					taskInGroup.m_reaminingMultiplicity--;
				}
				HK_TQ_MONITOR_ADD_VALUE("TASK_ACQUIRED", nextTask->m_graphId.value() * 1000.0f + nextTask->getTaskId().value(), HK_MONITOR_TYPE_INT);
				break;
			}
		}

		// Queue closed
		if (!m_isQueueOpen)
		{
			result = QUEUE_CLOSED;
			break;
		}

		// Return without task
		if (waitingMode == DONT_WAIT)
		{
			result = NO_TASK_AVAILABLE;
			break;
		}

		// Tell the queue to wake up all threads when there are no tasks remaining
		if (TASK_TYPE != CPU_TASK_IN_GRAPH && waitingMode == WAIT_UNTIL_NO_TASKS_REMAIN)
		{
			m_signalWhenNoTasksRemaining = true;
		}

		// Increment the number of threads waiting for the given type of task and unlock the queue
		if (TASK_TYPE == CPU_TASK)
		{
			m_numThreadsWaiting++;
		}
		if (TASK_TYPE == CPU_TASK_IN_GRAPH)
		{
			groupInfo->m_ownerWaitingMode = waitingMode;
		}
		if (TASK_TYPE == SPU_TASK)
		{
			m_isWaitingSpu = true;
		}
		m_queueLock.leave();

		// Wait for a task of the given type
		HK_TQ_TIMER_SPLIT_LIST("WaitForTasks");
		if (TASK_TYPE == CPU_TASK)
		{
			m_taskAvailableSignal.acquire();
		}
		if (TASK_TYPE == CPU_TASK_IN_GRAPH)
		{
			m_graphSignals[groupId.value()].acquire();
		}
		if (TASK_TYPE == SPU_TASK)
		{
			m_taskAvailableSignalSpu.acquire();
		}

		HK_TQ_TIMER_SPLIT_LIST("LockQueue");
		m_queueLock.enter();
	}

	if (result == TASK_OBTAINED)
	{
		const GraphId nextTaskGraphId = nextTask->m_graphId;
		GraphInfo& nextTaskGraphInfo = m_graphInfos[nextTaskGraphId];

		// Wake up the graph owner if we just obtained the last remaining task in the graph and the owner is waiting
		// for it.
		if (nextTaskGraphInfo.m_numRemainingTasks == 0 &&
			nextTaskGraphInfo.m_ownerWaitingMode == WAIT_UNTIL_NO_TASKS_REMAIN)
		{
			nextTaskGraphInfo.m_ownerWaitingMode = DONT_WAIT;
			m_graphSignals[nextTaskGraphId.value()].release();
		}

		// Wake up all threads waiting for CPU or SPU tasks (not the ones waiting in graph semaphores) if we just
		// obtained the last remaining task and there is someone waiting for it.
		if (m_numRemainingTasks == 0 && m_signalWhenNoTasksRemaining)
		{
			m_signalWhenNoTasksRemaining = false;
			numThreadsToWakeUp += m_numThreadsWaiting;
			m_numThreadsWaiting = 0;
			if (m_isWaitingSpu)
			{
				m_isWaitingSpu = false;
				wakeUpSpu = true;
			}
		}
	}

	m_queueLock.leave();

	// Wake threads up if necessary. By releasing the signal outside the critical section we are assuming the risk
	// that other threads may have already taken available tasks and thus we will be awakening sleeping threads for
	// nothing. In situations of low contingency this is unlikely and when contingency is high we favor spending
	// less time inside the critical section as releasing a signal is a costly operation.
	if (numThreadsToWakeUp)
	{
		HK_TQ_TIMER_BEGIN_RELEASE_SEMAPHORE(numThreadsToWakeUp);
		m_taskAvailableSignal.release(numThreadsToWakeUp);
		HK_TQ_TIMER_END();
	}

	// Wake up the PPU thread that launches SPU tasks
	if (wakeUpSpu)
	{
		HK_TQ_TIMER_BEGIN_RELEASE_SEMAPHORE(1);
		m_taskAvailableSignalSpu.release(1);
		HK_TQ_TIMER_END();
	}

	HK_TQ_TIMER_END_LIST();
	return result;
}




hkTaskQueue::hkTaskQueue(int spinCount)
	: m_queueLock(spinCount), m_numThreadsWaiting(0), m_numRemainingTasks(0), m_isQueueOpen(true),
	  m_signalWhenNoTasksRemaining(false)
{
	m_graphInfos.grow(MAX_GRAPHS);
}


hkTaskQueue::GraphId hkTaskQueue::addGraph(hkDefaultTaskGraph* taskGraph, Priority priority)
{
	HK_TQ_TIMER_GET_MONITOR_STREAM();
	HK_TQ_TIMER_BEGIN("AddGraph");

	// Get a new graph ID
	m_queueLock.enter();
	const GraphId graphId = m_graphInfos.allocate();
	m_graphInfos[graphId].m_numRemainingTasks = taskGraph->getNumTasks() - taskGraph->m_numInactiveTasks;
	m_numRemainingTasks += m_graphInfos[graphId].m_numRemainingTasks;
	m_queueLock.leave();

	if (graphId.value() >= MAX_GRAPHS)
	{
		HK_ERROR(0x4f36a4b8, "You cannot have more than " << MAX_GRAPHS << " task graphs in a task queue at any given time");
		return GraphId::invalid();
	}

	// Make sure the task graph has been pre-processed
	taskGraph->finish( taskGraph->m_maxAvailableTasks );

	// Fill in graph info
	GraphInfo& graphInfo = m_graphInfos[graphId];
	hkTaskScheduler& scheduler = graphInfo.m_scheduler;
	scheduler.initTaskScheduler(taskGraph);
	graphInfo.m_priority = priority;
	HK_ON_DEBUG(graphInfo.m_ownerThreadId = HK_THREAD_LOCAL_GET(hkThreadNumber));

	// Add graph tasks to queue to start processing them
	hkTaskQueueEx* self = (hkTaskQueueEx*)this;
	self->addAvailableTasksInGraph(graphId, LOCK);

	HK_TQ_TIMER_END();
	return graphId;
}


void hkTaskQueue::removeGraph(GraphId graphId)
{
	HK_ON_DEBUG(const GraphInfo& graphInfo = getGraphInfo(graphId));
	HK_ON_DEBUG(graphInfo.checkThreadOwnership());
	HK_ASSERT2(0x1572c8fb, !graphInfo.m_scheduler.getNumUnfinishedTasks(), "Graph is not finished");

	
	
	
	m_queueLock.enter();
	m_graphInfos.release(graphId);
	m_queueLock.leave();
}


void hkTaskQueue::activateTask(GraphId graphId, TaskId taskId, hkTask* task)
{
	m_queueLock.enter();
	m_graphInfos[graphId].m_scheduler.activateTask(taskId, task);
	m_queueLock.leave();
}


void hkTaskQueue::finishTask(const PrioritizedTask& finishedTask)
{
	HK_TQ_TIMER_GET_MONITOR_STREAM();
	HK_TQ_TIMER_BEGIN("FinishTask");

	GraphInfo& graphInfo = m_graphInfos[finishedTask.m_graphId];
	hkTaskScheduler& scheduler = graphInfo.m_scheduler;
	TaskId taskId = finishedTask.getTaskId();

	m_queueLock.enter();

	// Finish task in the scheduler
	const hkBool32 isGraphFinishedFlag = scheduler.finishTask(taskId);
	HK_TQ_MONITOR_ADD_VALUE("TASK_FINISHED", finishedTask.m_graphId.value() * 1000.0f + taskId.value(), HK_MONITOR_TYPE_INT);
	if (isGraphFinishedFlag)
	{
		// Release graph signal if required. We can't do this outside the critical section because the graph could
		// have been removed and the graph info reused already.
		if (graphInfo.m_ownerWaitingMode != DONT_WAIT)
		{
			graphInfo.m_ownerWaitingMode = DONT_WAIT;
			m_graphSignals[finishedTask.m_graphId.value()].release();
		}

		m_queueLock.leave();
		HK_TQ_TIMER_END();
		return;
	}

	// Add any new available tasks to queue
	hkTaskQueueEx* self = (hkTaskQueueEx*)this;
	self->addAvailableTasksInGraph(finishedTask.m_graphId, DONT_LOCK);

	m_queueLock.leave();
	HK_TQ_TIMER_END();
}


void hkTaskQueue::processTasks( WaitingMode waitingMode )
{
	hkTaskQueue::PrioritizedTask prioritizedTask;
	hkTaskQueue::GetNextTaskResult result = finishTaskAndGetNext( &prioritizedTask, waitingMode, HK_NULL );
	while( result == hkTaskQueue::TASK_OBTAINED )
	{
		hkTask* task = getTask( prioritizedTask );
		task->process();
		result = finishTaskAndGetNext( &prioritizedTask, waitingMode, &prioritizedTask );
	}
}

void hkTaskQueue::processGraph( GraphId graphId, WaitingMode waitingMode )
{
	hkTaskQueue::PrioritizedTask prioritizedTask;
	hkTaskQueue::GetNextTaskResult result = finishTaskAndGetNextInGraph( &prioritizedTask, waitingMode, graphId, HK_NULL );
	while( result == hkTaskQueue::TASK_OBTAINED )
	{
		hkTask* task = getTask( prioritizedTask );
		task->process();
		result = finishTaskAndGetNextInGraph( &prioritizedTask, waitingMode, graphId, &prioritizedTask );
	}
}


hkTaskQueue::GetNextTaskResult hkTaskQueue::finishTaskAndGetNext(
	PrioritizedTask* nextTask, WaitingMode waitingMode, const PrioritizedTask* finishedTask)
{
	HK_ASSERT2(0x7e5495b5, waitingMode != WAIT_UNTIL_ALL_TASKS_FINISHED, "Unsupported waiting mode");
	hkTaskQueueEx* self = (hkTaskQueueEx*)this;
	return self->finishTaskAndGetNextImpl<CPU_TASK>(nextTask, waitingMode, GraphId::invalid(), finishedTask);
}

hkTaskQueue::GetNextTaskResult hkTaskQueue::finishTaskAndGetNextInGraph(
	PrioritizedTask* nextTask, WaitingMode waitingMode, GraphId graphId, const PrioritizedTask* finishedTask)
{
	HK_ASSERT2(0x34263cda, waitingMode != WAIT_UNTIL_QUEUE_CLOSED, "Unsupported waiting mode");
	hkTaskQueueEx* self = (hkTaskQueueEx*)this;
	return self->finishTaskAndGetNextImpl<CPU_TASK_IN_GRAPH>(nextTask, waitingMode, graphId, finishedTask);
}

hkTaskQueue::GetNextTaskResult hkTaskQueue::finishTaskAndGetNextForSpu(
	PrioritizedTask* nextTask, WaitingMode waitingMode, const PrioritizedTask* finishedTask)
{
	HK_ASSERT2(0x47c6dd57, waitingMode != WAIT_UNTIL_ALL_TASKS_FINISHED, "Unsupported waiting mode");
	hkTaskQueueEx* self = (hkTaskQueueEx*)this;
	return self->finishTaskAndGetNextImpl<SPU_TASK>(nextTask, waitingMode, GraphId::invalid(), finishedTask);
}


void hkTaskQueue::close()
{
	// Lock the queue to prevent any additional thread from going to sleep
	m_queueLock.enter();

	if (m_isQueueOpen)
	{
		m_isQueueOpen = false;

		// Wake up threads waiting on graph signals
		for (int i = 0; i < MAX_GRAPHS; ++i)
		{
			GraphId graphId = (GraphId)i;
			if (m_graphInfos.isAllocated(graphId) && m_graphInfos[graphId].m_ownerWaitingMode != DONT_WAIT)
			{
				m_graphInfos[graphId].m_ownerWaitingMode = DONT_WAIT;
				m_graphSignals[i].release();
			}
		}

		// Wake up threads waiting on the queue
		if (m_numThreadsWaiting)
		{
			const int numThreadsWaiting = m_numThreadsWaiting;
			m_numThreadsWaiting = 0;
			m_taskAvailableSignal.release(numThreadsWaiting);
		}

		// Wake up PPU thread that launches SPU tasks
		if (m_isWaitingSpu)
		{
			m_isWaitingSpu = false;
			m_taskAvailableSignalSpu.release();
		}
	}

	m_queueLock.leave();
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
