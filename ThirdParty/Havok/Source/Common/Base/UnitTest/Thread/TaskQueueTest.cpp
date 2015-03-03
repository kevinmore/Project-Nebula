/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#ifndef HK_PLATFORM_RVL // wii doesn't use the task queue

#include <Common/Base/Thread/Task/hkTaskQueue.h>
#include <Common/Base/Thread/Task/hkTask.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Thread/Pool/hkCpuThreadPool.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>
#include <Common/Base/Monitor/MonitorStreamAnalyzer/hkMonitorStreamAnalyzer.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

// Uncomment this to generate the report file (TaskQueueTest.txt)
//#define	GENERATE_REPORT

extern HK_THREAD_LOCAL( int ) hkThreadNumber;

namespace
{
	enum OwnerWatingMode
	{
		JOIN_PROCESSING,
		POLL_GROUP_STATE
	};

	struct SortTask : public hkTask
	{
		SortTask(int size) : m_rng(0), m_numbers(size) {}

		virtual void process()
		{
			for (int i = 0; i < m_numbers.getSize(); ++i)
			{
				m_numbers[i] = m_rng.getRand32();
			}

			hkSort(m_numbers.begin(), m_numbers.getSize());
		}

		hkPseudoRandomGenerator m_rng;
		hkArray<int> m_numbers;
	};

	struct SleepTask : public hkTask
	{
		SleepTask(hkReal sleepMicroSecs) : m_sleepTime(sleepMicroSecs), m_timesProcessed(0) {}

		virtual void process()
		{
			hkStopwatch stopwatch;
			stopwatch.start();
			while (stopwatch.getElapsedSeconds() * 1e6f < m_sleepTime) {}
			hkCriticalSection::atomicExchangeAdd(&m_timesProcessed, 1);
		}

		hkReal m_sleepTime;
		hkUint32 m_timesProcessed;
	};

	// When processed, activates a number of inactive tasks in a group.
	struct ActivatorTask : public hkTask
	{
		ActivatorTask(
			hkTaskQueue* taskQueue, const hkArray<hkDefaultTaskGraph::TaskId>& idsToActivate,
			hkArray<hkTask*>& tasks)
		: m_taskQueue(taskQueue), m_idsToActivate(idsToActivate), m_tasks(tasks), m_groupId(hkTaskQueue::GraphId::invalid())
		{}

		virtual void process()
		{
			for (int i = 0; i < m_idsToActivate.getSize(); ++i)
			{
				m_taskQueue->activateTask(m_groupId, m_idsToActivate[i], m_tasks[i]);
			}
		}

		hkTaskQueue* m_taskQueue;
		const hkArray<hkDefaultTaskGraph::TaskId>& m_idsToActivate;
		hkArray<hkTask*>& m_tasks;
		hkTaskQueue::GraphId m_groupId;
	};
}


static void threadWorkerFunction(void* workLoad)
{
	hkTaskQueue* taskQueue = (hkTaskQueue*)workLoad;
	hkTaskQueue::PrioritizedTask prioritizedTask;
	hkTaskQueue::GetNextTaskResult result = taskQueue->finishTaskAndGetNext(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_QUEUE_CLOSED);
	while (result != hkTaskQueue::QUEUE_CLOSED)
	{
		taskQueue->getTask(prioritizedTask)->process();
		result = taskQueue->finishTaskAndGetNext(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_QUEUE_CLOSED, &prioritizedTask);
	}
}

static void testTaskMultiplicity(hkTaskQueue& queue)
{
	hkDefaultTaskGraph graphs[2];
	SleepTask* tasks[2] = { new SleepTask(10), new SleepTask(10) };
	graphs[0].addTask(tasks[0], 4);
	graphs[1].addTask(tasks[1], 8);
	hkTaskQueue::GraphId graphIds[2];
	graphIds[0] = queue.addGraph(graphs, 0);
	graphIds[1] = queue.addGraph(graphs + 1, 10);

	// Join processing starting with the group with the lowest priority to test obtaining tasks that are not on the
	// queue top.
	for (int i = 0; i < 2; ++i)
	{
		hkTaskQueue::GraphId graphId = graphIds[1 - i];
		hkTaskQueue::PrioritizedTask prioritizedTask;
		hkTaskQueue::GetNextTaskResult result =
			queue.finishTaskAndGetNextInGraph(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_ALL_TASKS_FINISHED, graphId, HK_NULL);
		while (result != hkTaskQueue::GRAPH_FINISHED)
		{
			queue.getTask(prioritizedTask)->process();
			result = queue.finishTaskAndGetNextInGraph(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_ALL_TASKS_FINISHED, graphId, &prioritizedTask);
		}
		queue.removeGraph(graphId);
	}

	// Check that each task has been processed the right number of times
	HK_TEST(tasks[0]->m_timesProcessed == 4);
	HK_TEST(tasks[1]->m_timesProcessed == 8);

	tasks[0]->removeReference();
	tasks[1]->removeReference();
}


static void testTaskActivation(hkTaskQueue& queue, OwnerWatingMode ownerMode)
{
	hkDefaultTaskGraph group;
	hkTask* sleepTask = new SleepTask(1);

	// Add two inactive tasks to the group
	hkDefaultTaskGraph::TaskId inactiveId1 = group.addTask(HK_NULL);
	hkDefaultTaskGraph::TaskId inactiveId2 = group.addTask(HK_NULL);

	// Set up the first of them to be activated by an activator task
	hkArray<hkDefaultTaskGraph::TaskId> idsToActivate;
	idsToActivate.pushBack(inactiveId1);
	hkArray<hkTask*> inactiveTasks;
	inactiveTasks.pushBack(sleepTask);
	ActivatorTask* activatorTask = new ActivatorTask(&queue, idsToActivate, inactiveTasks);
	hkDefaultTaskGraph::TaskId activatorId = group.addTask(activatorTask);

	// Add two other tasks to make things more exciting
	hkDefaultTaskGraph::TaskId activeId = group.addTask(sleepTask);
	hkDefaultTaskGraph::TaskId endId = group.addTask(sleepTask);

	// Set up dependencies
	group.addDependency(activatorId, activeId);
	group.addDependency(activeId, endId);
	group.addDependency(activatorId, inactiveId1);
	group.addDependency(inactiveId1, endId);
	group.addDependency(activatorId, inactiveId2);
	group.addDependency(inactiveId2, endId);
	const int maxNumAvailableTasks = 5;
	group.finish(maxNumAvailableTasks);

	// Add group
	const int priority = 0;
	hkTaskQueue::GraphId groupId = queue.addGraph(&group, priority);
	activatorTask->m_groupId = groupId;

	// Wait until group is finished
	if (ownerMode == JOIN_PROCESSING)
	{
		hkTaskQueue::PrioritizedTask prioritizedTask;
		hkTaskQueue::GetNextTaskResult result = queue.finishTaskAndGetNextInGraph(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_ALL_TASKS_FINISHED, groupId, HK_NULL);
		while (result != hkTaskQueue::GRAPH_FINISHED)
		{
			queue.getTask(prioritizedTask)->process();
			result = queue.finishTaskAndGetNextInGraph(&prioritizedTask, hkTaskQueue::WAIT_UNTIL_ALL_TASKS_FINISHED, groupId, &prioritizedTask);
		}
	}
	else
	{
		while (!queue.isGraphFinished(groupId))
		{
			sleepTask->process();
		}
	}
	queue.removeGraph(groupId);

	activatorTask->removeReference();
	sleepTask->removeReference();
}


static void printNode(hkMonitorStreamAnalyzer::Node* node, int threadId, double startTime, double* currentTime, int depth,
	hkOstream& outStream)
{
	if (node->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER)
	{
		double start = node->m_absoluteStartTime - startTime;
		double length = node->m_value[0];
		*currentTime = node->m_absoluteStartTime;
		outStream.printf("%2d\t%7.2f\t%*s%s\t%5.2f\n", threadId, start, 3 * depth, "", node->m_name, length);
	}
	else if (node->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_SINGLE)
	{
		double start = *currentTime - startTime;
		outStream.printf("%2d\t%7.2f\t%*s%s %d\t%5.2f\n", threadId, start, 3 * depth, "", node->m_name, (int)node->m_value[0], 0);
	}

	depth++;
	for (int i = 0; i < node->m_children.getSize(); ++i)
	{
		printNode(node->m_children[i], threadId, startTime, currentTime, depth, outStream);
	}
}


static void generateReport(hkCpuThreadPool& threadPool)
{
#if defined(GENERATE_REPORT)
	hkArray<hkTimerData>::Temp timers;

	// Get this thread's timer
	{
		hkTimerData& timerData = timers.expandOne();
		timerData.m_streamBegin = hkMonitorStream::getInstance().getStart();
		timerData.m_streamEnd = hkMonitorStream::getInstance().getEnd();
	}

	// Get worker threads' timers
	threadPool.appendTimerData(timers, hkArray<hkTimerData>::Temp::AllocatorType().get(&timers));
	const int numThreads = timers.getSize();

	hkMonitorStreamFrameInfo frameInfo;
	frameInfo.m_indexOfTimer1 = -1;
	frameInfo.m_absoluteTimeCounter = hkMonitorStreamFrameInfo::ABSOLUTE_TIME_TIMER_0;
	frameInfo.m_timerFactor0 = 1e6f / float(hkStopwatch::getTicksPerSecond());

	hkArray<hkMonitorStreamAnalyzer::Node*>::Temp statistics(numThreads);
	for (int i = 0; i < numThreads; ++i)
	{
		statistics[i] = hkMonitorStreamAnalyzer::makeStatisticsTreeForSingleFrame(timers[i].m_streamBegin, timers[i].m_streamEnd, frameInfo, "/", false);
	}

	// Print statistics as a sequence of events
	{
		// Figure out absolute starting time
		double start = DBL_MAX;
		for (int i = 0; i < numThreads; ++i)
		{
			const hkArray<hkMonitorStreamAnalyzer::Node*>& children = statistics[i]->m_children;
			for (int j = 0; j < children.getSize(); ++j)
			{
				if (children[j]->m_type == hkMonitorStreamAnalyzer::Node::NODE_TYPE_TIMER && children[j]->m_absoluteStartTime < start)
				{
					start = children[j]->m_absoluteStartTime;
				}
			}
		}

		// Print each thread separately
		hkOstream report("TaskQueueTest.txt");
		report.printf("THREAD\tSTART\tNAME\tDURATION\n");
		double currentTime = start;
		for (int i = 0; i < numThreads; ++i)
		{
			printNode(statistics[i], i, start, &currentTime, 0, report);
		}
	}

	for (int i = 0; i < statistics.getSize(); ++i)
	{
		delete statistics[i];
	}
#endif
}


int taskQueue_main()
{
	// Create thread pool
	hkCpuThreadPoolCinfo cInfo(&threadWorkerFunction);
	cInfo.m_numThreads = 2;
	cInfo.m_threadName = "TestWorkerThread";
	cInfo.m_timerBufferPerThreadAllocation = 1024 * 1024;
	hkCpuThreadPool threadPool(cInfo);

	// Set thread pool to work on a task queue
	hkTaskQueue taskQueue;
	threadPool.processWorkLoad(&taskQueue);

	testTaskActivation(taskQueue, JOIN_PROCESSING);
	testTaskActivation(taskQueue, POLL_GROUP_STATE);
	testTaskMultiplicity(taskQueue);

	taskQueue.close();
	threadPool.waitForCompletion();

	generateReport(threadPool);

	return 0;
}
#else //HK_PLATFORM_RVL
int taskQueue_main() { return 0; }
#endif

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

//HK_TEST_REGISTER(taskQueue_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

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
