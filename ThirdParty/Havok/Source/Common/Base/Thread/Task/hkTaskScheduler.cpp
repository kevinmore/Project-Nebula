/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Task/hkTaskScheduler.h>

// Make sure that TaskId and Multiplicity fit in an hkUint32
HK_COMPILE_TIME_ASSERT(sizeof(hkDefaultTaskGraph::TaskId) <= 2 && sizeof(hkDefaultTaskGraph::Multiplicity) <= 2);


void hkTaskScheduler::initTaskScheduler(const hkDefaultTaskGraph* taskGraph)
{
	const int numTasks = taskGraph->m_taskInfos.getSize();
	m_taskGraph = taskGraph;
	m_taskStates.setSize(numTasks);
	m_numUnfinishedTasks = numTasks - taskGraph->m_numInactiveTasks;
	m_availableTasks.setSize(taskGraph->m_maxAvailableTasks);
	m_activatedTasks.setSize(taskGraph->m_numInactiveTasks, HK_NULL);
	m_nextTask = 0;
	m_numAvailableTasks = 0;

	// Check the number of parents for each task and add to available tasks those with no parents
	for (int taskId = 0; taskId < numTasks; ++taskId)
	{
		const hkDefaultTaskGraph::TaskInfo& taskInfo = taskGraph->m_taskInfos[taskId];
		TaskState& taskState = m_taskStates[taskId];
		taskState.m_numUnfinishedParents = taskInfo.m_numParents;
		taskState.m_reaminingMultiplicity = taskInfo.m_multiplicity;
		if (taskInfo.m_numParents == 0)
		{
			if (m_numAvailableTasks < m_availableTasks.getSize())
			{
				m_availableTasks[m_numAvailableTasks++] = hkDefaultTaskGraph::TaskId(taskId);
			}
			else
			{
				HK_ERROR(0xd2303d5, "Too many available tasks for the buffer size");
			}
		}
	}
}



hkTaskScheduler::TaskIdAndMultiplicity hkTaskScheduler::getNextTask()
{
	hkDefaultTaskGraph::TaskId nextTask;

	if (m_numAvailableTasks)
	{
		nextTask = m_availableTasks[m_nextTask];
		m_nextTask = (m_nextTask + 1) % m_availableTasks.getSize();
		m_numAvailableTasks--;
	}
	else
	{
		nextTask = hkDefaultTaskGraph::TaskId::invalid();
	}

	return (m_taskStates[nextTask.value()].m_reaminingMultiplicity << 16) | nextTask.value();
}


void hkTaskScheduler::activateTask(hkDefaultTaskGraph::TaskId taskId, hkTask* task)
{
	HK_ASSERT(0x3ea35f12, taskId.value() < m_taskGraph->m_taskInfos.getSize());
	HK_ASSERT(0x35c2460a, !m_taskGraph->isTaskActive(taskId));

	// Add task to activated list
	m_activatedTasks[m_taskGraph->getInactiveTaskIndex(taskId)] = task;
	m_numUnfinishedTasks++;

	// Increase unfinished parent count in child tasks
	const hkDefaultTaskGraph::TaskInfo& taskInfo = m_taskGraph->m_taskInfos[taskId.value()];
	int childIndex = taskInfo.m_firstChildIndex;
	for (int i = 0; i < taskInfo.m_numChildren; ++i)
	{
		const hkDefaultTaskGraph::TaskId childId = m_taskGraph->m_children[childIndex++];
		if (m_taskStates[childId.value()].m_numUnfinishedParents)
		{
			m_taskStates[childId.value()].m_numUnfinishedParents++;
		}
		else
		{
			HK_WARN(0x1460b25d, "A task was activated after a child task had been made available");
		}
	}

	// Make task available if possible
	if (!m_taskStates[taskId.value()].m_numUnfinishedParents)
	{
		if (m_numAvailableTasks < m_availableTasks.getSize())
		{
			m_availableTasks[m_numAvailableTasks++] = taskId;
		}
		else
		{
			HK_ERROR(0x78e5cc61, "Too many available tasks for the buffer size");
		}
	}
}


hkBool32 hkTaskScheduler::finishTask(hkDefaultTaskGraph::TaskId taskId)
{
	const hkDefaultTaskGraph::TaskInfo& taskInfo = m_taskGraph->m_taskInfos[taskId.value()];
	int childIndex = taskInfo.m_firstChildIndex;

	// Reduce the multiplicity left
	if (m_taskStates[taskId.value()].m_reaminingMultiplicity > 1)
	{
		m_taskStates[taskId.value()].m_reaminingMultiplicity--;
		return false;
	}

	// Reduce the count of unfinished tasks in the group and see if this was the last one
	m_numUnfinishedTasks--;
	if (!m_numUnfinishedTasks)
	{
		return true;
	}

	// Reduce the number of unfinished parents in all child tasks
	for (int i = 0; i < taskInfo.m_numChildren; ++i, ++childIndex)
	{
		const hkDefaultTaskGraph::TaskId childId = m_taskGraph->m_children[childIndex];
		TaskState& childState = m_taskStates[childId.value()];
		childState.m_numUnfinishedParents--;

		// Make task available when there are no unfinished parents left and the task is active
		if (!childState.m_numUnfinishedParents &&
			(m_taskGraph->isTaskActive(childId) || m_activatedTasks[m_taskGraph->getInactiveTaskIndex(childId)]))
		{
			if (m_numAvailableTasks < m_availableTasks.getSize())
			{
				int taskIndex = (m_nextTask + m_numAvailableTasks) % m_availableTasks.getSize();
				m_availableTasks[taskIndex] = childId;
				m_numAvailableTasks++;
			}
			else
			{
				HK_ERROR(0x304ea150, "Too many available tasks for the buffer size");
			}
		}
	}

	return false;
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
