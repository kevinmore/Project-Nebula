/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_TASK_SCHEDULER_H
#define HK_TASK_SCHEDULER_H

#include <Common/Base/Thread/Task/hkTaskGraph.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

class hkTask;

/// WIP. This class receives a task graph and handles their execution order keeping track of the fulfilled
/// dependencies.
class hkTaskScheduler
{
	public:

		/// Task ID and multiplicity encoded in a single 32-bit value in the following way:
		/// (multiplicity << 16) | taskId
		typedef hkUint32 TaskIdAndMultiplicity;

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE_CLASS, hkTaskScheduler);

		void initTaskScheduler(const hkDefaultTaskGraph* taskGraph);

		/// Returns the next available task
		TaskIdAndMultiplicity getNextTask();

		/// Used to activate an optional task
		void activateTask(hkDefaultTaskGraph::TaskId taskId, hkTask* task);

		/// Translates a task ID into a task
		HK_FORCE_INLINE hkTask* getTask(hkDefaultTaskGraph::TaskId taskId) const
		{
			if (m_taskGraph->isTaskActive(taskId))
			{
				return m_taskGraph->getTask(taskId);
			}
			else
			{
				return m_activatedTasks[m_taskGraph->getInactiveTaskIndex(taskId)];
			}
		}

		HK_FORCE_INLINE int getNumAvailableTasks() const { return m_numAvailableTasks; }

		HK_FORCE_INLINE hkUint32 getNumUnfinishedTasks() const { return m_numUnfinishedTasks; }

		hkBool32 finishTask(hkDefaultTaskGraph::TaskId taskId);

	protected:

		struct TaskState
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS, hkTaskScheduler::TaskState );

			hkUint32 m_numUnfinishedParents;
			hkDefaultTaskGraph::Multiplicity m_reaminingMultiplicity;
		};

	protected:

		const hkDefaultTaskGraph* m_taskGraph;
		hkArray<TaskState> m_taskStates;
		hkUint32 m_numUnfinishedTasks;
		hkArray<hkDefaultTaskGraph::TaskId> m_availableTasks;
		hkArray<hkTask*> m_activatedTasks;
		int m_nextTask;
		int m_numAvailableTasks;
};

#endif // HK_TASK_SCHEDULER_H

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
