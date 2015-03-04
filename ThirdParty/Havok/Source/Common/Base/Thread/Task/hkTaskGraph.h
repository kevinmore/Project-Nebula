/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_TASK_GRAPH_H
#define HK_TASK_GRAPH_H

#include <Common/Base/Types/hkHandle.h>

class hkTask;


/// A interface for adding a set of tasks and optional dependencies between them.
/// When processing the graph, available tasks are expected to be processed in FIFO order.
struct hkTaskGraph
{
	/// A dependency between two tasks.
	struct Dependency
	{
		int m_parentId;
		int m_childId;
	};

	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS, hkTaskGraph );

	/// Virtual empty destructor. Non-virtual on SPU to avoid code bloat.
	HK_FORCE_INLINE HK_NOSPU_VIRTUAL ~hkTaskGraph() {}

	/// Add a set of tasks.
	/// A task is allowed to be added multiple times.
	/// The multiplicity of a task is the number of times that it has to be processed before it is considered finished.
	/// If the array of multiplicities is NULL all tasks will have multiplicity 1.
	/// If taskIdsOut is not NULL, it will be filled with IDs for the added tasks.
	virtual void addTasks(
		hkTask** tasks, int numTasks, const int* multiplicities = HK_NULL, int* taskIdsOut = HK_NULL ) = 0;

	/// Add a set of task dependencies.
	virtual void addDependencies( Dependency* dependencies, int numDependencies ) = 0;

	/// Get the number of tasks that have been added.
	virtual int getNumTasks() = 0;
};


/// A default task graph implementation, used by hkTaskQueue.
struct hkDefaultTaskGraph : public hkTaskGraph
{
	public:

		// Type used to handle pointers as unsigned integers
	#if HK_POINTER_SIZE == 4
		typedef hkUint32 UintPtr;
	#elif HK_POINTER_SIZE == 8
		typedef hkUint64 UintPtr;
	#endif

		HK_DECLARE_HANDLE(TaskId, hkUint16, 0xFFFF);
		typedef hkUint16 DependencyCount;

		/// Type used to store the multiplicity of a task
		typedef hkUint8 Multiplicity;

		enum
		{
			MAX_TASKS = 1 << 16,
			MAX_DEPENDENCIES = 0xFFFF
		};

		enum ExecutionPolicy
		{
			EXECUTION_POLICY_NONE,
			EXECUTION_POLICY_BREADTH_FIRST,
			EXECUTION_POLICY_DEPTH_FIRST,
		};

		struct TaskInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS, hkDefaultTaskGraph::TaskInfo );

			hkTask* m_task;
			int m_firstChildIndex;
			DependencyCount m_numParents;
			DependencyCount m_numChildren;
			Multiplicity m_multiplicity;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS, hkDefaultTaskGraph );

		hkDefaultTaskGraph() : m_maxAvailableTasks(0), m_numInactiveTasks(0) {}

		HK_FORCE_INLINE hkTask* getTask(TaskId taskId) const
		{
			return m_taskInfos[taskId.value()].m_task;
		}

		/// A task is active if it is set to a valid task pointer.
		/// Otherwise it is just a place holder for an optional task.
		HK_FORCE_INLINE hkBool32 isTaskActive(TaskId taskId) const
		{
			return !(UintPtr(m_taskInfos[taskId.value()].m_task) & 1);
		}

		HK_FORCE_INLINE hkUint16 getInactiveTaskIndex(TaskId taskId) const
		{
			return hkUint16(UintPtr(m_taskInfos[taskId.value()].m_task) >> 1);
		}

		/// The maximum number of tasks that can be available at any given time. Pass in 0 to calculate it automatically
		/// with calculateMaxAvailableTasks (beware, that this can be very costly).
		void finish(int maxAvailableTasks, ExecutionPolicy executionPolicy = EXECUTION_POLICY_NONE);

		/// Reset the task graph to its initial state
		void reset(bool unrefTasks = false);

		//
		// hkTaskGraph implementation
		//

		virtual void addTasks( hkTask** tasks, int numTasks, const int* multiplicities, int* taskIdsOut ) HK_OVERRIDE;

		virtual void addDependencies( Dependency* dependencies, int numDependencies ) HK_OVERRIDE;

		virtual int getNumTasks() HK_OVERRIDE;

		
		HK_FORCE_INLINE TaskId addTask( hkTask* task, int multiplicity = 1 )
		{
			hkTask** tasks = &task;
			int id;
			addTasks( tasks, 1, &multiplicity, &id );
			return TaskId(id);
		}

		
		HK_FORCE_INLINE void addDependency( TaskId parentId, TaskId childId )
		{
			Dependency d;
			d.m_parentId = parentId.value();
			d.m_childId = childId.value();
			addDependencies( &d, 1 );
		}

	protected:

		struct TaskDetph
		{
			TaskId m_taskId;
			int m_depth;
			static HK_FORCE_INLINE bool HK_CALL lessDepthFirst(const TaskDetph& jA, const TaskDetph& jB)	{ return jB.m_depth < jA.m_depth; }
			static HK_FORCE_INLINE bool HK_CALL lessBreadthFirst(const TaskDetph& jA, const TaskDetph& jB)	{ return jA.m_depth < jB.m_depth; }
		};

		void calculateTaskDepths(hkArray<TaskDetph>& taskDepths);
		int calculateTaskDepthsRec(TaskDetph& node, hkArray<TaskDetph>& taskDepths);
		void setDepthFirstExecOrderRec(TaskId taskId, int& newId, const hkArray<int>& taskIdToDepthId, hkArray<TaskDetph>& taskDepths, hkArray<int>& remapTable);
		void reorderTasksForExecutionPolicy(ExecutionPolicy executionPolicy);

		void reshuffleTasks(hkArray<int>& remapTable);

		
		int calculateMaxAvailableTasks() const;

		int calculateMaxAvailableTasks(TaskId taskId) const;

	public:

		hkArray<TaskInfo> m_taskInfos;
		hkArray<TaskId> m_children;
		hkArray<Dependency> m_dependencies;
		int m_maxAvailableTasks;
		hkUint16 m_numInactiveTasks;
};

#endif // HK_TASK_GRAPH_H

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
