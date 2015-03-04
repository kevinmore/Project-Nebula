/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Task/hkTaskGraph.h>
#include <Common/Base/Thread/Task/hkTask.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>


void hkDefaultTaskGraph::addTasks( hkTask** tasks, int numTasks, const int* multiplicities, int* taskIdsOut )
{
	HK_ASSERT(0x6ca69f2d, m_taskInfos.getSize() + numTasks < MAX_TASKS);
	m_taskInfos.reserve( m_taskInfos.getSize() + numTasks );
	for( int i=0; i<numTasks; i++ )
	{
		if( taskIdsOut )
		{
			taskIdsOut[i] = m_taskInfos.getSize();
		}

		TaskInfo* HK_RESTRICT taskInfo = m_taskInfos.expandByUnchecked(1);
		taskInfo->m_task = tasks[i];
		taskInfo->m_numParents = 0;
		taskInfo->m_numChildren = 0;
		HK_ASSERT2(0x7c8a7e94, !multiplicities || (multiplicities[i] > 0 && multiplicities[i] < 256),
			"Multiplicity value out of range");
		taskInfo->m_multiplicity = (Multiplicity) (multiplicities ? multiplicities[i] : 1);

		// If the task is NULL we will mark it as inactive and store its inactive task index
		if( !tasks[i] )
		{
			taskInfo->m_task = (hkTask*)((m_numInactiveTasks++ << 1) | 1);
		}
	}
}

void hkDefaultTaskGraph::addDependencies( Dependency* dependencies, int numDependencies )
{
	m_dependencies.reserve( m_dependencies.getSize() + numDependencies );
	for( int i=0; i<numDependencies; i++ )
	{
		Dependency& d = dependencies[i];
		HK_ASSERT( 0x386b8d39, (d.m_parentId < m_taskInfos.getSize()) && (d.m_childId < m_taskInfos.getSize()) );
		m_dependencies.pushBackUnchecked( d );

		// Increase parent count only if the parent is active
		if( isTaskActive( TaskId(d.m_parentId) ) )
		{
			m_taskInfos[d.m_childId].m_numParents++;
		}
	}
}

/// Reset the task graph to its initial state
void hkDefaultTaskGraph::reset(bool unrefTasks)
{
	if ( unrefTasks )
	{
		for (int i = m_taskInfos.getSize() - 1 ; i >= 0 ; i--)
		{
			m_taskInfos[i].m_task->removeReference();
		}
	}
	m_taskInfos.setSize(0);
	m_children.setSize(0);
	m_dependencies.setSize(0);
	m_numInactiveTasks = 0;
	m_maxAvailableTasks = 0;
}

int hkDefaultTaskGraph::getNumTasks()
{
	return m_taskInfos.getSize();
}

void hkDefaultTaskGraph::finish( int maxAvailableTasks, ExecutionPolicy executionPolicy )
{
	if (m_dependencies.getSize())
	{
		int numTasks = m_taskInfos.getSize();
		hkLocalBuffer<int> counts(numTasks + 1);
		for (int i = 0; i < numTasks + 1; ++i)
		{
			counts[i] = 0;
		}

		// Count the number of children per task
		for (int i = 0; i < m_dependencies.getSize(); i++)
		{
			counts[m_dependencies[i].m_parentId + 1]++;
		}

		// Set the number of children and the index of the first child in the children array
		for (int i = 0; i < numTasks; i++)
		{
			HK_ASSERT(0x369e32c6, counts[i + 1] <= MAX_DEPENDENCIES);
			m_taskInfos[i].m_numChildren = (DependencyCount)counts[i + 1];
			counts[i + 1] += counts[i];
			m_taskInfos[i].m_firstChildIndex = counts[i];
		}

		// Create children array
		m_children.setSize(counts[numTasks]);
		for (int i = 0; i < m_dependencies.getSize(); ++i)
		{
			const TaskId parentId( m_dependencies[i].m_parentId );
			const TaskId childId( m_dependencies[i].m_childId );
			m_children[counts[parentId.value()]++] = childId;

			// If the task is not active, make sure that the child task has active parents
			if (!isTaskActive(parentId) && !m_taskInfos[childId.value()].m_numParents)
			{
				HK_ERROR(0x67c01868, "You cannot have tasks with only inactive parents");
			}
		}

		reorderTasksForExecutionPolicy(executionPolicy);
	}
	m_dependencies.clearAndDeallocate();

	if (maxAvailableTasks)
	{
		m_maxAvailableTasks = maxAvailableTasks;
	}
	else
	{
		m_maxAvailableTasks = calculateMaxAvailableTasks();
	}
}

void hkDefaultTaskGraph::reorderTasksForExecutionPolicy( ExecutionPolicy executionPolicy )
{
	if ( executionPolicy == EXECUTION_POLICY_NONE )
	{
		return;
	}

	// First, compute depth for each task
	hkArray<TaskDetph> taskDepths;
	calculateTaskDepths(taskDepths);

	// Sort them and deduce task ordering
	hkArray<int> remapTable(m_taskInfos.getSize(), -1);

	if ( executionPolicy == EXECUTION_POLICY_DEPTH_FIRST )
	{
		hkSort(taskDepths.begin(), taskDepths.getSize(), TaskDetph::lessDepthFirst);

		// compute mapping taskId->depth id
		hkArray<int> taskIdToDepthId(taskDepths.getSize());
		for (int i = 0, s = taskDepths.getSize() ; i < s ; i++)
		{
			taskIdToDepthId[taskDepths[i].m_taskId.value()] = i;
		}

		// Set execution order by visiting the graph in a depth first manner, starting with highest depth nodes
		int newId = 0;
		for (int i = 0 ; i < taskDepths.getSize() ; i++)
		{
			setDepthFirstExecOrderRec(taskDepths[i].m_taskId, newId, taskIdToDepthId, taskDepths, remapTable);
		}
	}
	else if ( executionPolicy == EXECUTION_POLICY_BREADTH_FIRST )
	{
		// NOT TESTED YET !!!!
		hkSort(taskDepths.begin(), taskDepths.getSize(), TaskDetph::lessBreadthFirst);
		for (int i = 0 ; i < m_taskInfos.getSize() ; i++)
		{
			remapTable[taskDepths[i].m_taskId.value()] = i;
		}
	}
	else
	{
		HK_WARN(0x458aefc0, "Execution policy not defined!");
		return;
	}

	// Reshuffle according the new order
	reshuffleTasks(remapTable);
}

void hkDefaultTaskGraph::setDepthFirstExecOrderRec( TaskId taskId, int& newId, const hkArray<int>& taskIdToDepthId, hkArray<TaskDetph>& taskDepths, hkArray<int>& remapTable )
{
	const int prevId = taskId.value();

	// mapping already set for this task
	if ( remapTable[prevId] != -1)
	{
		return;
	}

	// set mapping
	remapTable[prevId]	= newId++;

	// Get concerned task info
	const TaskInfo& nodeInfo	= m_taskInfos[prevId];
	const TaskDetph& nodeDepth	= taskDepths[taskIdToDepthId[prevId]];
	const int numChildren		= nodeInfo.m_numChildren;
	int childIndex				= nodeInfo.m_firstChildIndex;

	// update recursively the children
	for (int i = 0; i < numChildren ; ++i, ++childIndex)
	{
		const int childTaskId	= m_children[childIndex].value();
		const int childDepthId	= taskIdToDepthId[childTaskId];
		TaskDetph& childDepth	= taskDepths[childDepthId];

		// update
		if ( childDepth.m_depth == nodeDepth.m_depth - 1 )
		{
			setDepthFirstExecOrderRec(TaskId(childTaskId), newId, taskIdToDepthId, taskDepths, remapTable);
		}
	}
}

void hkDefaultTaskGraph::reshuffleTasks(hkArray<int>& remapTable)
{
	// Create two brand new arrays to receive the result of the reshuffle
	hkArray<TaskInfo> tasksRemap(m_taskInfos.getSize());
	hkArray<TaskId> childrenRemap;
	childrenRemap.reserve(m_children.getSize());

	// Build indirect remap table
	hkArray<int> indirectRemapTable(remapTable.getSize());
	for (int i = 0, s = remapTable.getSize() ; i < s ; i++)
	{
		indirectRemapTable[remapTable[i]] = i;
	}

	// copy and remap tasks
	for (int i = 0, s = m_taskInfos.getSize() ; i < s ; i++)
	{
		const TaskInfo& oldJInfo	= m_taskInfos[indirectRemapTable[i]];
		tasksRemap[i]			= oldJInfo;
		tasksRemap[i].m_firstChildIndex = childrenRemap.getSize();

		//copy and remap children
		for (int c = 0, ci = oldJInfo.m_firstChildIndex ; c < oldJInfo.m_numChildren ; c++, ci++)
		{
			childrenRemap.pushBack(TaskId(remapTable[m_children[ci].value()]));
		}
	}

	// Swap arrays
	m_taskInfos.swap(tasksRemap);
	m_children.swap(childrenRemap);
}

void hkDefaultTaskGraph::calculateTaskDepths(hkArray<TaskDetph>& taskDepths)
{
	taskDepths.setSize(m_taskInfos.getSize());

	// Init depths values
	for (int i = 0; i < m_taskInfos.getSize(); ++i)
	{
		TaskDetph& taskDepth	= taskDepths[i];
		taskDepth.m_taskId	= TaskId(i);
		taskDepth.m_depth	= -1;
	}

	// find tasks with no parents as roots
	for (int i = 0; i < m_taskInfos.getSize(); ++i)
	{
		if (m_taskInfos[i].m_numParents == 0)
		{
			// update depth of the children
			calculateTaskDepthsRec(taskDepths[i], taskDepths);
		}
	}
}

int hkDefaultTaskGraph::calculateTaskDepthsRec(TaskDetph& node, hkArray<TaskDetph>& taskDepths)
{
	if ( node.m_depth != -1)
	{
		return node.m_depth; // depth already computed for this node
	}

	const TaskInfo& nodeInfo = m_taskInfos[node.m_taskId.value()];
	int numChildren		= nodeInfo.m_numChildren;
	int childIndex		= nodeInfo.m_firstChildIndex;
	node.m_depth		= 0;

	// set depth as max depth of the children + 1
	for (int i = 0; i < numChildren ; ++i, ++childIndex)
	{
		// calculate depth of this child
		TaskDetph& childDepth = taskDepths[m_children[childIndex].value()];
		calculateTaskDepthsRec(childDepth, taskDepths);

		// update current depth consequently
		if ( node.m_depth < childDepth.m_depth + 1 ) node.m_depth = childDepth.m_depth + 1;
	}

	return node.m_depth;
}

int hkDefaultTaskGraph::calculateMaxAvailableTasks() const
{
	int max = 0;
	for (int i = 0; i < m_taskInfos.getSize(); ++i)
	{
		if (m_taskInfos[i].m_numParents == 0)
		{
			max++;
			int childMax = calculateMaxAvailableTasks(TaskId(i));
			if (childMax > 1)
			{
				max += childMax - 1;
			}
		}
	}

	return max;
}


int hkDefaultTaskGraph::calculateMaxAvailableTasks(TaskId taskId) const
{
	const TaskInfo& taskInfo = m_taskInfos[taskId.value()];
	int max = taskInfo.m_numChildren;
	int childIndex = taskInfo.m_firstChildIndex;
	for (int i = 0; i < taskInfo.m_numChildren; ++i, ++childIndex)
	{
		int childMax = calculateMaxAvailableTasks(m_children[childIndex]);
		if (childMax > 1)
		{
			max += childMax - 1;
		}
	}
	return max;
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
