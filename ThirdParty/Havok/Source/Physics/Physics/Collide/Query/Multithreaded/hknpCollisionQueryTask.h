/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_QUERY_TASK_H
#define HKNP_COLLISION_QUERY_TASK_H

#include <Common/Base/Thread/Task/hkTask.h>
#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQuerySubTask.h>

/// Task used to process a collection of collision queries (sub tasks) multithreadedly. Add queries using the
/// allocateSubTask() methods and filling in the appropiate info in the returned objects. Once done, call process() from
/// as many threads as desired to start processing the queries. All queries will have been performed when the last
/// thread returns from the process() call.
class hknpCollisionQueryTask : public hkTask
{
	public:

#if defined(HK_PLATFORM_SPU)
		typedef void (*subTaskPreloadFP)(hknpCollisionQuerySubTask* pSubTask, void* preloadBuffer, hkUint32 dmaTag);
		typedef void (*subTaskProcessFP)(hknpCollisionQueryTask* task, hknpCollisionQuerySubTask* pSubTask, const hknpCollisionQuerySubTask* pSubTaskPpu, void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher);

		struct subTaskFunctions
		{
			subTaskPreloadFP m_preloadFP;
			subTaskProcessFP m_processFP;
		};

		/// Groups used for DMA transfers
		enum DmaGroup
		{
			DMA_GROUP_GENERAL_DATA,
			DMA_GROUP_THREAD_CONTEXT,
			DMA_GROUP_WRITE_BACK_STREAMS
		};
#endif

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpCollisionQueryTask();

		//
		// Methods used to add queries to the task
		//

		hknpAabbQuerySubTask* allocateAabbQuerySubTask();
		hknpPairGetClosestPointsSubTask* allocatePairGetClosestPointsSubTask();
		hknpWorldGetClosestPointsSubTask* allocateWorldGetClosestPointsSubTask();
		hknpPairShapeCastSubTask* allocatePairShapeCastSubTask();
		hknpWorldShapeCastSubTask* allocateWorldShapeCastSubTask();

		/// Process all queries
		virtual void process() HK_OVERRIDE;

		/// Clear the array of sub tasks.
		void reset();

#if defined(HK_PLATFORM_HAS_SPU)
		/// Returns the elf used to process the task on SPU.
		virtual void* getElf() HK_OVERRIDE;
	#if defined(HK_PLATFORM_SPU)
		/// Clears the g_subTaskFunction table.
		HK_FORCE_INLINE static void resetSubtasks();

		/// Registers a sub task type on SPU.
		HK_FORCE_INLINE static void registerSubTask(hknpCollisionQuerySubTask::SubTaskType subTaskType);
	#endif
#endif

	public:

		/// Data shared between all subtasks.
		hknpQuerySharedData m_querySharedData;

		/// Array of independent subtasks.
		hkArray<hknpCollisionQuerySubTask> m_subTasks;

		/// Current subtask. Atomically incremented during process().
		hkUint32 m_currentSubTaskIndex;

		/// The number of subtasks to grab at a time (on SPU).
		hkPadSpu<hkUint32> m_maxGrab;

#if defined(HK_PLATFORM_HAS_SPU)
		/// PPU v-table pointers of each shape type
		void* m_shapeVTablesPpu[hknpShapeType::NUM_SHAPE_TYPES];
	#if defined(HK_PLATFORM_SPU)
		static subTaskFunctions g_subTaskFunctions[hknpCollisionQuerySubTask::NUM_SUB_TASK_TYPES];
	#endif
#endif
};

#include <Physics/Physics/Collide/Query/Multithreaded/hknpCollisionQueryTask.inl>

#endif // HKNP_COLLISION_QUERY_TASK_H

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
