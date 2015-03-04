/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_RAYCAST_TASK_H
#define HKNP_RAYCAST_TASK_H

#include <Common/Base/Thread/Task/hkTask.h>
#include <Physics/Physics/Collide/Query/Multithreaded/hknpQuerySharedData.h>

class hknpRaycastTask;


/// A subtask for use with hknpRaycastTask, to cast a ray against a shape.
class hknpShapeRaycastSubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeRaycastSubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpShapeRaycastSubTask();

		void process( hknpRaycastTask* task );

#else

		/// Transfers all required data from PPU and processes the subtask.
		void process(
			hknpRaycastTask* spuJob, const hknpShapeRaycastSubTask* subTaskPpu, void* preloadBuffer,
			hknpCollisionQueryDispatcherBase* pQueryDispatcher );

#endif

	public:

		hknpRayCastQuery m_query;

		const hknpShape* m_targetShape;
		hkUint32 m_targetShapeSize;
		hkTransform m_targetShapeToWorld;
		hknpQueryFilterData m_targetShapeFilterData;
		hknpShapeQueryInfo m_targetShapeInfo;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;
};


/// A subtask for use with hknpRaycastTask, to cast a ray through a world.
class hknpWorldRaycastSubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpWorldRaycastSubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpWorldRaycastSubTask();

		void process( hknpRaycastTask* task );

#else

		/// Transfers all required data from PPU and processes the subtask.
		void process(
			hknpRaycastTask* spuJob, const hknpWorldRaycastSubTask* taskPpu, void* preloadBuffer,
			hknpCollisionQueryDispatcherBase* pQueryDispatcher );

#endif

	public:

		hknpWorld* m_pWorld;

		hknpRayCastQuery m_query;

		// for filtering:
		const hknpShapeTagCodec* m_shapeTagCodec;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;
};


/// A task which performs a set of raycast subtasks.
class hknpRaycastTask : public hkTask
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		enum SubTaskType
		{
			SHAPE_RAYCAST,
			WORLD_RAYCAST,
			NUM_SUB_TASK_TYPES
		};

#if defined(HK_PLATFORM_SPU)

		/// Groups used for DMA transfers
		enum DmaGroup
		{
			DMA_GROUP_GENERAL_DATA,
			DMA_GROUP_THREAD_CONTEXT,
			DMA_GROUP_WRITE_BACK_STREAMS
		};

#endif

		/// A structure which stores a subtask of any type.
		struct SubTask
		{
			SubTaskType m_subTaskType;

			HK_ALIGN16(union)
			{
				hkUint8 m_shapeRaycastSubTask[sizeof(hknpShapeRaycastSubTask)];
				hkUint8 m_worldRaycastSubTask[sizeof(hknpWorldRaycastSubTask)];
			} m_taskData;

			hknpShapeRaycastSubTask* asShapeRaycastSubTask() { HK_ASSERT(0x4fbe0754, m_subTaskType == SHAPE_RAYCAST); return (hknpShapeRaycastSubTask*)m_taskData.m_shapeRaycastSubTask; }
			hknpWorldRaycastSubTask* asWorldRaycastSubTask() { HK_ASSERT(0x4fbe0754, m_subTaskType == WORLD_RAYCAST); return (hknpWorldRaycastSubTask*)m_taskData.m_worldRaycastSubTask; }
		};

#if !defined(HK_PLATFORM_SPU)

		hknpRaycastTask();

		void reset();

		hknpShapeRaycastSubTask* allocateShapeRaycastSubTask();
		hknpWorldRaycastSubTask* allocateWorldRaycastSubTask();

		// hkTask implementation
		virtual void process() HK_OVERRIDE;

#endif

#if defined(HK_PLATFORM_HAS_SPU)

		// hkTask implementation
		virtual void* getElf() HK_OVERRIDE;

#endif

	public:

		/// Data shared between all subtasks.
		hknpQuerySharedData m_querySharedData;

		/// Array of independent subtasks.
		hkArray<SubTask> m_subTasks;

		/// Current subtask. Atomically incremented during process().
		hkUint32 m_currentSubTaskIndex;

		/// The number of subtasks to grab at a time (on SPU).
		hkPadSpu<hkUint32> m_maxGrab;

#if defined(HK_PLATFORM_HAS_SPU)

		/// PPU v-table pointers of each shape type
		void* m_shapeVTablesPpu[hknpShapeType::NUM_SHAPE_TYPES];

#endif
};

#endif // HKNP_RAYCAST_TASK_H

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
