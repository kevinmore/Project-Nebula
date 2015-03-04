/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_QUERY_SUB_TASK_H
#define HKNP_COLLISION_QUERY_SUB_TASK_H

#include <Physics/Physics/Collide/Query/Multithreaded/hknpQuerySharedData.h>

class hknpCollisionQueryTask;
struct hknpCollisionQuerySubTask;


/// A subtask for use with hknpCollisionQueryTask, to perform an AABB query.
class hknpAabbQuerySubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpAabbQuerySubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpAabbQuerySubTask();

		void initialize( const hknpShape* pShape, hkAabb& queryAabb );

		void process( hknpCollisionQueryTask* task );

#else

		static void preload( hknpCollisionQuerySubTask* pSubTask, void* preloadBuffer, hkUint32 dmaTag );

		static void process(
			hknpCollisionQueryTask* task, hknpCollisionQuerySubTask* pSubTask, const hknpCollisionQuerySubTask* pSubTaskPpu,
			void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

		/// Transfers all required data from PPU and processes the subtask.
		void process(hknpCollisionQueryTask* task, const hknpAabbQuerySubTask* subTaskPpu, void* preloadBuffer);

#endif

	public:

		const hknpShape* m_pShape;
		int m_shapeSize;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;

		hkAabb m_aabb;
		hkPadSpu< hknpCollisionFilter* > m_filter;
		hknpMaterialId m_materialId;	///< The material id associated with the shape. Allowed to be 'invalid'.
		hkUint32 m_collisionFilterInfo;	///< The collision filter info associated with the shape.
		hkUint64 m_userData;			///< The user data associated with the shape.
};


/// A subtask for use with hknpCollisionQueryTask, to get closest points between two shapes.
class hknpPairGetClosestPointsSubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpPairGetClosestPointsSubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpPairGetClosestPointsSubTask();

		void initialize( const hknpBody& queryBody, const hknpBody& targetBody );

		void process( hknpCollisionQueryTask* task );

#else

		static void preload( hknpCollisionQuerySubTask* pSubTask, void* preloadBuffer, hkUint32 dmaTag );

		static void process(
			hknpCollisionQueryTask* task, hknpCollisionQuerySubTask* pSubTask, const hknpCollisionQuerySubTask* pSubTaskPpu,
			void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

		/// Transfers all required data from PPU and processes the subtask.
		void process(
			hknpCollisionQueryTask* task, const hknpPairGetClosestPointsSubTask* subTaskPpu, void* preloadBuffer,
			const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

#endif

	public:

		const hknpShape* m_queryShape;
		hkUint32 m_queryShapeSize;
		hkTransform m_queryShapeToWorld;

		const hknpShape* m_targetShape;
		hkUint32 m_targetShapeSize;
		hkTransform m_targetShapeToWorld;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;
};


/// A subtask for use with hknpCollisionQueryTask, to get the closest points to any shape in a world.
class hknpWorldGetClosestPointsSubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpWorldGetClosestPointsSubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpWorldGetClosestPointsSubTask();

		void initialize( hknpWorld* pWorld, hknpClosestPointsQuery& query );

		void process( hknpCollisionQueryTask* task );

#else

		static void preload( hknpCollisionQuerySubTask* pSubTask, void* preloadBuffer, hkUint32 dmaTag );

		static void process(
			hknpCollisionQueryTask* task, hknpCollisionQuerySubTask* pSubTask, const hknpCollisionQuerySubTask* pSubTaskPpu,
			void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

		/// Transfers all required data from PPU and processes the subtask.
		void process(
			hknpCollisionQueryTask* task, const hknpWorldGetClosestPointsSubTask* subTaskPpu, void* preloadBuffer,
			const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

#endif

	public:

		hknpWorld* m_pWorld;
		hknpClosestPointsQuery m_query;
		hkUint32 m_queryShapeSize;
		hkTransform m_queryShapeTransform;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;
};


/// A subtask for use with hknpCollisionQueryTask, to cast a shape against another.
class hknpPairShapeCastSubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpPairShapeCastSubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpPairShapeCastSubTask();

		void process( hknpCollisionQueryTask* task );

#else

		static void preload( hknpCollisionQuerySubTask* pSubTask, void* preloadBuffer, hkUint32 dmaTag );

		static void process(
			hknpCollisionQueryTask* task, hknpCollisionQuerySubTask* pSubTask, const hknpCollisionQuerySubTask* pSubTaskPpu,
			void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

		/// Transfers all required data from PPU and processes the subtask.
		void process(
			hknpCollisionQueryTask* task, const hknpPairShapeCastSubTask* subTaskPpu, void* preloadBuffer,
			const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

#endif

	public:

		hknpShapeCastQuery m_query;

		hkUint32 m_queryShapeSize;
		hkRotation m_queryShapeOrientationInWorld;
		hkTransform m_queryShapeToWorld;

		const hknpShape* m_targetShape;
		hkUint32 m_targetShapeSize;
		hkTransform m_targetShapeToWorld;

		// for filtering:
		hknpShapeQueryInfo m_queryShapeInfo;
		hknpShapeQueryInfo m_targetShapeInfo;
		hknpQueryFilterData m_targetShapeFilterData;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;
};


/// A subtask for use with hknpCollisionQueryTask, to cast a shape through a world.
class hknpWorldShapeCastSubTask
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpWorldShapeCastSubTask );

#if !defined(HK_PLATFORM_SPU)

		hknpWorldShapeCastSubTask();

		void process( hknpCollisionQueryTask* task );

#else

		static void preload( hknpCollisionQuerySubTask* pSubTask, void* preloadBuffer, hkUint32 dmaTag );

		static void process(
			hknpCollisionQueryTask* task, hknpCollisionQuerySubTask* pSubTask, const hknpCollisionQuerySubTask* pSubTaskPpu,
			 void* preloadBuffer, const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

		/// Transfers all required data from PPU and processes the subtask.
		void process(
			hknpCollisionQueryTask* task, const hknpWorldShapeCastSubTask* taskPpu, void* preloadBuffer,
			const hknpCollisionQueryDispatcherBase* pQueryDispatcher );

#endif

	public:

		hknpWorld* m_pWorld;

		hknpShapeCastQuery m_query;

		const hknpShape* m_queryShape;
		hkUint32 m_queryShapeSize;
		hkRotation m_queryShapeOrientationInWorld;
		hkTransform m_queryShapeToWorld;

		// for filtering:
		hknpShapeQueryInfo m_queryShapeInfo;

		hkPadSpu<hknpCollisionResult*> m_resultArray;
		hkPadSpu<int> m_resultArraySize;
		hkPadSpu<int> m_numHits;
};


/// A hknpCollisionQueryTask subtask of any type.
struct hknpCollisionQuerySubTask
{
	enum SubTaskType
	{
		COLLISION_QUERY_AABB,
		COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS,
		COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS,
		COLLISION_QUERY_PAIR_SHAPE_CAST,
		COLLISION_QUERY_WORLD_SHAPE_CAST,

		NUM_SUB_TASK_TYPES
	};

	SubTaskType m_subTaskType;

	HK_ALIGN16(union)
	{
		hkUint8 m_aabbQuerySubTask[sizeof(hknpAabbQuerySubTask)];
		hkUint8 m_pairGetClosestPointsSubTask[sizeof(hknpPairGetClosestPointsSubTask)];
		hkUint8 m_worldGetClosestPointsSubTask[sizeof(hknpWorldGetClosestPointsSubTask)];
		hkUint8 m_pairShapeCastSubTask[sizeof(hknpPairShapeCastSubTask)];
		hkUint8 m_worldShapeCastSubTask[sizeof(hknpWorldShapeCastSubTask)];
	} m_taskData;

	hknpAabbQuerySubTask* asAabbQuerySubTask() { HK_ASSERT(0x4fbe0754, m_subTaskType == COLLISION_QUERY_AABB); return (hknpAabbQuerySubTask*)m_taskData.m_aabbQuerySubTask; }
	hknpPairGetClosestPointsSubTask* asPairGetClosestPointsSubTask() { HK_ASSERT(0x68aaf89d, m_subTaskType == COLLISION_QUERY_PAIR_GET_CLOSEST_POINTS); return (hknpPairGetClosestPointsSubTask*)m_taskData.m_pairGetClosestPointsSubTask; }
	hknpWorldGetClosestPointsSubTask* asWorldGetClosestPointsSubTask() { HK_ASSERT(0xbe45fc8, m_subTaskType == COLLISION_QUERY_WORLD_GET_CLOSEST_POINTS); return (hknpWorldGetClosestPointsSubTask*)m_taskData.m_worldGetClosestPointsSubTask; }
	hknpPairShapeCastSubTask* asPairShapeCastSubTask() { HK_ASSERT(0x30fb29b9, m_subTaskType == COLLISION_QUERY_PAIR_SHAPE_CAST); return (hknpPairShapeCastSubTask*)m_taskData.m_pairShapeCastSubTask; }
	hknpWorldShapeCastSubTask* asWorldShapeCastSubTask() { HK_ASSERT(0xed2f27f, m_subTaskType == COLLISION_QUERY_WORLD_SHAPE_CAST); return (hknpWorldShapeCastSubTask*)m_taskData.m_worldShapeCastSubTask; }

	const hknpAabbQuerySubTask* asAabbQuerySubTaskConst() const { return (const hknpAabbQuerySubTask*)m_taskData.m_aabbQuerySubTask; }
	const hknpPairGetClosestPointsSubTask* asPairGetClosestPointsSubTaskConst() const { return (const hknpPairGetClosestPointsSubTask*)m_taskData.m_pairGetClosestPointsSubTask; }
	const hknpWorldGetClosestPointsSubTask* asWorldGetClosestPointsSubTaskConst() const { return (const hknpWorldGetClosestPointsSubTask*)m_taskData.m_worldGetClosestPointsSubTask; }
	const hknpPairShapeCastSubTask* asPairShapeCastSubTaskConst() const { return (const hknpPairShapeCastSubTask*)m_taskData.m_pairShapeCastSubTask; }
	const hknpWorldShapeCastSubTask* asWorldShapeCastSubTaskConst() const { return (const hknpWorldShapeCastSubTask*)m_taskData.m_worldShapeCastSubTask; }
};

#endif // HKNP_COLLISION_QUERY_SUB_TASK_H

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
