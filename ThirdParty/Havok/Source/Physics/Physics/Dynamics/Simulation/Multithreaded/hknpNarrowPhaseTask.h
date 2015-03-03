/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_NARROW_PHASE_TASK_H
#define HKNP_NARROW_PHASE_TASK_H

#include <Common/Base/Thread/Task/hkTask.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpCollidePipeline.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>

#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>

class hknpInternalCollideSharedData;
class hknpLiveJacobianInfoGrid;
struct hknpNarrowPhaseWriters;


/// A task which processes a set of independent narrow phase collision detection subtasks.
/// This task can be processed multiple times on different threads in order to process the subtasks in parallel.
class hknpNarrowPhaseTask : public hkTask
{
	public:

		///
		struct ThreadOutput
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpNarrowPhaseTask::ThreadOutput );

				ThreadOutput() {}

				void initData( const hknpSimulationThreadContext& threadContext );
				void exitData( const hknpSimulationThreadContext& threadContext );

			public:

				// True if the init function is called
				hkBool m_isInitialized;

				hknpCdCacheStream m_cdCacheStream;
				hknpCdCacheStream m_inactiveCdCacheStream;
				hknpCdCacheStream m_crossGridCdCacheStream;

				hknpCdCacheStream m_childCdCacheStream;
				hknpCdCacheStream m_inactiveChildCdCacheStream;
				hknpCdCacheStream m_crossChildCdCacheStream;
		};

		/// A narrow phase subtask. Corresponds to a single grid link.
		struct SubTask
		{
			HK_FORCE_INLINE bool operator< (const SubTask& other) const { return (m_relativeCost > other.m_relativeCost); }

			hknpLinkId m_linkIndex;		///< The grid link index
			hknpCellIndex m_cellIndex;	///< If link is on the diagonal of the link grid, this cellIndex is set
			int m_relativeCost;			///< An estimated processing cost, relative to other subtasks
			HK_ON_PLATFORM_HAS_SPU(hkUint32 m_padding[2]);
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

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

	#if !defined(HK_PLATFORM_SPU)

		/// Constructor.
		hknpNarrowPhaseTask(
			hknpSimulationContext& simulationContext, hknpCollisionCacheManager& cdCacheManager,
			hknpSolverData& solverData, bool processOnlyNew, hknpCdCacheGrid& inactiveCdCacheGrid,
			hknpCdCacheGrid& crossGridCdCacheGrid, bool processPpuCaches = false );

	#else

		/// Destructor.
		hknpNarrowPhaseTask() {}

	#endif

		//
		// hknpTask implementation
		//

		/// Process subtasks while there are any available.
		HK_ON_CPU(virtual) void process() HK_ON_CPU(HK_OVERRIDE);

	#if defined(HK_PLATFORM_PPU)

		virtual void* getElf() HK_OVERRIDE;

	#endif

	protected:

		/// Process a single subtask.
		HK_FORCE_INLINE bool processSubTask(
			const hknpSimulationThreadContext& threadContext, const hknpInternalCollideSharedData& sharedData,
			hknpSolverData* HK_RESTRICT solverData, hkUint32 subTaskIndex, hknpNarrowPhaseWriters& writers );

	public:

		// Current work item, index into the task list, atomically incremented by process().
		hkPadSpu<hkUint32> m_currentSubTaskIndex;
		hkPadSpu<hkUint32*> m_currentSubTaskIndexPpu;

		// List of subtasks. These can be processed in parallel.
		hkArray<SubTask> m_subTasks;

		hknpCollideSharedData m_sharedData;

		// Jacobians
		hkPadSpu<hknpSolverData*> m_solverData;

		// Temp streams used by the threads to put the data
		hkArray<ThreadOutput> m_cdCacheStreamsOut;

		// Input caches
		hkPadSpu<hknpCdCacheStream*>	m_cdCacheStreamIn;
		hknpCdCacheGridEntries			m_cdCacheGridIn;
		hkPadSpu<hknpCdCacheStream*>	m_childCdCacheStreamIn;
		hkPadSpu<hknpCdCacheStream*>	m_cdCacheStreamIn2;
		hknpCdCacheGridEntries			m_cdCacheGridIn2;
		hkPadSpu<hknpCdCacheStream*>	m_childCdCacheStreamIn2;

		// Output caches
		hknpCdCacheGridEntries	m_cdCacheGridOut;
		hknpCdCacheGridEntries	m_inactiveCdCacheGridOut;
		hknpCdCacheGridEntries	m_crossGridCdCacheGridOut;

		// This is subtracted from the local thread number to access the corresponding thread output.
		hkUint8 m_firstThread;

		// True if this job must run on PPU only.
		hkBool m_runOnPpu;

	#if defined(HK_PLATFORM_HAS_SPU)

		/// PPU v-table pointers of each shape type
		void* m_shapeVTablesPpu[hknpShapeType::NUM_SHAPE_TYPES];

	#endif
};


#if defined(HK_PLATFORM_SPU)

#include <Common/Base/Thread/JobQueue/hkJobQueue.h>

/// SPU based multi-threaded narrow phase
void HK_CALL hknpNarrowPhaseTaskSpu(const hknpNarrowPhaseTask* task);

#endif


#endif // HKNP_NARROW_PHASE_TASK_H

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
