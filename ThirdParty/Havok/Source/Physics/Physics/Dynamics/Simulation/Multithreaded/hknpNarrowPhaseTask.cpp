/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpNarrowPhaseTask.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>
#include <Physics/Physics/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianInfo.h>

#if !defined(HK_PLATFORM_SPU)
	#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
	#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
	#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>
#else
	#include <Physics/Physics/Dynamics/Simulation/Multithreaded/hknpNarrowPhaseJobSpu.cxx>
	#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

extern HK_THREAD_LOCAL(int) hkThreadNumber;


#if !defined(HK_PLATFORM_SPU)

hknpNarrowPhaseTask::hknpNarrowPhaseTask(
	hknpSimulationContext& simulationContext, hknpCollisionCacheManager& cdCacheManager, hknpSolverData& solverData,
	bool processOnlyNew, hknpCdCacheGrid& inactiveCdCacheGrid, hknpCdCacheGrid& crossGridCdCacheGrid,
	bool processPpuCaches )
	: m_runOnPpu(processPpuCaches)
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& mStream = hkMonitorStream::getInstance();
#endif
	HK_TIMER_BEGIN_LIST2(mStream, "CreateNarrowPhaseTask", "InitOutputGrids");

	hknpSimulationThreadContext* HK_RESTRICT threadContext = simulationContext.getThreadContext();
	hknpWorld* world = threadContext->m_world;
	const int numLinks = world->m_spaceSplitter->getNumLinks();

	// We will merge the output into inactiveCacheStreamOut later
	inactiveCdCacheGrid.setSize(numLinks);
	crossGridCdCacheGrid.setSize(numLinks);

	// Create shared data
	HK_TIMER_SPLIT_LIST2(mStream, "CreateSharedData" );
	{
		m_currentSubTaskIndex = 0;
		m_currentSubTaskIndexPpu = &m_currentSubTaskIndex.ref();

		hknpCdCacheGrid* newCdCacheGrid = &cdCacheManager.m_newCdCacheGrid;
		hknpCdCacheGrid* cdCacheGrid = &cdCacheManager.m_cdCacheGrid;
	#if defined(HK_PLATFORM_HAS_SPU)
		if (processPpuCaches)
		{
			newCdCacheGrid = &cdCacheManager.m_newCdCachePpuGrid;
			cdCacheGrid = &cdCacheManager.m_cdCachePpuGrid;
		}
	#endif
		hkUint32 enableRebuildCdCaches;
		if (processOnlyNew)
		{
			HK_ASSERT(0xf034de76, newCdCacheGrid->m_entries.getSize() == numLinks);
			m_cdCacheGridIn.set(newCdCacheGrid);
			m_cdCacheGridOut.set(newCdCacheGrid);
			m_cdCacheStreamIn = &cdCacheManager.m_newCdCacheStream;
			m_childCdCacheStreamIn = &cdCacheManager.m_newChildCdCacheStream;
			m_cdCacheGridIn2.set(HK_NULL);
			m_cdCacheStreamIn2 = HK_NULL;
			m_childCdCacheStreamIn2 = HK_NULL;
			enableRebuildCdCaches = 0;
		}
		else
		{
			HK_ASSERT(0xf034de76, cdCacheGrid->m_entries.getSize() == numLinks);
			m_cdCacheGridIn.set(cdCacheGrid);
			m_cdCacheGridOut.set(cdCacheGrid);
			m_cdCacheStreamIn = &cdCacheManager.m_cdCacheStream;
			m_childCdCacheStreamIn = &cdCacheManager.m_childCdCacheStream;
			m_cdCacheGridIn2.set(newCdCacheGrid);
			m_cdCacheStreamIn2 = &cdCacheManager.m_newCdCacheStream;
			m_childCdCacheStreamIn2 = &cdCacheManager.m_newChildCdCacheStream;
			enableRebuildCdCaches = hkUint32(~0);
		}

		m_sharedData.m_spaceSplitter = world->m_spaceSplitter;
		m_sharedData.m_spaceSplitterSize = world->m_spaceSplitter->getSize();
		m_sharedData.m_heapAllocator = threadContext->m_heapAllocator->m_blockStreamAllocator;
		m_sharedData.m_tempAllocator = threadContext->m_tempAllocator->m_blockStreamAllocator;
		m_sharedData.m_simulationContext = &simulationContext;
		m_sharedData.m_solverInfo = &world->m_solverInfo;
		m_sharedData.m_collisionTolerance = world->m_collisionTolerance;
		m_sharedData.m_bodies = world->m_bodyManager.accessBodyBuffer();
		m_sharedData.m_motions = world->m_motionManager.accessMotionBuffer();
		m_sharedData.m_qualities = threadContext->m_qualities;
		m_sharedData.m_numQualities = threadContext->m_numQualities;
		m_sharedData.m_materials = threadContext->m_materials;
		m_sharedData.m_numMaterials = threadContext->m_numMaterials;
		m_sharedData.m_intSpaceUtil = &world->m_intSpaceUtil;
		m_sharedData.m_enableRebuildCdCaches1 = enableRebuildCdCaches;
		m_sharedData.m_enableRebuildCdCaches2 = enableRebuildCdCaches;
	#if defined(HK_PLATFORM_HAS_SPU)
		m_sharedData.m_collisionFilter = world->m_modifierManager->getCollisionFilter();
		m_sharedData.m_collisionFilterType = m_sharedData.m_collisionFilter->m_type;
		m_sharedData.m_shapeTagCodec = world->getShapeTagCodec();
		m_sharedData.m_shapeTagCodecType = m_sharedData.m_shapeTagCodec->m_type;
		m_sharedData.m_globalModifierFlags = world->m_modifierManager->getGlobalBodyFlags();
	#endif

		m_solverData = &solverData;
		m_inactiveCdCacheGridOut.set(&inactiveCdCacheGrid);
		m_crossGridCdCacheGridOut.set(&crossGridCdCacheGrid);

		// Figure out the number of threads that will be working on this job and the lowest thread number
		int numThreads = simulationContext.getNumCpuThreads();
		int firstThread = 0;
	#if defined(HK_PLATFORM_HAS_SPU)
		if (!processPpuCaches)
		{
			firstThread = numThreads;
			numThreads = simulationContext.getNumSpuThreads();
		}
	#endif

		// Setup the thread outputs
		m_cdCacheStreamsOut.setSize(numThreads);
		for (int ti = 0; ti < numThreads; ti++)
		{
		#if defined( HK_PLATFORM_HAS_SPU )
			m_cdCacheStreamsOut[ti].initData( *simulationContext.getThreadContext(ti + firstThread) );
		#else
			m_cdCacheStreamsOut[ti].m_isInitialized = false;
		#endif
		}
		m_firstThread = (hkUint8) firstThread;
	}

	// Create subtasks
	HK_TIMER_SPLIT_LIST2( mStream, "CreateSubTasks" );
	{
		m_subTasks.reserve(numLinks);
		hknpSpaceSplitter* splitter = world->m_spaceSplitter;

		for (int cellIndexA = 0; cellIndexA < splitter->getNumCells(); cellIndexA++)
		{
			for (int cellIndexB = cellIndexA; cellIndexB < splitter->getNumCells(); cellIndexB++)
			{
				const int linkIndex = splitter->getLinkIdxUnchecked(cellIndexA, cellIndexB);

				// Body pair
				int numPairsCollisions = m_cdCacheGridIn.m_entries[linkIndex].getNumElements();
				if (!processOnlyNew)
				{
					numPairsCollisions += m_cdCacheGridIn2.m_entries[linkIndex].getNumElements();
				}

				if (numPairsCollisions > 0)
				{
					hknpNarrowPhaseTask::SubTask& subtask = *m_subTasks.expandByUnchecked(1);

					const int avgNumberOfCvxPairPerMeshCollision = 10;
					int estimatedCost = avgNumberOfCvxPairPerMeshCollision * numPairsCollisions;

					subtask.m_linkIndex = hknpLinkId(linkIndex);
					subtask.m_cellIndex = ( cellIndexA == cellIndexB ) ? hknpCellIndex(cellIndexA) : hknpCellIndex(HKNP_INVALID_CELL_IDX);
					subtask.m_relativeCost = estimatedCost;
				}
			}
		}

		// Sort by estimated cost, descending
		hkSort( m_subTasks.begin(), m_subTasks.getSize() );
	}

#if defined(HK_PLATFORM_PPU)
	hkString::memCpy4(m_shapeVTablesPpu, hknpShapeVirtualTableUtil::getVTables(), sizeof(m_shapeVTablesPpu) >> 2);
#endif

	HK_TIMER_END_LIST2(mStream); // "CreateNarrowPhaseTask"
}

#endif


struct hknpNarrowPhaseWriters
{
	public:

		hknpNarrowPhaseWriters(
			const hknpSimulationThreadContext& threadContext, hknpSolverData::ThreadData* HK_RESTRICT threadData,
			hknpNarrowPhaseTask::ThreadOutput& threadOutput )
			: m_movingJacobianSorter(&m_movingJacobianWriter), m_fixedJacobianSorter(&m_fixedJacobianWriter)
		{
			init(threadContext, threadData, threadOutput);
		}

		~hknpNarrowPhaseWriters()
		{
			m_cdCacheWriter.finalize();
			m_childCdCacheWriter.finalize();
			m_crossCdCacheWriter.finalize();
			m_crossGridChildCdCacheWriter.finalize();
			m_inactiveCdCacheWriter.finalize();
			m_inactiveChildCdCacheWriter.finalize();

		#if defined(HK_PLATFORM_SPU)
			m_inactiveChildCdCacheWriter.exitSpu();
			m_inactiveCdCacheWriter.exitSpu();
			m_crossGridChildCdCacheWriter.exitSpu();
			m_crossCdCacheWriter.exitSpu();
			m_childCdCacheWriter.exitSpu();
			m_cdCacheWriter.exitSpu();
		#endif

			// Finalize jacobian writers
			m_movingJacobianWriter.finalize();
			m_fixedJacobianWriter.finalize();
			HK_ON_CPU(m_liveJacobianInfoWriter.finalize());
			m_activePairWriter.finalize();

		#if defined(HK_PLATFORM_SPU)
			m_fixedJacobianWriter.exitSpu();
			m_movingJacobianWriter.exitSpu();
			m_activePairWriter.exitSpu();
		#endif
		}

	protected:

		void HK_INIT_FUNCTION(init)(
			const hknpSimulationThreadContext& threadContext, hknpSolverData::ThreadData* HK_RESTRICT threadData,
			hknpNarrowPhaseTask::ThreadOutput& threadOutput );

	public:

		hknpConstraintSolverJacobianWriter m_movingJacobianWriter;
		hknpConstraintSolverJacobianWriter m_fixedJacobianWriter;
		hknpMxJacobianSorter m_movingJacobianSorter;
		hknpMxJacobianSorter m_fixedJacobianSorter;

		// This writer is not used on SPU
		hknpLiveJacobianInfoWriter m_liveJacobianInfoWriter;

		hknpCdPairWriter m_activePairWriter;

		hknpCdCacheWriter	m_cdCacheWriter;
		hknpCdCacheWriter	m_childCdCacheWriter;
		hknpCdCacheWriter	m_crossCdCacheWriter;
		hknpCdCacheWriter	m_crossGridChildCdCacheWriter;
		hknpCdCacheWriter	m_inactiveCdCacheWriter;
		hknpCdCacheWriter	m_inactiveChildCdCacheWriter;
};


void hknpNarrowPhaseWriters::init(
	const hknpSimulationThreadContext& threadContext, hknpSolverData::ThreadData* HK_RESTRICT threadData,
	hknpNarrowPhaseTask::ThreadOutput& threadOutput )
{
#if defined(HK_PLATFORM_SPU)
	m_activePairWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 1, "ActivePairWriter");
	m_movingJacobianWriter.initSpu(HK_SPU_DMA_GROUP_STALL,
		1 + HKNP_MAX_NUM_MANIFOLDS_PER_BATCH / HKNP_NUM_MXJACOBIANS_PER_BLOCK,
		"JacMovingWriter");
	m_fixedJacobianWriter.initSpu(HK_SPU_DMA_GROUP_STALL,
		1 + HKNP_MAX_NUM_MANIFOLDS_PER_BATCH / HKNP_NUM_MXJACOBIANS_PER_BLOCK,
		"JacFixedWriter");
#endif

	// Note: Normally those output streams are empty, but sometimes one thread might be completely stalled, so that another
	// thread kicks in is entering this function for a second time.
	HK_ON_CPU(m_liveJacobianInfoWriter.setToEndOfStream(threadContext.m_tempAllocator, &threadData->m_liveJacInfoStream));
	m_activePairWriter.setToEndOfStream(threadContext.m_tempAllocator, &threadData->m_activePairStream);
	m_movingJacobianWriter.setToEndOfStream(threadContext.m_tempAllocator, &threadData->m_jacMovingStream);
	m_fixedJacobianWriter.setToEndOfStream(threadContext.m_tempAllocator, &threadData->m_jacFixedStream);

#if defined(HK_PLATFORM_SPU)
	m_cdCacheWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 0, "bodyPairWriter");
	const int numCachesPerBlock = hkBlockStreamBase::Block::BLOCK_DATA_SIZE / HKNP_MAX_CVX_CVX_CACHE_SIZE;
	const int numOpenCacheBlocks = 1 + HKNP_MAX_NUM_MANIFOLDS_PER_BATCH/numCachesPerBlock;
	m_childCdCacheWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 1 + numOpenCacheBlocks, "collideCacheWriter");
	m_crossCdCacheWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 0, "crossGridPairWriter");
	m_crossGridChildCdCacheWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 0, "crossGridMeshCvxWriter");
	m_inactiveCdCacheWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 0, "inactiveBodyPairWriter");
	m_inactiveChildCdCacheWriter.initSpu(HK_SPU_DMA_GROUP_STALL, 0, "inactiveChildCdCacheWriter");
#endif

	// Setup body pair cd cache writers
	m_cdCacheWriter.setToEndOfStream(threadContext.m_heapAllocator, &threadOutput.m_cdCacheStream);
	m_childCdCacheWriter.setToEndOfStream(threadContext.m_heapAllocator, &threadOutput.m_childCdCacheStream);
	m_crossCdCacheWriter.setToEndOfStream(threadContext.m_heapAllocator, &threadOutput.m_crossGridCdCacheStream);
	m_crossGridChildCdCacheWriter.setToEndOfStream(threadContext.m_heapAllocator, &threadOutput.m_crossChildCdCacheStream);
	m_inactiveCdCacheWriter.setToEndOfStream(threadContext.m_heapAllocator, &threadOutput.m_inactiveCdCacheStream);
	m_inactiveChildCdCacheWriter.setToEndOfStream(threadContext.m_heapAllocator, &threadOutput.m_inactiveChildCdCacheStream);
}


#if !defined(HK_PLATFORM_SPU)

void hknpNarrowPhaseTask::ThreadOutput::initData(const hknpSimulationThreadContext& threadContext)
{
	m_cdCacheStream.initBlockStream(threadContext.m_heapAllocator, true);
	m_inactiveCdCacheStream.initBlockStream(threadContext.m_heapAllocator);
	m_crossGridCdCacheStream.initBlockStream(threadContext.m_heapAllocator);

	m_childCdCacheStream.initBlockStream(threadContext.m_heapAllocator, true);
	m_inactiveChildCdCacheStream.initBlockStream(threadContext.m_heapAllocator);
	m_crossChildCdCacheStream.initBlockStream(threadContext.m_heapAllocator);

	m_isInitialized = true;
}

void hknpNarrowPhaseTask::ThreadOutput::exitData(const hknpSimulationThreadContext& threadContext)
{
	m_cdCacheStream.clear(threadContext.m_heapAllocator);
	m_inactiveCdCacheStream.clear(threadContext.m_heapAllocator);
	m_crossGridCdCacheStream.clear(threadContext.m_heapAllocator);

	m_childCdCacheStream.clear(threadContext.m_heapAllocator);
	m_inactiveChildCdCacheStream.clear(threadContext.m_heapAllocator);
	m_crossChildCdCacheStream.clear(threadContext.m_heapAllocator);
}

#endif

void hknpNarrowPhaseTask::process()
{
	HK_CHECK_FLUSH_DENORMALS();

	// Obtain thread monitor stream just once as access to TLS may be expensive
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif

	const int threadNumber = HK_THREAD_LOCAL_GET(hkThreadNumber);
	if( threadNumber )
	{
		// If this is being processed in a worker thread, set up the same timer path as the master thread
		HK_TIMER_BEGIN2( timerStream, "Physics", HK_NULL );
		HK_TIMER_BEGIN2( timerStream, "Collide", HK_NULL );
	}
	HK_TIMER_BEGIN_LIST2( timerStream, "NarrowPhaseTask", "Init" );

	hknpSimulationThreadContext* threadContext;
	hknpSolverData* solverData;
	hknpNarrowPhaseTask::ThreadOutput* threadOutput;

#if !defined(HK_PLATFORM_SPU)
	threadContext = m_sharedData.m_simulationContext->getThreadContext(threadNumber);
	solverData = m_solverData;
	threadOutput = &m_cdCacheStreamsOut[threadNumber - m_firstThread];
	if( !threadOutput->m_isInitialized )
	{
		threadOutput->initData( *threadContext );
	}
#else
	Buffers* buffers = hknpNarrowPhaseTask_init(*this);
	threadContext = (hknpSimulationThreadContext*)&buffers->m_threadContextBuffer;
	solverData = &buffers->m_solverData;
	threadOutput = &buffers->m_cdCacheStreamsOut;
#endif

	{
		// Initialize writers
		hknpNarrowPhaseWriters writers(*threadContext, &solverData->m_threadData[threadNumber], *threadOutput);

		// Process subtasks while available
		while (1)
		{
			const hkUint32 subTaskIndex = hkDmaManager::atomicExchangeAdd( m_currentSubTaskIndexPpu, 1 );
			if( int(subTaskIndex) < m_subTasks.getSize() )
			{
				HK_TIMER_SPLIT_LIST2( timerStream, "SubTask" );

			#if !defined(HK_PLATFORM_SPU)
				processSubTask(*threadContext, m_sharedData, solverData, subTaskIndex, writers);
			#else
				processSubTask(*threadContext, buffers->m_collideSharedData, solverData, subTaskIndex, writers);
			#endif
			}
			else
			{
				break;
			}
		}

		// Time the writers destructor
		HK_TIMER_SPLIT_LIST2( timerStream, "FinalizeWriters" );
	}

	HK_ON_SPU( hknpNarrowPhaseTask_finish(*this, buffers) );

	HK_TIMER_END_LIST2( timerStream );
	if( threadNumber )
	{
		HK_TIMER_END2( timerStream );
		HK_TIMER_END2( timerStream );
	}
}


// SPU_GET creates a local variable and loads the contents from the source PPU address into it
// SPU_RELEASE calls hkSpuDmaManager::performFinalChecks to indicate that the memory obtained is not going to be used anymore
#if !defined(HK_PLATFORM_SPU)
	#define SPU_GET(NAME, SRC, TYPE, MODE)	TYPE& NAME = *(SRC)
	#define SPU_RELEASE(NAME, SRC, TYPE)
#else
	#define SPU_GET(NAME, SRC, TYPE, MODE)	\
		HK_ALIGN16(TYPE NAME);	\
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(&NAME, SRC, sizeof(TYPE), (MODE));
	#define SPU_RELEASE(NAME, SRC, TYPE)	hkSpuDmaManager::performFinalChecks(SRC, &NAME, sizeof(TYPE));
#endif

HK_FORCE_INLINE bool hknpNarrowPhaseTask::processSubTask(
	const hknpSimulationThreadContext& threadContext, const hknpInternalCollideSharedData& sharedData,
	hknpSolverData* HK_RESTRICT solverData, hkUint32 taskIdx, hknpNarrowPhaseWriters& writers)
{
	// Obtain link and cell indexes from task
	int linkIdx;
	int cellIdx;
	{
		
		SPU_GET(collideTask, &m_subTasks[taskIdx], hknpNarrowPhaseTask::SubTask, hkSpuDmaManager::READ_ONLY);

		// Read link and cell indexes
		linkIdx = collideTask.m_linkIndex.value();
		cellIdx = collideTask.m_cellIndex;

		SPU_RELEASE(collideTask, &m_subTasks[taskIdx], hknpNarrowPhaseTask::SubTask);
	}

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	hkCheckDeterminismUtil::Fuid jobFuid = hkCheckDeterminismUtil::Fuid::getZeroFuid();
	jobFuid.m_0 = hkTaskType::HKNP_NARROW_PHASE_TASK;
	jobFuid.m_2 = hkUint16((sharedData.m_enableRebuildCdCaches1 != 0) ? 1 : 0);	// check if this is the first or second run
	jobFuid.m_3 = cellIdx;
	jobFuid.m_4 = linkIdx;
	hkCheckDeterminismUtil::registerAndStartJob(jobFuid);
#endif

	// Initialize output jacobian ranges
	hknpConstraintSolverJacobianRange2 movingJacobianRangeOut;
	hknpConstraintSolverJacobianRange2 fixedJacobianRangeOut;
	HK_ON_CPU(hknpLiveJacobianInfoRange liveJacobianInfoRangeOut);
	{
		movingJacobianRangeOut.initRange(hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER,
			hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS |
			hknpConstraintSolverJacobianRange2::SOLVER_TEMPS);
		movingJacobianRangeOut.setStartPoint(&writers.m_movingJacobianWriter);

		fixedJacobianRangeOut.initRange(hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER,
			hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS |
			hknpConstraintSolverJacobianRange2::SOLVER_TEMPS);
		fixedJacobianRangeOut.setStartPoint (&writers.m_fixedJacobianWriter);

		HK_ON_CPU(liveJacobianInfoRangeOut.setStartPoint(&writers.m_liveJacobianInfoWriter));
	}

	// Initialize first cd cache consumer
	hknpCdCacheConsumer cdCacheConsumer;
	HK_ON_SPU(cdCacheConsumer.initSpu(HK_SPU_DMA_GROUP_STALL, 1, "CdCacheConsumer1"));
	{
		SPU_GET(cdCacheRange, &m_cdCacheGridIn.m_entries[linkIdx], hkBlockStreamBase::Range, hkSpuDmaManager::READ_COPY); 

		// Set consumer to range
		cdCacheConsumer.setToRange(
			threadContext.m_heapAllocator, m_cdCacheStreamIn, sharedData.m_cdCacheStreamInOnPpu, &cdCacheRange );

		SPU_RELEASE(cdCacheRange, &m_cdCacheGridIn.m_entries[linkIdx], hkBlockStreamBase::Range);
	}

	// Initialize second cd cache consumer if there is a second input cd cache grid
	hknpCdCacheConsumer cdCacheConsumer2;
	HK_ON_SPU(cdCacheConsumer2.initSpu(HK_SPU_DMA_GROUP_STALL, 1, "CdCacheConsumer2"));
	if (m_cdCacheGridIn2.m_entries)
	{
		SPU_GET(cdCacheRange, &m_cdCacheGridIn2.m_entries[linkIdx], hkBlockStreamBase::Range, hkSpuDmaManager::READ_COPY); 

		// Set consumer to range
		cdCacheConsumer2.setToRange(threadContext.m_heapAllocator, m_cdCacheStreamIn2, sharedData.m_cdCacheStreamIn2OnPpu,
			&cdCacheRange);

		SPU_RELEASE(cdCacheRange, &m_cdCacheGridIn2.m_entries[linkIdx], hkBlockStreamBase::Range);
	}
	else
	{
		cdCacheConsumer2.setEmpty();
	}

	// Process input cd caches
	{
		// Create output ranges (only for top level cd caches which require deterministic sorting)
		hknpCdCacheRange cdCacheRangeOut;
		hknpCdCacheRange inactiveCdCacheRangeOut;
		hknpCdCacheRange crossGridCdCacheRangeOut;
		cdCacheRangeOut.setStartPoint(&writers.m_cdCacheWriter);
		inactiveCdCacheRangeOut.setStartPoint(&writers.m_inactiveCdCacheWriter);
		crossGridCdCacheRangeOut.setStartPoint(&writers.m_crossCdCacheWriter);

		// Process all input cd caches
		threadContext.beginCommands(linkIdx);
		hknpCollidePipeline::mergeAndCollide2Streams(threadContext, sharedData, linkIdx,
			cdCacheConsumer, *m_childCdCacheStreamIn, sharedData.m_childCdCacheStreamInOnPpu,
			&cdCacheConsumer2, m_childCdCacheStreamIn2, sharedData.m_childCdCacheCacheStreamIn2OnPpu,
			writers.m_cdCacheWriter, writers.m_childCdCacheWriter,
			&writers.m_inactiveCdCacheWriter,  &writers.m_inactiveChildCdCacheWriter,
			&writers.m_crossCdCacheWriter, &writers.m_crossGridChildCdCacheWriter,
			writers.m_activePairWriter, &writers.m_liveJacobianInfoWriter,
			&writers.m_movingJacobianSorter, &writers.m_fixedJacobianSorter);
		threadContext.endCommands(linkIdx);

		// Close output ranges
		cdCacheRangeOut.setEndPoint(&writers.m_cdCacheWriter);
		inactiveCdCacheRangeOut.setEndPoint(&writers.m_inactiveCdCacheWriter);
		crossGridCdCacheRangeOut.setEndPoint(&writers.m_crossCdCacheWriter);

		// Put output ranges in output grids
#if !defined(HK_PLATFORM_SPU)
		m_cdCacheGridOut.m_entries[linkIdx] = cdCacheRangeOut;
		m_inactiveCdCacheGridOut .m_entries[linkIdx] = inactiveCdCacheRangeOut;
		m_crossGridCdCacheGridOut.m_entries[linkIdx] = crossGridCdCacheRangeOut;
#else
		
		hkSpuDmaManager::putToMainMemory(&m_cdCacheGridOut.m_entries[linkIdx], &cdCacheRangeOut, sizeof(hknpCdCacheRange),
			hkSpuDmaManager::WRITE_NEW); 
		hkSpuDmaManager::putToMainMemory(&m_inactiveCdCacheGridOut.m_entries[linkIdx], &inactiveCdCacheRangeOut,
			sizeof(hknpCdCacheRange), hkSpuDmaManager::WRITE_NEW); 
		hkSpuDmaManager::putToMainMemory(&m_crossGridCdCacheGridOut.m_entries[linkIdx], &crossGridCdCacheRangeOut,
			sizeof(hknpCdCacheRange), hkSpuDmaManager::WRITE_NEW); 
		hkSpuDmaManager::waitForDmaCompletion(); 
		hkSpuDmaManager::performFinalChecks(&m_cdCacheGridOut.m_entries[linkIdx], &cdCacheRangeOut, sizeof(hknpCdCacheRange));
		hkSpuDmaManager::performFinalChecks(&m_inactiveCdCacheGridOut .m_entries[linkIdx], &inactiveCdCacheRangeOut, sizeof(hknpCdCacheRange));
		hkSpuDmaManager::performFinalChecks(&m_crossGridCdCacheGridOut.m_entries[linkIdx], &crossGridCdCacheRangeOut, sizeof(hknpCdCacheRange));

		// Wait for pending DMAs and release stack allocations for both cache consumers
		cdCacheConsumer2.exitSpu();
		cdCacheConsumer.exitSpu();
#endif
	}

	writers.m_movingJacobianSorter.flush();
	writers.m_fixedJacobianSorter.flush();

	// Close jacobian ranges
	movingJacobianRangeOut.setEndPoint(&writers.m_movingJacobianWriter);
	fixedJacobianRangeOut.setEndPoint(&writers.m_fixedJacobianWriter);
	HK_ON_CPU(liveJacobianInfoRangeOut.setEndPoint(&writers.m_liveJacobianInfoWriter));
	HK_ASSERT(0xf0343454, (cellIdx != HKNP_INVALID_CELL_IDX) || fixedJacobianRangeOut.isEmpty());

	// Put ranges in output grids
	{
		hknpConstraintSolverJacobianGrid* HK_RESTRICT jacMovingGrid = &solverData->m_jacMovingGrid;
		hknpConstraintSolverJacobianGrid* HK_RESTRICT jacFixedGrid = &solverData->m_jacFixedGrid;
	#if defined(HK_PLATFORM_PPU)
		if (m_runOnPpu)
		{
			jacMovingGrid = &solverData->m_jacMovingPpuGrid;
			jacFixedGrid = &solverData->m_jacFixedPpuGrid;
		}
	#endif

		jacMovingGrid->addRange(writers.m_movingJacobianWriter,  linkIdx, movingJacobianRangeOut); 
		if (cellIdx != HKNP_INVALID_CELL_IDX)
		{
			jacFixedGrid->addRange(writers.m_fixedJacobianWriter, cellIdx, fixedJacobianRangeOut);
		}
		HK_ON_CPU(solverData->m_liveJacInfoGrid.addRange(writers.m_liveJacobianInfoWriter,linkIdx, liveJacobianInfoRangeOut));
	}

#if defined(HK_PLATFORM_SPU)
	hkSpuDmaManager::waitForDmaCompletion();
#endif

	HK_ON_DETERMINISM_CHECKS_ENABLED(hkCheckDeterminismUtil::finishJob(jobFuid, false););
	return true;
}

#if defined(HK_PLATFORM_PPU)
void* hknpNarrowPhaseTask::getElf()
{
#if defined(HK_PLATFORM_PS3_PPU)
	extern char _binary_hknpSpursCollide_elf_start[];
	return m_runOnPpu ? HK_INVALID_ELF : _binary_hknpSpursCollide_elf_start;
#else
	return m_runOnPpu ? HK_INVALID_ELF : (void*)hkTaskType::HKNP_NARROW_PHASE_TASK;
#endif
}
#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
