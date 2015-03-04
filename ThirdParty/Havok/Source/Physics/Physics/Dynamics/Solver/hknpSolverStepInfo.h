/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_MANAGER_H
#define HKNP_SOLVER_MANAGER_H

#include <Physics/Physics/hknpConfig.h>

#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>

struct hknpConstraintSolverJacobianRange2;
class hknpSimulationContext;
class hkSimpleSchedulerTaskBuilder;
class hknpSolverSumVelocity;
class hknpSolverVelocity;
class hknpLiveJacobianInfoRange;
class hkIntSpaceUtil;
struct hknpSolverSimpleSchedulerTask;
struct hknpIdxRange;
class hknpSolverData;
class hknpDeactivationStepInfo;
class hknpMotionProperties;


///
class hknpSolverStepInfo
{
	public:

		//+hk.MemoryTracker(ignore=True)

		enum DmaChannels
		{
			DMA_ID_SPLITTER     = 4,
			DMA_ID_MOTION_IDS   = 5,
			DMA_ID_SOLVER_STATE = 6,
		};

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSolverStepInfo );

	public:

		// mgmt stuff
		hkPadSpu<hkBlockStreamAllocator*>		m_tempAllocator;
		hkPadSpu<hkBlockStreamAllocator*>		m_heapAllocator;

		hknpSimulationContext*					m_simulationContext;	// only for PPU
		hknpSolverData*							m_solverData;			// only for PPU, needed for solver parallel tasks

	#if defined(HKNP_ENABLE_LOCKLESS_SOLVER)
		// jobs data
		//	new scheduler
		hkPadSpu<hkSimpleSchedulerTaskBuilder*>	 m_taskList;
	#endif

		// const solver setup data
		hkPadSpu<const hknpSolverInfo*>			m_solverInfo;
		hkPadSpu<const hknpSpaceSplitter*>		m_spaceSplitter;
		hkPadSpu<hkUint32>						m_spaceSplitterSize;
		hkPadSpu<const hkIntSpaceUtil*>			m_intSpaceUtil;
		hkPadSpu<hknpIdxRange*>					m_motionGridEntriesStart;

		// general working data
		hkPadSpu<hknpBody*>						m_bodies;
		hkPadSpu<hknpMotion*>					m_motions;
		hkPadSpu<const hknpBodyQuality*>		m_qualities;
		hkPadSpu<hknpSolverVelocity*>			m_solverVelocities;

		// specific data for hknpStepConstraintJacobianProcess
		hkPadSpu<hknpConstraintSolverJacobianRange2*>	m_jacGridEntries[hknpJacobianGridType::NUM_TYPES];
		hkPadSpu<hknpLiveJacobianInfoRange*>			m_liveJacInfoGridEntries;

		// specific data for hknpSubIntegrateProcess
		hkPadSpu<hknpSolverSumVelocity*>		m_solverSumVelocities;
		hkPadSpu<hknpDeactivationState*>		m_deactivationStates;
		hkPadSpu<const hknpMotionProperties*>	m_motionProperties;
		hkPadSpu<hkUint32>						m_numMotionProperties;

		// specific data for hknpUpdateBodiesProcess
		hkPadSpu<const hknpBodyId*>				m_dynamicBodyIds;
		hkPadSpu<hkUint32>						m_numDynamicBodyIds;
		hkPadSpu<hkUint32*>						m_numBodyChunksUpdated;

		HK_ALIGN16(hkUint32						m_numBodyChunksUpdatedVar);	// a shared state variable (on PPU), pointed to by m_numBodyChunksUpdated
};


void HK_CALL hknpStepConstraintJacobianProcess(
	const hknpSimulationThreadContext& tl, int iStep, int iMicroStep, int threadIdx,
	hkUint8 grid, hkUint8 gridEntryIndex, hknpCellIndex cellA, hknpCellIndex cellB, hknpSolverStepInfo* solverStepInfo );

void HK_CALL hknpSubIntegrateProcess(
	const hknpSimulationThreadContext& tl, int iStep, int iMicroStep, int threadIdx,
	hknpCellIndex motionGridIdx, hknpSolverStepInfo* solverStepInfo );

void HK_CALL hknpUpdateBodiesProcess(
	const hknpSimulationThreadContext& tl, hknpSolverStepInfo* solverStepInfo );

void HK_CALL hknpAddActiveBodyPairsProcess(
	const hknpSolverData* solverData, hknpDeactivationStepInfo* deactivationStepInfo,
	hknpIdxRangeGrid* m_cellIdxToGlobalSolverId );

#if HKNP_ENABLE_SOLVER_PARALLEL_TASKS == 1
void HK_CALL hknpGarbageCollectInactiveCachesProcess( const hknpSimulationThreadContext& tl, hknpCellIndex cellIdx );
#endif


#endif // HKNP_SOLVER_MANAGER_H

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
