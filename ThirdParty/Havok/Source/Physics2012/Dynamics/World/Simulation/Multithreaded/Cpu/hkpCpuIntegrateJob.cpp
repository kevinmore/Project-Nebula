/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/PointerMap/hkMap.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>

#include <Physics2012/Dynamics/Action/hkpAction.h>
#include <Physics2012/Dynamics/Entity/Util/hkpEntityAabbUtil.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>
#include <Physics2012/Dynamics/Motion/Util/hkpRigidMotionUtil.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuIntegrateJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Cpu/hkpCpuSplitSimulationIslandJob.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.h>
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpMultithreadedSimulation.h>
#include <Physics2012/Dynamics/World/Util/hkpWorldOperationUtil.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Collide/hkpResponseModifier.h>


namespace hkBuildConstraintBatches
{
	typedef hkUint8 BatchIndex;
#define INVALID_BODY       hkObjectIndex(-1)
#define FIXED_BODY_INDEX   hkObjectIndex(-1)
#define INVALID_BATCH      hkBuildConstraintBatches::BatchIndex(-1)


	struct Constraint
	{
		hkConstraintInternal* m_internal;
		hkObjectIndex	m_bodyA;
		hkObjectIndex	m_bodyB;
		BatchIndex		m_batchIndex;
	};

	struct Body
	{
		BatchIndex		m_batchIndex;
	};

	struct Header
	{
		Header(): m_numConstraints(0){}

		hkBool      m_startNewBatch;
		int         m_numConstraints;
		Constraint *m_firstConstraint;
		Constraint *m_currentConstraint;
	};
}



static HK_LOCAL_INLINE hkBool canSolveOnSingleProcessor(const hkpBuildJacobianTaskHeader& taskHeader)
{
#if defined (HK_PLATFORM_HAS_SPU)
	return ( calcStackMemNeededForSolveOnSpu(taskHeader) < HK_SPU_TOTAL_PHYSICS_BUFFER_SIZE );
#else
	// We force solving of large islands on multiple spu's for non-PlayStation(R)3 platforms.
	return false;
#endif
}


namespace
{
	struct hkConstraintInternalInfo
	{
		HK_ALIGN(const hkConstraintInternal* m_internal,8);
		hkObjectIndex m_entityAIndex;
		hkObjectIndex m_entityBIndex;
	};


	/// To allow callback constraints to adjust their corresponding atomInfo in the
	/// buildJacobianTasks, we provide them with a pointer. This is only used when 
	/// a contactPointCallback adds a hkpResponseModifier.
	struct hkConstraintInternalCallbackInfo : public hkConstraintInternalInfo
	{
		hkpBuildJacobianTask::AtomInfo* m_atomInfo;
	};
}

	/// \param callbackConstraints are collected in this function so they can have a pointer to their corresponding atomInfo.
static HK_FORCE_INLINE void HK_CALL createAndAppendNewTask(const hkpBuildJacobianTaskHeader* taskHeader,
														   hkpBuildJacobianTaskCollection* outBuildCollection, hkpSolveJacobiansTaskCollection* outSolveCollection,
														   hkpBuildJacobianTask*& prevBuildTask, hkpSolveConstraintBatchTask*& prevSolveTask,
														   hkpSolveConstraintBatchTask*& firstTaskOfPreviousBatch, hkpSolveConstraintBatchTask*& firstTaskOfCurrentBatch,
														   hkpJacobianSchema*& schemas, hkpSolverElemTemp*& elemTemp,
#if defined(HK_PLATFORM_HAS_SPU)
														   hkLocalArray<hkObjectIndex>& accumulatorIndicesForThisTask,
														   int* entityAccumulatorInterIndexForThisTask,
#endif
														   hkLocalArray<hkConstraintInternalInfo>& constraintsForThisTask,
														   int& batchIndexOfLastTask, int currentBatchIndex, hkUint16*& ptrToBatchSize,
														   hkBool putOnPpuOnlyBuildJacobianTaskList, hkBool generateSolveTasks,
														   hkArray<hkConstraintInternalCallbackInfo>& callbackConstraints)
{
	hkpJacobianSchema* const schemaStart = schemas;
	hkpSolverElemTemp* const elemTempStart = elemTemp;

	{
// Disable warning on SNC
#if (defined HK_COMPILER_SNC)
#	pragma diag_push
#	pragma diag_suppress=1646
#endif
		hkpBuildJacobianTask* buildTask = new hkpBuildJacobianTask;
#if (defined HK_COMPILER_SNC)
#	pragma diag_pop
#endif
		if (!prevBuildTask)
		{
			if (putOnPpuOnlyBuildJacobianTaskList)
			{
#if defined(HK_PLATFORM_HAS_SPU)
				outBuildCollection->m_ppuOnlyBuildJacobianTasks = buildTask;
				outBuildCollection->m_numPpuOnlyBuildJacobianTasks = 0; // increased after the following block
#else
				HK_ASSERT2(0XAD5432AD, false, "Illegal path for platforms without Spu's");
#endif
			}
			else
			{
				outBuildCollection->m_buildJacobianTasks = buildTask;
				outBuildCollection->m_numBuildJacobianTasks = 0; // increased after the following block
			}
		}
		else
		{
			// Make sure this is not a previous task from ppu-only list -- if now we're processing spu-ok ones. !!! -- keep this in synch with Ln417 (case  0: prevBuildTask = HK_NULL; sourceConstraints = &normalPriorityConstraints; break;)
			prevBuildTask->m_next = buildTask;
			prevBuildTask->m_schemasOfNextTask = schemas;
		}

#if defined(HK_PLATFORM_HAS_SPU)
		(putOnPpuOnlyBuildJacobianTaskList ? outBuildCollection->m_numPpuOnlyBuildJacobianTasks : outBuildCollection->m_numBuildJacobianTasks)++;
		buildTask->m_onPpuOnly = putOnPpuOnlyBuildJacobianTaskList;
#else
		outBuildCollection->m_numBuildJacobianTasks++;
#endif

		//
		//

		buildTask->m_taskHeader = const_cast<hkpBuildJacobianTaskHeader*>(taskHeader);
		buildTask->m_numAtomInfos = constraintsForThisTask.getSize();
		buildTask->m_accumulators = taskHeader->m_accumulatorsBase;
		buildTask->m_schemas = schemaStart;

			//
			// Fill atom infos
			//
		int numConstraints = constraintsForThisTask.getSize();
#if defined(HK_PLATFORM_PS3_PPU)
		int numConstraintsMinus4 = numConstraints-4;
#endif
		for (int c = 0; c < numConstraints; c++)
		{
#if defined(HK_PLATFORM_PS3_PPU)
			if ( c < numConstraintsMinus4 )
			{
				hkMath::prefetch128( constraintsForThisTask[c+4].m_internal );
			}
#endif

			hkConstraintInternalInfo& info = constraintsForThisTask[c];
			const hkConstraintInternal& internal = *info.m_internal;
			hkpBuildJacobianTask::AtomInfo& atomInfo = buildTask->m_atomInfos[c];

			// collect constraints that need to fire a callback
			if ( internal.m_callbackRequest & ( hkpConstraintAtom::CALLBACK_REQUEST_SETUP_PPU_ONLY | hkpConstraintAtom::CALLBACK_REQUEST_NEW_CONTACT_POINT | hkpConstraintAtom::CALLBACK_REQUEST_CONTACT_POINT_CALLBACK ) ) 
			{
				// We fire callbacks here for new contact points because they need the simple collision response.
				hkConstraintInternalCallbackInfo *const newCallbackConstraint = callbackConstraints.expandByUnchecked( 1 );
				*(static_cast<hkConstraintInternalInfo*>( newCallbackConstraint )) = info;
				newCallbackConstraint->m_atomInfo = &atomInfo;				
			}

			atomInfo.m_atoms				=  internal.getAtoms();
			atomInfo.m_atomsSize			=  internal.getAtomsSize();
			atomInfo.m_instance				=  internal.m_constraint;
			atomInfo.m_runtime				=  internal.m_runtime;
			atomInfo.m_runtimeSize			=  internal.m_runtimeSize;
			int indexA = info.m_entityAIndex;
			int indexB = info.m_entityBIndex;
			atomInfo.m_accumulatorIndexA	=  hkObjectIndex(indexA+1);	// this handles the fixed accumulator as well
			atomInfo.m_accumulatorIndexB	=  hkObjectIndex(indexB+1);
	#		if defined(HK_PLATFORM_HAS_SPU)
			if (taskHeader->m_solveInSingleThread)
			{
				atomInfo.m_accumulatorInterIndexA = atomInfo.m_accumulatorIndexA;
				atomInfo.m_accumulatorInterIndexB = atomInfo.m_accumulatorIndexB;
			}
			else
			{
				atomInfo.m_accumulatorInterIndexA = hkObjectIndex( indexA == HK_INVALID_OBJECT_INDEX ? 0 : entityAccumulatorInterIndexForThisTask[indexA] );
				atomInfo.m_accumulatorInterIndexB = hkObjectIndex( indexB == HK_INVALID_OBJECT_INDEX ? 0 : entityAccumulatorInterIndexForThisTask[indexB] );
				HK_ASSERT2(0xad674332, unsigned(atomInfo.m_accumulatorInterIndexA) < hkpSolveConstraintBatchTask::MAX_NUM_ACCUMULATORS_PER_TASK, "Invalid accumulator index.");
				HK_ASSERT2(0xad674332, unsigned(atomInfo.m_accumulatorInterIndexB) < hkpSolveConstraintBatchTask::MAX_NUM_ACCUMULATORS_PER_TASK, "Invalid accumulator index.");
			}
	#		endif
			atomInfo.m_transformA			= &internal.m_entities[0]->getMotion()->getTransform();
			atomInfo.m_transformB			= &internal.m_entities[1]->getMotion()->getTransform();

			schemas   = hkAddByteOffset(schemas, internal.m_sizeOfSchemas);
			elemTemp += internal.m_numSolverElemTemps;

			HK_ASSERT2(0XADBC87DA, 0 == (hkUlong(schemas) & (HK_REAL_ALIGNMENT-1)), "Schemas not SIMD aligned ?? !!");
		}
		constraintsForThisTask.clear();

		schemas  = hkAddByteOffset(schemas, hkpJacobianSchemaInfo::End::Sizeof);

		buildTask->m_schemasOfNextTask = schemas;

		// round up only after the size of elemTemps is calculated
		elemTemp = reinterpret_cast<hkpSolverElemTemp*>( HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, hkUlong(elemTemp)) );

		prevBuildTask = buildTask;
	}

	//
	// Handle solve tasks
	//
	if (generateSolveTasks)
	{
		hkpSolveConstraintBatchTask* solveTask = new hkpSolveConstraintBatchTask();

		if (prevSolveTask)
		{
			prevSolveTask->m_next = solveTask;
		}

		if (currentBatchIndex != batchIndexOfLastTask)
		{
			HK_ASSERT2(0xad876543, currentBatchIndex == batchIndexOfLastTask + 1, "This cannot happen.");
			if (currentBatchIndex == 0)
			{
				outSolveCollection->m_firstSolveJacobiansTask = solveTask;
			}

			batchIndexOfLastTask = currentBatchIndex;

			if (prevSolveTask)
			{
				HK_ASSERT2(0xad8757dd, firstTaskOfCurrentBatch, "Internal error: firstTaskOfCurrentBatch and prevSolveTask inconsistent.");
				hkpSolveConstraintBatchTask* aTaskFromPreviousBatch = firstTaskOfPreviousBatch;
				hkUint16 batchSize = (*ptrToBatchSize);
				while(aTaskFromPreviousBatch && aTaskFromPreviousBatch != firstTaskOfCurrentBatch)
				{
					aTaskFromPreviousBatch->m_sizeOfNextBatch = batchSize;
					aTaskFromPreviousBatch->m_firstTaskInNextBatch = firstTaskOfCurrentBatch;
					aTaskFromPreviousBatch = aTaskFromPreviousBatch->m_next;
				}

				// Keep order! this is done after the above!
				prevSolveTask->m_isLastTaskInBatch = true;
				ptrToBatchSize = &prevSolveTask->m_sizeOfNextBatch;

			}

			firstTaskOfPreviousBatch = firstTaskOfCurrentBatch;
			firstTaskOfCurrentBatch = solveTask;
		}

		(*ptrToBatchSize)++;

		solveTask->m_taskHeader = const_cast<hkpBuildJacobianTaskHeader*>(taskHeader);
#	if defined(HK_PLATFORM_HAS_SPU)
		HK_ASSERT2(0xad6532a1, accumulatorIndicesForThisTask.getSize() <= hkpSolveConstraintBatchTask::MAX_NUM_ACCUMULATORS_PER_TASK, "Too many accumualtors in a Task.");
		//todo: memcpy16 instead
		HK_ASSERT2(0xad6532a2, accumulatorIndicesForThisTask[0] == 0, "The first accumulator offset must be zero (for fixed bodies).");
		for (int a = 0; a < accumulatorIndicesForThisTask.getSize(); a++)
		{
			solveTask->m_accumulatorInterIndices[a] = accumulatorIndicesForThisTask[a];
		}
		solveTask->m_numOfAccumulatorInterIndices = accumulatorIndicesForThisTask.getSize();
		accumulatorIndicesForThisTask.clear();
		accumulatorIndicesForThisTask.pushBackUnchecked(0);
#	endif
		solveTask->m_accumulators = taskHeader->m_accumulatorsBase;
		solveTask->m_schemas = schemaStart;
		HK_ASSERT2(0xad7854bd, 0 == (hkUlong(elemTempStart) & (HK_REAL_ALIGNMENT-1)), "SolverElemTemp buffer must be SIMD aligned.");
		solveTask->m_solverElemTemp = elemTempStart;
		solveTask->m_sizeOfSchemaBuffer = hkGetByteOffsetInt(schemaStart, schemas);
		solveTask->m_sizeOfSolverElemTempBuffer = hkGetByteOffsetInt(elemTempStart, elemTemp);

		prevSolveTask = solveTask;
	}
#	if defined(HK_PLATFORM_HAS_SPU)
	else
	{
		// if not building solver tasks we still need to clear some arrays
		accumulatorIndicesForThisTask.clear();
		accumulatorIndicesForThisTask.pushBackUnchecked(0);
	}
#	endif
}

	// check whether solver data is in the same cache line as m_constraintMaster
//HK_COMPILE_TIME_ASSERT( (HK_OFFSET_OF(hkpEntity,m_constraintsMaster) & 0xff80) == (HK_OFFSET_OF(hkpEntity,m_solverData) & 0xff80) );

class constraintInfoLess
{
public:

	HK_FORCE_INLINE hkBool32 operator() ( const hkConstraintInternalInfo& a, const hkConstraintInternalInfo& b )
	{
		if (a.m_internal->m_constraintType != hkpConstraintInstance::TYPE_CHAIN)  return false;

		// A is chain
		if (b.m_internal->m_constraintType != hkpConstraintInstance::TYPE_CHAIN)  return true; // place chains before other constraints

		// Both are chains
		hkpConstraintChainInstance* ccA = static_cast<hkpConstraintChainInstance*>(a.m_internal->m_constraint);
		hkpConstraintChainInstance* ccB = static_cast<hkpConstraintChainInstance*>(b.m_internal->m_constraint);

		return (ccA->m_chainConnectedness > ccB->m_chainConnectedness); // place chains with higher connectedness first
	}
};

	/// This creates a list of (jacobian-building) tasks. Those tasks are referenced by the taskHeader
static /*HK_FORCE_INLINE*/ void HK_CALL createBuildJacobianAndSolveJacobianTaskCollection( hkpSimulationIsland* island, hkBool forceCoherentConstraintOrderingInSolver, hkpBuildJacobianTaskHeader* taskHeader, hkpBuildJacobianTaskCollection* outBuildCollection, hkpSolveJacobiansTaskCollection* outSolveCollection )
{
	HK_TIME_CODE_BLOCK("BuildJacTask", HK_NULL);

	//
	//
	//

	hkpEntity*const*const bodies = island->getEntities().begin();
	int numBodies = island->getEntities().getSize();
	hkpEntity*const*const bodiesEnd = bodies + numBodies;


	//
	// Put all constraint internals onto those two lists
	//
	// Also gather constraints requiring special processing
	//

	hkArray<hkConstraintInternalInfo>::Temp normalPriorityConstraints; normalPriorityConstraints.reserve(island->m_numConstraints);
	hkArray<hkConstraintInternalInfo>::Temp highPriorityConstraints; highPriorityConstraints.reserve(island->m_numConstraints);
	hkLocalArray<hkConstraintInternalCallbackInfo> callbackConstraints(island->m_numConstraints);
#	if defined(HK_PLATFORM_HAS_SPU)
	hkArray<hkConstraintInternalInfo>::Temp ppuSetupOnlyConstraints; ppuSetupOnlyConstraints.reserve(island->m_numConstraints);
#	endif
	int numChainConstraints = 0;

	hkUlong* pointerBuffer = 0;
	int po2 = 64; 
	{
		while ( po2 < numBodies ){ po2 *= 2; }	// power of 2 which is bigger than numEntities
		po2 *= 4;	// 2* for values 2* to keep hash entries small
		pointerBuffer = hkMemTempBlockAlloc<hkUlong>(po2);
	}
	hkMap<hkUlong>* entityPtrToIndex = new hkMap<hkUlong>(pointerBuffer, po2 * sizeof(hkUlong) );

	{
		//HK_TIMER_BEGIN("Collect Constraints", HK_NULL );

		// put all entities into a hash table
		{
			//HK_TIMER_BEGIN("CollectEntities", HK_NULL );
			{
				for (int bi = 0; bi < numBodies; bi++)
				{
					HK_ASSERT2(0xad875def, bodies[bi]->m_storageIndex == bi, "Body storage index corrupted?");
					entityPtrToIndex->insert(hkUlong(bodies[bi]), hkUlong(bi));
				}
			}
			//HK_TIMER_END();
			//HK_TIMER_BEGIN("CollectConstraints", HK_NULL );
			hkpEntity*const* b = bodies;
			hkUint32 accumOffset = sizeof(hkpVelocityAccumulator); // skip the fixed body
			int bi;
			for (b = bodies, bi = 0; b < bodiesEnd; b++, bi++)
			{
				hkpEntity* body = *b;

#if defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_XBOX360)
				if ( b+4 < bodiesEnd )
				{
					const void* p = b[2]->getConstraintMasters().begin();
					int size = b[2]->getConstraintMasters().getSize() * sizeof(hkConstraintInternal);
					do
					{
						hkMath::forcePrefetch<256>( p );
						p = hkAddByteOffsetConst(p,256);
						size -= 256;
					} while ( size > 0 );
					hkMath::prefetch128( &(b[4]->m_solverData ) );
				}
#endif

				//
				// Initialize motions' solverData before there are used below
				//
				{
					// try to avoid making the cacheline dirty
					if ( accumOffset != body->m_solverData)
					{
						body->m_solverData = accumOffset;
					}
					HK_ASSERT2(0xad6754dd, accumOffset / sizeof(hkpVelocityAccumulator) <= 0xffff, "Accumulator index to large.");
					accumOffset += sizeof(hkpVelocityAccumulator);
				}

				const hkConstraintInternal* c    = body->getConstraintMasters().begin();
				const hkConstraintInternal* cEnd = body->getConstraintMasters().end();

				for ( ;c < cEnd; c++ )
				{
					hkConstraintInternalInfo info;
					{
						info.m_internal = c;

						hkObjectIndex* entityIndices = &info.m_entityAIndex;
						HK_ASSERT2(0xad7655dd, &info.m_entityBIndex == &entityIndices[1], "Internal error.");
						HK_ASSERT2(0xad875ddd, c->getMasterEntity()->m_storageIndex == bi, "Body storage index corrupted?");

						const int whoIsSlave = 1 - c->m_whoIsMaster;

						entityIndices[c->m_whoIsMaster] = hkObjectIndex(bi);
						entityIndices[whoIsSlave]       = hkObjectIndex(entityPtrToIndex->getWithDefault( hkUlong(c->m_entities[whoIsSlave]), HK_INVALID_OBJECT_INDEX ));
					}
					HK_COMPILE_TIME_ASSERT( hkpConstraintInstance::TYPE_NORMAL==0 && hkpConstraintInstance::TYPE_CHAIN==1 );
					numChainConstraints += c->m_constraintType;

#			if defined(HK_PLATFORM_HAS_SPU)
					// collect constraints that only can be setup on PPU in a separate list
					if ( c->m_callbackRequest & hkpConstraintAtom::CALLBACK_REQUEST_SETUP_PPU_ONLY )
					{
						// we will deliberately ignore increasing the schemas and jacobians pointers;
						// this will be done in hkSetSchemasPtrInTasks() once the ppu-only constraints have been processed
						ppuSetupOnlyConstraints.pushBackUnchecked(info);
					}
					else
#			endif
						if (c->m_priority < hkpConstraintInstance::PRIORITY_TOI_HIGHER)
						{
							normalPriorityConstraints.pushBackUnchecked(info);
						}
						else
						{
							highPriorityConstraints.pushBackUnchecked(info);
						}
				}
			}
		}
	//	HK_TIMER_END();
	}


#if HKP_MAX_NUMBER_OF_CHAIN_CONSTRAINTS_IN_ISLAND_WHEN_MULTITHREADING != 0
	HK_ASSERT2(0xad438211, (numChainConstraints <= HKP_MAX_NUMBER_OF_CHAIN_CONSTRAINTS_IN_ISLAND_WHEN_MULTITHREADING), "Too many chain constraints in the island. Change HKP_MAX_NUMBER_OF_CHAIN_CONSTRAINTS_IN_ISLAND_WHEN_MULTITHREADING.");
	if (numChainConstraints>0)
	{
		// Sort constraints to process highly connected chains first.
		hkSort(normalPriorityConstraints.begin(), normalPriorityConstraints.getSize(), constraintInfoLess());
		hkSort(highPriorityConstraints.begin(),   highPriorityConstraints.getSize(),   constraintInfoLess());
		numChainConstraints = 0;			// no special handling of chain constraints
	}
#endif

	HK_MONITOR_ADD_VALUE( "NumEntities",   float(numBodies), HK_MONITOR_TYPE_INT );
	HK_MONITOR_ADD_VALUE( "NumConstraints",float(island->m_numConstraints), HK_MONITOR_TYPE_INT );
	HK_MONITOR_ADD_VALUE( "NumJacobians",  float(island->m_constraintInfo.m_numSolverResults), HK_MONITOR_TYPE_INT );

	//
	// Decide on using single- or multi-threaded solving
	//
	taskHeader->m_solveInSingleThreadOnPpuOnly = (0 !=numChainConstraints );
	taskHeader->m_solveInSingleThread = taskHeader->m_solveInSingleThreadOnPpuOnly || canSolveOnSingleProcessor(*taskHeader);
	const bool generateSolveTasks = ! taskHeader->m_solveInSingleThread;

	//
	// Stream pointers
	//
	hkpJacobianSchema* schemas = taskHeader->m_schemasBase;
	hkpSolverElemTemp* elemTemp = taskHeader->m_solverTempBase;

	int currentBatchIndex    = -1;
	int batchIndexOfLastTask = -1;

	hkUint16* ptrToBatchSize = &outSolveCollection->m_firstBatchSize;
	*ptrToBatchSize = 0;

	//
	// Proceed with on-spu constraints
	//
	hkpBuildJacobianTask* prevBuildTask = HK_NULL;
	hkpSolveConstraintBatchTask* prevSolveTask = HK_NULL;

	hkpSolveConstraintBatchTask* firstTaskOfPreviousBatch = HK_NULL;
	hkpSolveConstraintBatchTask* firstTaskOfCurrentBatch = HK_NULL;

	// List of constraint internals for the next task.
	hkLocalArray<hkConstraintInternalInfo> constraintsForNextTask(hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS);

#	if defined(HK_PLATFORM_HAS_SPU)
	// Intermediate array storing accumulator indices
	int accIndicesSize = hkpSolveConstraintBatchTask::MAX_NUM_ACCUMULATORS_PER_TASK; 
	if (!generateSolveTasks)
	{	// MAX_NUM_ACCUMULATORS_PER_TASK is ok for when generateSolverTasks is true. Otherwise se must make sure we won't overflow.
		accIndicesSize = hkMath::max2(accIndicesSize, hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS * 2 + 1);
	}
	hkLocalArray<hkObjectIndex> accumulatorIndices(accIndicesSize); 
	accumulatorIndices.pushBackUnchecked(0); 
#	endif

	// Mapping: Entity storage index -> task group
	int* entityGroup = hkMemTempBlockAlloc<int>(numBodies);
#	if defined(HK_PLATFORM_HAS_SPU)
	int* entityAccumulatorInterIndex = hkMemTempBlockAlloc<int>(numBodies);
#	endif
	bool flushConstraintsForNextTask = false;
#	if defined(HK_PLATFORM_HAS_SPU)
	for(int il = -1; il < 2; il++)
#	else
	for(int il =  0; il < 2; il++)
#	endif
	{
		hkArray<hkConstraintInternalInfo>::Temp* sourceConstraints;
		hkBool putOnPpuOnlyBuildJacobianTaskList = false;
		switch(il)
		{
#		if defined(HK_PLATFORM_HAS_SPU)
			case -1:  sourceConstraints = &ppuSetupOnlyConstraints;   putOnPpuOnlyBuildJacobianTaskList = true; break;
#		endif
			case  0:  sourceConstraints = &normalPriorityConstraints; prevBuildTask = HK_NULL;  break;
			default:  sourceConstraints = &highPriorityConstraints;   break;
		}

		// while constraints on the list do:
		while ( sourceConstraints->getSize() )
		{
			// Zero entire group array.
			for(int i = 0; i < numBodies; entityGroup[i++] = 0) { };

			int currentGroup = 1;

			// create new batch
			currentBatchIndex++;
			HK_ASSERT2(0xad78bcd1, constraintsForNextTask.isEmpty(), "Previous task must be saved before starting a new batch.");
#			if defined(HK_PLATFORM_HAS_SPU)
			HK_ASSERT2(0xad78bcd2, accumulatorIndices.getSize() == 1, "Previous accumulators arrray must be saved before starting a new batch (size must be 1 (the shared fixed accumulator)).");
			HK_ON_DEBUG( for(int i = 0; i < numBodies; i++) { entityAccumulatorInterIndex[i] = -1; } );
#			endif

			// iterate through constraints and create tasks for this batch
			hkConstraintInternalInfo* c = sourceConstraints->begin();
			hkConstraintInternalInfo* const cEnd = sourceConstraints->end();
			// left out constraints:
			hkConstraintInternalInfo* nextConstraintLeftOut = sourceConstraints->begin();
			for ( ; c < cEnd; c++)
			{
				const hkConstraintInternalInfo& info = *c;

				// check groups of entities
#				if defined(HK_PLATFORM_HAS_SPU)
				HK_ON_DEBUG( hkpEntity* entityA = info.m_internal->m_entities[0] );
				HK_ON_DEBUG( hkpEntity* entityB = info.m_internal->m_entities[1] );
#				endif
				int fixedEntityGroup = 0;
				int& entityAGroup = (info.m_entityAIndex == HK_INVALID_OBJECT_INDEX) ? fixedEntityGroup : entityGroup[info.m_entityAIndex];
				int& entityBGroup = (info.m_entityBIndex == HK_INVALID_OBJECT_INDEX) ? fixedEntityGroup : entityGroup[info.m_entityBIndex];

				// check if they require adding new accumulators ?
#				if defined(HK_PLATFORM_HAS_SPU)
				const int accumulatorAUntouched = (entityAGroup == 0) && (info.m_entityAIndex != HK_INVALID_OBJECT_INDEX);
				const int accumulatorBUntouched = (entityBGroup == 0) && (info.m_entityBIndex != HK_INVALID_OBJECT_INDEX);
#				endif


				// check current size of tasks: numConstraints, numAccumulators
				HK_ASSERT2(0xad765bcd, constraintsForNextTask.getSize() <= hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS, "This couldn't happen.");
				if (  constraintsForNextTask.getSize() == hkpBuildJacobianTask::MAX_NUM_ATOM_INFOS || flushConstraintsForNextTask
#					if defined(HK_PLATFORM_HAS_SPU)
					|| (generateSolveTasks * (accumulatorIndices.getSize() + accumulatorAUntouched + accumulatorBUntouched) > hkpSolveConstraintBatchTask::MAX_NUM_ACCUMULATORS_PER_TASK)
#					endif
					)
				{
					// create new task.
					createAndAppendNewTask(taskHeader,
										   outBuildCollection, outSolveCollection,
										   prevBuildTask, prevSolveTask,
										   firstTaskOfPreviousBatch, firstTaskOfCurrentBatch,
										   schemas, elemTemp,
#					if defined(HK_PLATFORM_HAS_SPU)
										   accumulatorIndices,
										   entityAccumulatorInterIndex,
#					endif
										   constraintsForNextTask,
										   batchIndexOfLastTask, currentBatchIndex, ptrToBatchSize,
										   putOnPpuOnlyBuildJacobianTaskList, generateSolveTasks,
										   callbackConstraints );
					// we don't distinguish between groups when solving in a single thread
					currentGroup += generateSolveTasks;
					flushConstraintsForNextTask = false;
				}

				if (forceCoherentConstraintOrderingInSolver && currentGroup > 1)
				{
					// This code allows only one task per solve batch. This results in the single-threaded solving behavior.

					// Push all remaining constraints onto the leftOut list at once
					for ( ; c < cEnd; c++)
					{
						*(nextConstraintLeftOut++) = *c;
					}
					break;
				}
				else
				{

					bool allEntitiesOkForCurrentGroup = true;

#				if !defined(HK_PLATFORM_HAS_SPU)
					if (info.m_internal->m_constraintType == hkpConstraintInstance::TYPE_CHAIN )
					{
						hkpConstraintChainInstance* instance = static_cast<hkpConstraintChainInstance*>(info.m_internal->m_constraint);
						for (int i = 0; i < instance->m_chainedEntities.getSize(); i++)
						{
							hkObjectIndex entityIdx = hkObjectIndex(entityPtrToIndex->getWithDefault( hkUlong(instance->m_chainedEntities[i]), HK_INVALID_OBJECT_INDEX ));
							int thisEntityGroup = (entityIdx == HK_INVALID_OBJECT_INDEX) ? fixedEntityGroup : entityGroup[entityIdx];
							if ( thisEntityGroup != 0 && thisEntityGroup != currentGroup )
							{
								allEntitiesOkForCurrentGroup = false;
								break;
							}
						}
					}
					else
					{
						allEntitiesOkForCurrentGroup = ( entityAGroup == 0 || entityAGroup == currentGroup)
							&& ( entityBGroup == 0 || entityBGroup == currentGroup);
					}
#				else
					allEntitiesOkForCurrentGroup = ( entityAGroup == 0 || entityAGroup == currentGroup)
												&& ( entityBGroup == 0 || entityBGroup == currentGroup);

#				endif

					// Decide whether this internal can be put into the current task:
					// Check if the entities' groups are ok
					if ( !allEntitiesOkForCurrentGroup )
					{
						*(nextConstraintLeftOut++) = *c;
					}
					else
					{
						constraintsForNextTask.pushBackUnchecked(*c);
#					if !defined(HK_PLATFORM_HAS_SPU)
						if (info.m_internal->m_constraintType == hkpConstraintInstance::TYPE_CHAIN)
						{
							hkpConstraintChainInstance* instance = static_cast<hkpConstraintChainInstance*>(info.m_internal->m_constraint);
							for (int i = 0; i < instance->m_chainedEntities.getSize(); i++)
							{
								hkObjectIndex entityIdx = hkObjectIndex(entityPtrToIndex->getWithDefault( hkUlong(instance->m_chainedEntities[i]), HK_INVALID_OBJECT_INDEX ));
								int& thisEntityGroup = (entityIdx == HK_INVALID_OBJECT_INDEX) ? fixedEntityGroup : entityGroup[entityIdx];
								thisEntityGroup = currentGroup;
							}
							flushConstraintsForNextTask = generateSolveTasks;
						}
						else
						{
							entityAGroup = currentGroup;
							entityBGroup = currentGroup;
						}
#					else
						entityAGroup = currentGroup;
						entityBGroup = currentGroup;

						if (accumulatorAUntouched)
						{
							entityAccumulatorInterIndex[info.m_entityAIndex] = accumulatorIndices.getSize();
							HK_ASSERT2(0xad8758dd, HK_ACCUMULATOR_OFFSET_TO_INDEX(entityA->m_solverData) == info.m_entityAIndex+1, "Entity's solver data corrupted.");
							accumulatorIndices.pushBackUnchecked( info.m_entityAIndex+1 );
						}
						if (accumulatorBUntouched)
						{
							entityAccumulatorInterIndex[info.m_entityBIndex] = accumulatorIndices.getSize();
							HK_ASSERT2(0xad8755de, HK_ACCUMULATOR_OFFSET_TO_INDEX(entityB->m_solverData) == info.m_entityBIndex+1, "Entity's solver data corrupted.");
							accumulatorIndices.pushBackUnchecked( info.m_entityBIndex+1 );
						}
#					endif
					}
				}
			}


			// force saving whatever task is there (if there are any constraints to be saved.)
			if (constraintsForNextTask.getSize())
			{
				createAndAppendNewTask(taskHeader,
									outBuildCollection, outSolveCollection,
									prevBuildTask, prevSolveTask,
									firstTaskOfPreviousBatch, firstTaskOfCurrentBatch,
									schemas, elemTemp,
#				if defined(HK_PLATFORM_HAS_SPU)
									accumulatorIndices,
									entityAccumulatorInterIndex,
#				endif
									constraintsForNextTask,
									batchIndexOfLastTask, currentBatchIndex, ptrToBatchSize,
									putOnPpuOnlyBuildJacobianTaskList, generateSolveTasks,
									callbackConstraints );
				currentGroup++;
			}

			const int newSourceConstraintsSize = int(nextConstraintLeftOut - sourceConstraints->begin());
			sourceConstraints->setSizeUnchecked(newSourceConstraintsSize);

			HK_ASSERT2(0xad53a212, !generateSolveTasks || *ptrToBatchSize != 0, "Internal error: No tasks saved in a batch!");
			HK_ASSERT2(0xad654433, !generateSolveTasks || prevSolveTask != HK_NULL, "Internal error: A solve task must be created on the first pass of the loop.");

		}

	}

	// dealloc in reverse order
#	if defined(HK_PLATFORM_HAS_SPU)
	hkMemTempBlockFree<int>(entityAccumulatorInterIndex, numBodies);
#   endif
	hkMemTempBlockFree<int>(entityGroup, numBodies);

	delete entityPtrToIndex; entityPtrToIndex = 0;
	hkMemTempBlockFree<hkUlong>(pointerBuffer,po2);

	HK_ASSERT2(0XAD67544D, !generateSolveTasks || (*ptrToBatchSize != 0 && prevSolveTask && prevSolveTask->m_sizeOfNextBatch == 0), "Batch sizes incorrect.");

	if (generateSolveTasks)
	{
		//if (prevSolveTask)
		{
			HK_ASSERT2(0xad8759dd, firstTaskOfCurrentBatch, "Internal error: firstTaskOfCurrentBatch and prevSolveTask inconsistent.");
			hkpSolveConstraintBatchTask* aTaskFromPreviousBatch = firstTaskOfPreviousBatch;
			hkUint16 batchSize = (*ptrToBatchSize);
			while(aTaskFromPreviousBatch && aTaskFromPreviousBatch != firstTaskOfCurrentBatch)
			{
				aTaskFromPreviousBatch->m_sizeOfNextBatch = batchSize;
				aTaskFromPreviousBatch->m_firstTaskInNextBatch = firstTaskOfCurrentBatch;
				aTaskFromPreviousBatch = aTaskFromPreviousBatch->m_next;
			}

			// Keep order! this is done after the above!

			// Manually mark the last solve batch finished
			prevSolveTask->m_isLastTaskInBatch = true;
			//ptrToBatchSize = &prevSolveTask->m_sizeOfNextBatch;
		}


		// Only here (if generateSolveTasks) the below data matter:
		const unsigned int numAccumulators = 1 + taskHeader->m_numAllEntities;
		taskHeader->m_accumulatorsEnd = hkAddByteOffset(taskHeader->m_accumulatorsBase, numAccumulators * sizeof(hkpVelocityAccumulator));

		taskHeader->m_numApplyGravityJobs        = hkObjectIndex( 1 + (numAccumulators - 1) / hkpSolveApplyGravityJob::MAX_NUM_ACCUMULATORS_FOR_APPLY_GRAVITY_JOB );
		taskHeader->m_numIntegrateVelocitiesJobs = hkObjectIndex( 1 + (numAccumulators - 1) / hkpSolveIntegrateVelocitiesJob::MAX_NUM_ACCUMULATORS_FOR_INTEGRATE_VELOCITIES_JOB );
	}

	//
	// Final checks
	//

	HK_ON_DEBUG( void* bufferEnd = hkAddByteOffset(taskHeader->m_buffer, taskHeader->m_bufferSize ) );
	HK_ASSERT( 0xf032eddf, (void*)schemas <= bufferEnd );
	HK_ASSERT( 0xf032eddf, (void*)schemas <= taskHeader->m_solverTempBase );

	//
	// collect constraints that need to fire callbacks
	//
	{
		// default values in case no callbacks are fired
		outBuildCollection->m_numCallbackConstraints = 0;
		outBuildCollection->m_callbackConstraints = HK_NULL;

		int numCC = callbackConstraints.getSize();
		if ( numCC )
		{
			hkpBuildJacobianTaskCollection::CallbackPair* cc = hkAllocateChunk<hkpBuildJacobianTaskCollection::CallbackPair>( numCC, HK_MEMORY_CLASS_CONSTRAINT_SOLVER );
			// if we keep a pointer to the callback constraints, we need to mark the island read only, so that nobody removes are adds constraints
			callbackConstraints[0].m_internal->getMasterEntity()->getSimulationIsland()->getMultiThreadCheck().markForRead();
			outBuildCollection->m_numCallbackConstraints = numCC;
			outBuildCollection->m_callbackConstraints = cc;
			for (int i = 0; i < numCC; i++)
			{
				cc[i].m_callbackConstraints = callbackConstraints[i].m_internal;
				cc[i].m_atomInfo = callbackConstraints[i].m_atomInfo;
			}
		}
	}

#if defined (HK_DEBUG)
	// check that number of build & solve tasks is the same
	int numBuild = 0;
	int numSolve = 0;

	hkpBuildJacobianTask* bjTask = outBuildCollection->m_buildJacobianTasks;
	hkpSolveConstraintBatchTask* sjTask = outSolveCollection->m_firstSolveJacobiansTask;
	while(bjTask)    { numBuild++; bjTask = bjTask->m_next; }
	while(sjTask)    { numSolve++; sjTask = sjTask->m_next; }

#	if defined (HK_PLATFORM_HAS_SPU)
	hkpBuildJacobianTask* bjPpuTask = outBuildCollection->m_ppuOnlyBuildJacobianTasks;
	while(bjPpuTask) { numBuild++; bjPpuTask = bjPpuTask->m_next; }
#	endif

	HK_ASSERT2(0XAD7655DD, (numBuild == numSolve) || !numSolve, "Number of tasks don't match");
#endif
}


// For now, simply do all integration here.
hkJobQueue::JobStatus HK_CALL integrateJob( hkpMtThreadStructure&		tl,
											hkJobQueue&					jobQueue,
											hkJobQueue::JobQueueEntry&	nextJobOut,
											hkBool&						jobWasCancelledOut )
{
	const hkpIntegrateJob& job = reinterpret_cast<hkpIntegrateJob&>(nextJobOut);
	hkpSimulationIsland* island = job.m_island;
	jobWasCancelledOut = false;

	int solverBufferSize;
	int solverBufferCapacity;
	char* solverBuffer;

	if (island->m_constraintInfo.m_sizeOfSchemas != 0)
	{
#ifndef HK_PLATFORM_SIM_SPU
		HK_ASSERT(0x660f3951, hkMemoryRouter::getInstance().stack().numExternalAllocations() == 0);
#endif
		solverBufferSize = hkpConstraintSolverSetup::calcBufferSize( *island );
		solverBufferCapacity = solverBufferSize;
		solverBuffer = hkMemSolverBufAlloc<char>( solverBufferCapacity );
		HK_ASSERT(0x5598a7b2, (((hkUlong)solverBuffer) & 0xf) == 0 );

		while ( solverBuffer == HK_NULL )
		{	
			// if there is not solver buffer available, delay the execution of the job
			// until another broadphase job is freeing enougth memory
			hkJobQueue::JobQueueEntry nextJob;

			// we have to fake our hkThreadNumber to be the thread number which can take broadphase jobs.
			// otherwise we risk that we are only getting integrate jobs, which cannot allocate buffers.
			// (Or in other words, we want to make the integrate jobs really low priority, lower than the
			// broadphase jobs)
			int oldThreadNumber = HK_THREAD_LOCAL_GET(hkThreadNumber);
			HK_THREAD_LOCAL_SET(hkThreadNumber, HK_BROAD_PHASE_THREAD_AFFINITY);
			hkJobQueue::JobStatus status = jobQueue.getNextJob( nextJob, hkJobQueue::DO_NOT_WAIT_FOR_NEXT_JOB );
			HK_THREAD_LOCAL_SET(hkThreadNumber, oldThreadNumber);

			if ( status == hkJobQueue::GOT_NEXT_JOB)
			{
#ifdef HK_ENABLE_DETERMINISM_CHECKS
				// Cancel the current job, as it will be picked up later
				hkCheckDeterminismUtil::Fuid jobFuid = job.getFuid();
				hkCheckDeterminismUtil::getInstance().finishJob( jobFuid, true );
				hkCheckDeterminismUtil::getInstance().cancelJob( jobFuid );
#endif
				// add my own job again (to be executed later)
				jobQueue.addJob( nextJobOut, hkJobQueue::JOB_LOW_PRIORITY );

				// finish a dummy job so that the open job counter get updated correctly
				hkJobQueue::JobQueueEntry jEntry;
				jEntry.m_jobType =	  HK_JOB_TYPE_DYNAMICS;
				jEntry.m_jobSubType = hkpDynamicsJob::DYNAMICS_JOB_DUMMY;
				jobQueue.finishJob( &jEntry );

				nextJobOut = nextJob;
				jobWasCancelledOut = true;
				return status;
			}

			// avoid starving the job queue (especially on PlayStation(R)3)
			{
				volatile int k;
				for (int i=0; i < 4000; i++)
				{
					k = i;
				}
			}

			// Retry alloc
			solverBuffer = hkMemSolverBufAlloc<char>( solverBufferCapacity );
		}
	}
	else
	{
		// No solver buffer needed
		solverBufferSize = 0;
		solverBufferCapacity = 0;
		solverBuffer = HK_NULL;
	}

	HK_TIMER_BEGIN_LIST("Integrate", "Init");

#ifdef HK_DEBUG_MULTI_THREADING
	island->m_inIntegrateJob = true;
#endif

#if defined (HK_ENABLE_DETERMINISM_CHECKS)
	hkCheckDeterminismUtil::checkMt(0xf0000090, island->m_storageIndex);
	hkCheckDeterminismUtil::checkMt(0xf0000091, island->m_entities.getSize());
	for (int ei = 0; ei < island->m_entities.getSize(); ei++)
	{
		hkpEntity* entity = island->m_entities[ei];
		hkCheckDeterminismUtil::checkMtCrc(0xf0000092, &entity->getMotion()->getTransform(),1);
		hkCheckDeterminismUtil::checkMt(0xf0000093, entity->getMotion()->getLinearVelocity());
		hkCheckDeterminismUtil::checkMt(0xf0000094, entity->getMotion()->getAngularVelocity());
	}
#endif

	const hkpWorldDynamicsStepInfo& stepInfo = tl.m_world->m_dynamicsStepInfo;

	// actions
	if (!tl.m_world->m_processActionsInSingleThread)
	{
        HK_TIMER_SPLIT_LIST("Actions");
		hkArray<hkpAction*>& actions = island->m_actions;
		for (int j = 0; j < actions.getSize(); j++)
		{
			hkpAction* action = actions[j];
			if (action)
			{
#				ifdef HK_DEBUG_MULTI_THREADING
					hkInplaceArray<hkpEntity*, 16> entities;
					action->getEntities(entities);
					{ for (int e = 0; e < entities.getSize(); e++) { entities[e]->markForWrite();   } }
					action->applyAction( stepInfo.m_stepInfo );
					{ for (int e = 0; e < entities.getSize(); e++) { entities[e]->unmarkForWrite(); } }
#				else // not HK_DEBUG_MULTI_THREADING
					action->applyAction( stepInfo.m_stepInfo );
#if defined (HK_ENABLE_DETERMINISM_CHECKS)
					hkInplaceArray<hkpEntity*, 16> entities;
					action->getEntities(entities);
					for (int e = 0; e < entities.getSize(); e++)
					{
						hkpEntity* entity = entities[e];
						hkCheckDeterminismUtil::checkMtCrc(0xf0000092, &entity->getMotion()->getTransform(),1);
						hkCheckDeterminismUtil::checkMtCrc(0xf0000093, &entity->getMotion()->getLinearVelocity(),1);
						hkCheckDeterminismUtil::checkMtCrc(0xf0000094, &entity->getMotion()->getAngularVelocity(),1);
 
					}
#endif
#				endif // HK_DEBUG_MULTI_THREADING
			}
		}
	}

	hkpBuildJacobianTaskHeader* taskHeader = new hkpBuildJacobianTaskHeader;

	HK_TIMER_SPLIT_LIST("Island Splitting");
	hkpEntity*const* entities = island->getEntities().begin();

#if defined( HK_PLATFORM_HAS_SPU)
	{
		// make sure entities are aligned
		if ( hkUlong(entities) & 0x0f )
		{
			island->m_entities.reserve(4);
			entities = island->getEntities().begin();
		}
		HK_ASSERT( 0xf053fd43, (hkUlong(entities)&0x0f) == 0 );
	}
#endif

	unsigned int numEntities = island->getEntities().getSize();

	taskHeader->m_numAllEntities					= hkObjectIndex(numEntities);
	taskHeader->m_allEntities						= entities;
	taskHeader->m_entitiesCapacity					= hkObjectIndex(island->getEntities().getCapacity());
	taskHeader->m_numUnfinishedJobsForBroadphase	= 0;
	taskHeader->m_islandShouldBeDeactivated			= 1;		// will be set by the integrateMotion jobs to 0
	taskHeader->m_buffer							= HK_NULL;
	taskHeader->m_bufferSize						= 0;
	taskHeader->m_impulseLimitsBreached				= HK_NULL;
	taskHeader->m_tasks.m_buildJacobianTasks		= 0;
	taskHeader->m_accumulatorsBase					= HK_NULL;

	//
	// try to decide whether we want a split job
	//
	int splitRequested = 0;
	{
        HK_TIMER_SPLIT_LIST("SplitIslands");
		island->m_splitCheckFrameCounter++;
		hkCheckDeterminismUtil::checkMt(0xf0000098, island->m_isSparse);
		hkCheckDeterminismUtil::checkMt(0xf0000099, island->m_splitCheckRequested);
		hkCheckDeterminismUtil::checkMt(0xf000009a, island->m_splitCheckFrameCounter);
		hkCheckDeterminismUtil::checkMt(0xf000009b, tl.m_world->m_wantSimulationIslands);
		if ( !island->m_isSparse )
		{
			// check for normal island splitting every 8th frame
			if ( island->m_splitCheckRequested && ((island->m_splitCheckFrameCounter & 0x7) == 0) && tl.m_world->m_wantSimulationIslands )
			{
				splitRequested = 1;
			}
		}
		else
		{
			// now we have a not fully connected island. Check whether it grew over the limit size. Do this check every frame
			const int islandSize = hkpWorldOperationUtil::estimateIslandSize( island->m_entities.getSize(), island->m_numConstraints );
			const hkBool canIslandBeSparse = hkpWorldOperationUtil::canIslandBeSparse( tl.m_world, islandSize );
			hkCheckDeterminismUtil::checkMt(0xf000009c, islandSize);
			hkCheckDeterminismUtil::checkMt(0xf000009d, canIslandBeSparse);

			if ( !canIslandBeSparse )
			{
					// our island got too big, break it up with only 3 frames delay
				island->m_isSparse = false;

				if ( (island->m_splitCheckFrameCounter & 0x3) == 0 )
				{
					splitRequested = 1;
				}
			}
			else
			{
				// try to split off deactivated sub islands
				if ( (island->m_splitCheckFrameCounter & 0x7) == 0 )
				{
					splitRequested = 1;
				}
			}

		}
	}

	if ( island->m_constraintInfo.m_sizeOfSchemas == 0 )  
	{
		HK_ASSERT2(0xad343281, !solverBuffer, "Solver buffer is expected to be not allocated.");

		if(/*!splitRequested &&*/ tl.m_world->m_allowIntegrationOfIslandsWithoutConstraintsInASeparateJob)
		{
			taskHeader->m_buffer = HK_NULL;
			taskHeader->m_accumulatorsBase = HK_NULL;
			taskHeader->m_schemasBase = HK_NULL;
			taskHeader->m_solverTempBase = HK_NULL;
			taskHeader->m_bufferSize = 0;
			taskHeader->m_bufferCapacity = 0;

			HK_ASSERT2(0xad343282, island->m_constraintInfo.m_numSolverResults == 0 && island->m_constraintInfo.m_numSolverElemTemps == 0, "Ensuring solver values are nulled/zeroed when running Integrate Motions job without the solver.");
			taskHeader->m_numSolverResults		= island->m_constraintInfo.m_numSolverResults;
			taskHeader->m_numSolverElemTemps    = island->m_constraintInfo.m_numSolverElemTemps;
			taskHeader->m_constraintQueryIn		= &tl.m_constraintQueryIn;
			taskHeader->m_bufferCapacity		= solverBufferCapacity;

			int numIntegrateMotionJobs = 1 + ((numEntities-1)/ hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB);
			taskHeader->m_numUnfinishedJobsForBroadphase	= splitRequested + numIntegrateMotionJobs;	// split job and integrate motion job
			taskHeader->m_splitCheckRequested               = hkChar(splitRequested);
			taskHeader->m_solveTasks.m_firstSolveJacobiansTask = HK_NULL;

			// MOVE THIS BLOCK UPWARDS, WHERE IT BELONGS LOGICALLY-- ALSO DON'T DO SOLVER FOR THAT CASE.
			hkpIntegrateMotionJob& j = reinterpret_cast<hkpIntegrateMotionJob&>(nextJobOut);
			new (&j) hkpIntegrateMotionJob(job, *taskHeader);
			{
				//j.m_jobSubType			= hkpDynamicsJob::DYNAMICS_JOB_INTEGRATE_MOTION;
				j.m_taskHeader			= taskHeader;
				//j.m_firstEntityIdx		= 0;
				//j.m_numEntities			= hkObjectIndex(numEntities);
				//j.m_solveConstraintBatchTask = HK_NULL;
				//j.m_buffer              = HK_NULL;
				j.m_applyForcesAndStepMotionOnly = true;
				j.m_jobSpuType			= HK_JOB_SPU_TYPE_ENABLED;

				//j.m_jobPriority   = hkJobQueue::JOB_HIGH_PRIORITY;


				// islandIndex, mtThreadStructure, jobSid for determinism
			}

			taskHeader->m_exportFinished = hkChar(1);
			taskHeader->m_openJobs       = numIntegrateMotionJobs;

			HK_TIMER_END_LIST();

			// We need to put this here because the integrate (and the broadphase) job doesn't call finishDynamicsJob
			//
			// This is same as spawnSplitSimulationIslandJob( j, *taskHeader, jobQueue, data );
			if ( splitRequested )
			{ 
				HK_ALIGN16(char dynamicDataStorage[sizeof(hkJobQueue::DynamicData)]);
				hkJobQueue::DynamicData* data = jobQueue.lockQueue( dynamicDataStorage );

				hkpSplitSimulationIslandJob splitJob( j, sizeof(hkpSplitSimulationIslandJob) );
				jobQueue.addJobQueueLocked( data, (const hkJobQueue::JobQueueEntry&)splitJob, hkJobQueue::JOB_HIGH_PRIORITY);

				jobQueue.unlockQueue( data );
			}

			return jobQueue.finishAddAndGetNextJob( HK_JOB_TYPE_DYNAMICS, hkJobQueue::JOB_HIGH_PRIORITY, nextJobOut ); 
		}
		else
		{
			//
			//	Single object
			//
			HK_TIMER_SPLIT_LIST("SingleObj");
			{	 for ( unsigned int i=0; i < numEntities; i++ ){ entities[i]->markForWrite(); }	}
			int numInactiveFrames = hkRigidMotionUtilApplyForcesAndStep( stepInfo.m_solverInfo, stepInfo.m_stepInfo, stepInfo.m_solverInfo.m_globalAccelerationPerStep, (hkpMotion*const*)island->m_entities.begin(), island->m_entities.getSize(), HK_OFFSET_OF(hkpEntity,m_motion) );
			hkpEntityAabbUtil::entityBatchRecalcAabb(island->getWorld()->getCollisionInput(), island->m_entities.begin(), island->m_entities.getSize());
			{	 for ( unsigned int i=0; i < numEntities; i++ ){ entities[i]->unmarkForWrite(); }	}
			if ( numInactiveFrames <= hkpMotion::NUM_INACTIVE_FRAMES_TO_DEACTIVATE)
			{
				taskHeader->m_islandShouldBeDeactivated = 0;
			}
			HK_ASSERT2(0xad382112, solverBuffer == HK_NULL && solverBufferCapacity == 0, "No solver buffer expected.");
			goto BUILD_BROAD_PHASE_JOB_AND_RETURN;
		}
	}



		// if we have chain constraints, the constraint number is just plain wrong, so use the number of entities instead
	HK_ASSERT2(0xad382111, island->m_constraintInfo.m_sizeOfSchemas > 0, "No schemas case expected to be handled earlier in code.");
	hkUint32 estimatedNumConstraints; estimatedNumConstraints = hkMath::max2( island->m_numConstraints, island->getEntities().getSize() );
	if ( estimatedNumConstraints < tl.m_simulation->m_multithreadConfig.m_maxNumConstraintsSolvedSingleThreaded )
	{

		//
		// single threaded execution
		//
		HK_TIMER_SPLIT_LIST("Solver 1Cpu");
		{
			{	 for ( unsigned int i=0; i < numEntities; i++ ){ entities[i]->markForWrite(); }	}
			island->getMultiThreadCheck().markForRead();

			hkpConstraintQueryIn in = tl.m_constraintQueryIn;
			int numInactiveFrames = hkpConstraintSolverSetup::solve( stepInfo.m_stepInfo, stepInfo.m_solverInfo, in, *island,	solverBuffer, solverBufferSize, entities,    numEntities );


			island->getMultiThreadCheck().unmarkForRead();
			{	 for ( unsigned int i=0; i < numEntities; i++ ){ entities[i]->unmarkForWrite(); }	}

			if ( numInactiveFrames <= hkpMotion::NUM_INACTIVE_FRAMES_TO_DEACTIVATE)
			{
				taskHeader->m_islandShouldBeDeactivated = 0;
			}
		}
		hkMemSolverBufFree(solverBuffer, solverBufferCapacity );
BUILD_BROAD_PHASE_JOB_AND_RETURN:

		hkCheckDeterminismUtil::checkMt(0xf000009e, splitRequested);
		if ( splitRequested )
		{
			hkCpuSplitSimulationIslandJobImpl( island, taskHeader->m_newSplitIslands );
		}

		//
		// XXX Island and world (if this is the last island) post integrate callbacks should be called here, but whats the point?
		//
		HK_TIMER_END_LIST();
		{
			taskHeader->m_exportFinished = 1;
			new (&nextJobOut) hkpBroadPhaseJob( job, taskHeader );
			HK_ON_DEBUG(hkJobQueue::JobStatus status = )
				jobQueue.finishAddAndGetNextJob( HK_JOB_TYPE_DYNAMICS, hkJobQueue::JOB_HIGH_PRIORITY, nextJobOut ); 
			HK_ASSERT( 0xf0213445, status == hkJobQueue::GOT_NEXT_JOB);
		}
		return hkJobQueue::GOT_NEXT_JOB;
	}


	{
		HK_TIMER_SPLIT_LIST("SetupJobs");
		//
		// multi threaded solution
		//

		{
			hkpConstraintSolverSetup::calcBufferOffsetsForSolve( *island, solverBuffer, solverBufferSize, *taskHeader );
			taskHeader->m_numSolverResults		= island->m_constraintInfo.m_numSolverResults;
			taskHeader->m_numSolverElemTemps    = island->m_constraintInfo.m_numSolverElemTemps;
			taskHeader->m_constraintQueryIn		= &tl.m_constraintQueryIn;
			taskHeader->m_bufferCapacity		= solverBufferCapacity;
			taskHeader->m_exportFinished		= hkChar(0);

			int numIntegrateMotionJobs = 1 + ((numEntities-1)/ hkpIntegrateMotionJob::ACCUMULATORS_PER_JOB);
			taskHeader->m_numUnfinishedJobsForBroadphase	= splitRequested + numIntegrateMotionJobs;	// split job and integrate motion job
			taskHeader->m_splitCheckRequested               = hkChar(splitRequested);

			taskHeader->m_openJobs = 1 + 1 + ((numEntities-1)/hkpBuildAccumulatorsJob::ACCUMULATORS_PER_JOB); // 1 hkpBuildAccumulatorsJob + 1 hkpCreateJacobianTasksJob (see below)

			//
			// create a new high-priority job for building the accumulators and add it to the queue
			//
			hkpBuildAccumulatorsJob& j = reinterpret_cast<hkpBuildAccumulatorsJob&>(nextJobOut);
			{
				j.m_jobSubType			= hkpDynamicsJob::DYNAMICS_JOB_BUILD_ACCUMULATORS;
				j.m_taskHeader			= taskHeader;
				j.m_islandEntitiesArray	= entities;
				j.m_firstEntityIdx		= 0;
				j.m_numEntities			= hkObjectIndex(numEntities);
				j.m_jobSpuType			= HK_JOB_SPU_TYPE_ENABLED;
			}

			jobQueue.addJob( nextJobOut, hkJobQueue::JOB_HIGH_PRIORITY );
		}

		//
		// immediately execute 'buildJacobianTask' job (without actually adding it to the queue).
		//
		hkpBuildJacobianTaskCollection outBuildCollection;
		hkpSolveJacobiansTaskCollection outSolveCollection;
		{
			island->markAllEntitiesReadOnly();
			island->getMultiThreadCheck().markForRead();
			createBuildJacobianAndSolveJacobianTaskCollection( island, stepInfo.m_solverInfo.m_forceCoherentConstraintOrderingInSolver, taskHeader, &outBuildCollection, &outSolveCollection );
			island->getMultiThreadCheck().unmarkForRead();
			island->unmarkAllEntitiesReadOnly();
		}

		HK_TIMER_END_LIST();
		taskHeader->m_tasks = outBuildCollection;
		taskHeader->m_solveTasks = outSolveCollection;

		hkpCreateJacobianTasksJob fjob( job, taskHeader );

		// we have to tell the job queue that hkpCreateJacobianTasksJob has finished,
		// so that the next dependent job can be activated
		hkJobQueue::JobStatus status = jobQueue.finishJobAndGetNextJob( (hkJobQueue::JobQueueEntry*)&fjob, nextJobOut );
		return status;
	}
}

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
