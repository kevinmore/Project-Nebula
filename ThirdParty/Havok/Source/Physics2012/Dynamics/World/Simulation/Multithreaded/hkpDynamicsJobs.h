/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_DYNAMICS_JOBS_H
#define HK_DYNAMICS_JOBS_H

#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>
#include <Common/Base/Types/Physics/hkStepInfo.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics2012/Dynamics/Constraint/Setup/hkpConstraintSolverSetup.h>
#include <Physics2012/Dynamics/World/Simulation/Continuous/hkpContinuousSimulation.h>
#include <Common/Base/Thread/JobQueue/hkJobQueue.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnTrack.h>
#include <Physics2012/Dynamics/World/CommandQueue/hkpPhysicsCommandQueue.h>
#include <Physics2012/Collide/Agent3/Machine/Midphase/hkpShapeKeyTrack.h>

#if defined (HK_PLATFORM_HAS_SPU)
	// Needed for the accumulator size (when pre-calculating the number of hkSolveApplyGravityJobs)
#	include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#endif

struct hkpBuildJacobianTaskHeader;

	/// The base class for all dynamics jobs
	// Important: the 16bit m_jobType HAS to be the first member of this class and it HAS to be 16byte aligned! See hkJob for more details.
class hkpDynamicsJob : public hkJob
{
	public:

		enum JobSubType
		{
			DYNAMICS_JOB_INTEGRATE,
			DYNAMICS_JOB_BUILD_ACCUMULATORS,
			DYNAMICS_JOB_CREATE_JACOBIAN_TASKS,
			DYNAMICS_JOB_FIRE_JACOBIAN_SETUP_CALLBACK,
			DYNAMICS_JOB_BUILD_JACOBIANS,
			DYNAMICS_JOB_SPLIT_ISLAND,
			DYNAMICS_JOB_SOLVE_CONSTRAINTS,
			DYNAMICS_JOB_SOLVE_APPLY_GRAVITY,
			DYNAMICS_JOB_SOLVE_CONSTRAINT_BATCH, 
			DYNAMICS_JOB_SOLVE_INTEGRATE_VELOCITIES,
			DYNAMICS_JOB_SOLVE_EXPORT_RESULTS,
			DYNAMICS_JOB_INTEGRATE_MOTION,
			DYNAMICS_JOB_BROADPHASE,
			DYNAMICS_JOB_AGENT_SECTOR,
			DYNAMICS_JOB_POST_COLLIDE,
			DYNAMICS_JOB_AGENT_NN_ENTRY,
			DYNAMICS_JOB_DUMMY,		///< for finishing a dummy job
			DYNAMICS_JOB_END
		};

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpDynamicsJob );

	public:

		HK_FORCE_INLINE hkpDynamicsJob( JobSubType newType, hkUint16 newSize, const hkpDynamicsJob& srcJob, hkJobSpuType jobSpuType = HK_JOB_SPU_TYPE_ENABLED );

		enum NoJob { NO_SRC_JOB };
		HK_FORCE_INLINE hkpDynamicsJob( JobSubType type, hkUint16 size, NoJob noSrcJob, hkJobSpuType jobSpuType = HK_JOB_SPU_TYPE_ENABLED  );
		
		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popDynamicsJobTask( hkpDynamicsJob& out );

	protected:
		friend class hkpFireJacobianSetupCallback;
		friend class hkpBuildJacobiansJob;
		friend class hkpSolveConstraintsJob;
		friend class hkpSolveApplyGravityJob;
		friend class hkpSolveConstraintBatchJob;
		friend class hkpSolveIntegrateVelocitiesJob;
		friend class hkpSolveExportResultsJob;
		friend class hkpPostCollideJob;
		friend class hkpBroadPhaseJob;
		friend class hkpAgentSectorJob;
		friend class hkpAgentNnEntryJob;
		friend class hkpCreateJacobianTasksJob;
		friend class hkpSplitSimulationIslandJob;
		friend class hkpIntegrateMotionJob;


	public: // for debugging
		/// this island index is only used internally in the job queue, use m_island instead
		hkObjectIndex m_islandIndex;
			// Job sequential id.
			// increased for each popped job of the same type; zeroed when job morphs into a new one.
		HK_ON_CPU_DETERMINISM_CHECKS_ENABLED( hkObjectIndex m_jobSid; ) 
	public:

		/// the simulation island: this is set by popJobTask, no need to set it by hand
		hkpSimulationIsland* m_island;

		HK_CPU_PTR(struct hkpBuildJacobianTaskHeader*) m_taskHeader;

		struct hkpMtThreadStructure* m_mtThreadStructure;

			// Get Frame-unique ID
		HK_ON_DETERMINISM_CHECKS_ENABLED( hkCheckDeterminismUtil::Fuid getFuid() const; )
};



class hkpIntegrateJob : public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpIntegrateJob);

		HK_FORCE_INLINE hkpIntegrateJob(int numIslands);

		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpIntegrateJob& out );

	public:

		int m_numIslands;
};


class hkpBuildAccumulatorsJob: public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpBuildAccumulatorsJob );

		enum { ACCUMULATORS_PER_JOB = 128 };

		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpBuildAccumulatorsJob& out );

	protected:
		HK_FORCE_INLINE hkpBuildAccumulatorsJob( const hkpDynamicsJob& job, JobSubType type = DYNAMICS_JOB_BUILD_ACCUMULATORS ) : hkpDynamicsJob(type, sizeof(hkpBuildAccumulatorsJob), job) { }

	public:

			// pointer to the island's entity list in main memory
		HK_CPU_PTR(hkpEntity*const*) m_islandEntitiesArray;

			// this offset into m_islandEntitiesArray defines the first entity to be processed in this job/batch
		hkObjectIndex m_firstEntityIdx;

			// number of entities to be processed in this job/batch
		hkObjectIndex m_numEntities;
};


	// This job is never actually added to the queue.
	// This is because it is done by the same job that adds the build accumulators
	// job to the queue.  It is only used as an input to the finishJobAndGetNextJob function.
class hkpCreateJacobianTasksJob: public hkpBuildAccumulatorsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpCreateJacobianTasksJob );
		HK_FORCE_INLINE hkpCreateJacobianTasksJob(const hkpIntegrateJob& job, hkpBuildJacobianTaskHeader* newTaskHeader) : hkpBuildAccumulatorsJob( job, DYNAMICS_JOB_CREATE_JACOBIAN_TASKS) { m_taskHeader = newTaskHeader; }
};

class hkpFireJacobianSetupCallback : public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpFireJacobianSetupCallback );

		HK_FORCE_INLINE hkpFireJacobianSetupCallback(const hkpBuildAccumulatorsJob& baj) : hkpDynamicsJob( DYNAMICS_JOB_FIRE_JACOBIAN_SETUP_CALLBACK,  sizeof(hkpFireJacobianSetupCallback), baj, HK_JOB_SPU_TYPE_DISABLED) {}
};


class hkpBuildJacobiansJob : public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpBuildJacobiansJob );

		HK_FORCE_INLINE hkpBuildJacobiansJob(const hkpDynamicsJob& dynamicsJob, struct hkpBuildJacobianTask* firstTaskInMainMemory, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy);

		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpBuildJacobiansJob& out );

		union
		{
			struct hkpBuildJacobianTask*             m_buildJacobianTask;
			HK_CPU_PTR(struct hkpBuildJacobianTask*) m_buildJacobianTaskInMainMemory;
		};

		HK_CPU_PTR(const hkpConstraintQueryIn*) m_constraintQueryIn; 

		hkBool m_finishSchemasWithGoto;
};

	// this job runs parallel to the solve constraint job 
class hkpSplitSimulationIslandJob : public hkpDynamicsJob
{	
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpSplitSimulationIslandJob );

		HK_FORCE_INLINE hkpSplitSimulationIslandJob(const hkpDynamicsJob& job, hkUint16 size, JobSubType type = DYNAMICS_JOB_SPLIT_ISLAND) : hkpDynamicsJob(type, size, job, HK_JOB_SPU_TYPE_DISABLED ) {}

		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpSplitSimulationIslandJob& out );

};


class hkpSolveConstraintsJob : public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpSolveConstraintsJob );

		HK_FORCE_INLINE hkpSolveConstraintsJob(const hkpDynamicsJob& job, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy);

		HK_CPU_PTR(void*) m_buffer;
		hkUint32   m_bufferSize;

		hkUint32 m_accumulatorsOffset;	// <todo> remove those variables as they make the job pretty big
		hkUint32 m_schemasOffset;
		hkUint32 m_solverTempOffset;

		hkInt32  m_numSolverResults;
		hkInt32  m_numSolverElemTemps;
};

class hkpSolveApplyGravityJob : public hkpDynamicsJob
{
public:
	enum { MAX_NUM_ACCUMULATORS_FOR_APPLY_GRAVITY_JOB = 128 };

	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpSolveApplyGravityJob  );

	HK_FORCE_INLINE hkpSolveApplyGravityJob(const hkpDynamicsJob& dynamicsJob, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy);

	HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpSolveApplyGravityJob& out );

	HK_ALIGN16( HK_CPU_PTR(hkpVelocityAccumulator*) m_accumulators );
	HK_CPU_PTR(hkpVelocityAccumulator*) m_accumulatorsEnd;
};

class hkpSolveConstraintBatchJob : public hkpDynamicsJob
{
public:
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpSolveConstraintBatchJob );

	HK_FORCE_INLINE hkpSolveConstraintBatchJob(const hkpSolveApplyGravityJob& sagj, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy);
	HK_FORCE_INLINE hkpSolveConstraintBatchJob(const hkpSolveConstraintBatchJob& scbij); 
	HK_FORCE_INLINE hkpSolveConstraintBatchJob(const class hkpSolveIntegrateVelocitiesJob& sivbij, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy );

	HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpSolveConstraintBatchJob& out );

	HK_ALIGN16( hkUint32 m_currentSolverStep );
	hkUint32 m_numSolverMicroSteps; // this is only initialized within the job itself.
	hkUint32 m_currentSolverMicroStep;

	union
	{
		struct hkpSolveConstraintBatchTask*             m_solveConstraintBatchTask;
		HK_CPU_PTR(struct hkpSolveConstraintBatchTask*) m_solveConstraintBatchTaskInMainMemory;
	};
};

class hkpSolveExportResultsJob : public hkpSplitSimulationIslandJob
{
public:
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpSolveExportResultsJob  );

		// This is needed to instantiate the derived hkpIntegrateMotionJob.
	HK_FORCE_INLINE hkpSolveExportResultsJob( const hkpDynamicsJob& job, hkUint16 size, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy, JobSubType type = DYNAMICS_JOB_SOLVE_EXPORT_RESULTS );

	HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpSolveExportResultsJob& out );

	union
	{
		struct hkpSolveConstraintBatchTask*             m_solveConstraintBatchTask;
		HK_CPU_PTR(struct hkpSolveConstraintBatchTask*) m_solveConstraintBatchTaskInMainMemory;
	};
#if !defined(HK_PLATFORM_SPU)
	hkpImpulseLimitBreachedHeader* m_impulseLimitsBreachedPadding;
	int m_numImpulseLimitsBreachedPadding;
#else
		// this is used by the finishDynamicsJob function to build a linked list of impulseLimitBreached elems
	struct hkpImpulseLimitBreachedHeader* m_impulseLimitsBreached;
	int m_numImpulseLimitsBreached;
#endif
};

class hkpSolveIntegrateVelocitiesJob : public hkpDynamicsJob
{
public:
 	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpSolveIntegrateVelocitiesJob );

	enum { MAX_NUM_ACCUMULATORS_FOR_INTEGRATE_VELOCITIES_JOB = 128 };

	HK_FORCE_INLINE hkpSolveIntegrateVelocitiesJob(const hkpSolveConstraintBatchJob& scbij, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy );

	HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpSolveIntegrateVelocitiesJob& out );

	HK_ALIGN16( HK_CPU_PTR(hkpVelocityAccumulator*) m_accumulators );
	HK_CPU_PTR(hkpVelocityAccumulator*) m_accumulatorsEnd; 

	hkUint32 m_currentSolverStep;

		// We need that because:
		//  we only access hkpWorld::dynamicsStepInfo::solverInfo::numSolverSteps from inside the 'process' job function. and only there we can check whether the current step is the last one.
		// We then query the flag in the jobQueue to know what to do when the IntegrateVelocities job is done.
	hkBool m_solvingFinished;
};






class hkpIntegrateMotionJob : public hkpSolveExportResultsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpIntegrateMotionJob);

		//constructor
		HK_FORCE_INLINE hkpIntegrateMotionJob( const hkpDynamicsJob& job, const hkpBuildJacobianTaskHeader& localTaskHeaderCopy );

		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpIntegrateMotionJob& out );

	public:

		enum { ACCUMULATORS_PER_JOB  = 128 };

		HK_CPU_PTR(void*) m_buffer; //we need accumulators

		hkObjectIndex m_numEntities;
		hkObjectIndex m_firstEntityIdx;

			// the number of inactive frames. This variable is set by the job and analyzed
			// by finish job func
		int m_numInactiveFrames;
			// The flag is set, when the job is used without the solver, and gravity should be applied 
			// when integrating motions.
		hkBool m_applyForcesAndStepMotionOnly;
};


	// BROADPHASE
	// Creates hkAgentSectorJobs, adds to global lists of new and old pairs
	// For now does all broadphase
	// if m_newSplitIslands is set, the pop job assumes that these island
	// were created by the splitIslandJob and finalizes the split,
	// this includes adding new broadphase jobs to the jobqueue for each new
	// simulation island
class hkpBroadPhaseJob : public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpBroadPhaseJob);
		HK_FORCE_INLINE hkpBroadPhaseJob( const hkpDynamicsJob& job, hkpBuildJacobianTaskHeader* newTaskHeader );

		hkJobQueue::JobPopFuncResult popJobTask( hkArray<hkpSimulationIsland*>& islands, hkpBroadPhaseJob& out );

	public:
		hkObjectIndex m_numIslands;
};


// Base class for hkpAgentSectorJob and hkAgentNnEntryJob.
class hkpAgentBaseJob : public hkpDynamicsJob
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpAgentBaseJob);

			// This maximum number of hkpAgentNnEntry that can be processed in one hkpAgentBaseJob (hkpAgentSectorJob or hkpAgentNnEntryJob).
#if (HK_POINTER_SIZE == 4)
		
		enum { MAX_AGENT_NN_ENTRIES_PER_TASK = 32 };
#elif (HK_POINTER_SIZE == 8)
		enum { MAX_AGENT_NN_ENTRIES_PER_TASK = 48 };
#endif

	public:

		HK_FORCE_INLINE hkpAgentBaseJob(const hkpDynamicsJob& job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf);
		HK_FORCE_INLINE hkpAgentBaseJob(hkpDynamicsJob::NoJob noJob, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf);

	public:

		HK_CPU_PTR(struct hkpAgentSectorHeader*) m_header;

		hkUint16	m_taskIndex;		// used to access the hkpAgentSectorHeader
		hkUint16	m_numElements;
		hkUint16	m_numElementsPerTask;
		hkEnum<hkpAgentNnTrackType, hkUchar> m_agentNnTrackType;
		
		void*const*	m_elements; // this can point to an array of hkpAgentNnSector* or hkpAgentNnEntry*

		hkStepInfo	m_stepInfo; 
};

class hkpAgentSectorBaseJob : public hkpAgentBaseJob
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpAgentSectorBaseJob);
		HK_FORCE_INLINE hkpAgentSectorBaseJob(const hkpDynamicsJob& job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf );

	public:
			/// Only used by the split (midphase/narrowphase) collision pipeline.
		hkpShapeKeyTrack* m_shapeKeyTrack;
		hkUint16	m_bytesUsedInLastSector;
};

// Perform narrowphase collision detection on many agent sectors
class hkpAgentSectorJob : public hkpAgentSectorBaseJob
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpAgentSectorJob);
		HK_FORCE_INLINE hkpAgentSectorJob(const hkpBroadPhaseJob& job, const hkStepInfo& stepInfo, struct hkpAgentNnSector*const* sectors, int numSectors, int maxNumSectorsPerTask, int bytesInLastSector, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf );

		HK_FORCE_INLINE  hkJobQueue::JobPopFuncResult popJobTask( hkpAgentSectorJob& out );
};

class hkpAgentNnEntryBaseJob : public hkpAgentBaseJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpAgentNnEntryBaseJob );	
		HK_FORCE_INLINE hkpAgentNnEntryBaseJob( const hkpDynamicsJob& job, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf );
		HK_FORCE_INLINE hkpAgentNnEntryBaseJob( hkpDynamicsJob::NoJob noJob, hkUint16 size, const hkStepInfo& stepInfo, void*const* elements, int numElements, int maxNumElementsPerTask, JobSubType type, hkpAgentNnTrackType nnTrackType, bool useStaticCompoundElf );

	public:
			/// Only used by the split (midphase/narrowphase) collision pipeline.
		hkpShapeKeyTrack* m_shapeKeyTrack;
		hkEnum<hkpContinuousSimulation::CollisionQualityOverride, hkUchar> m_collisionQualityOverride;
};

	/// Job representing a collection of agent entries to process.
class hkpAgentNnEntryJob : public hkpAgentNnEntryBaseJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpAgentNnEntryJob );	

			/// If processed on CPU, the track type won't matter. If processed on SPU, the track type must be provided.
		HK_FORCE_INLINE hkpAgentNnEntryJob( const hkStepInfo& stepInfo, struct hkpAgentNnEntry*const* entries, int numEntries, int maxNumAgentNnEntriesPerTask, hkpContinuousSimulation::CollisionQualityOverride collisionQualityOverride, bool useStaticCompoundElf, hkpAgentNnTrackType nnTrackType = HK_AGENT3_INVALID_TRACK );
	
		HK_FORCE_INLINE hkJobQueue::JobPopFuncResult popJobTask( hkpAgentNnEntryJob& out );
};


	// This structure is created if several threads are doing agent sector jobs on one island.
	// The layout is that this header is followed by an array of pointers to JobInfo.
	// There is one jobInfo for each agentSector job.
	// At the end a hkpPostCollideJob is fired to complete the missing constraint modifications
struct hkpAgentSectorHeader
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpAgentSectorHeader );

	static hkpAgentSectorHeader* allocate(int numTasks, int numElementsPerTask);
	void deallocate();
	static HK_FORCE_INLINE int HK_CALL getAllocatedSize( int numQueues ){ return sizeof(void*) * HK_HINT_SIZE16(numQueues) + sizeof(hkpAgentSectorHeader); }

	// JobInfo uses a dynamically-sized command queue. The command queue buffer for this is 'appended' to the actual JobInfo
	// struct, i.e. we manually allocate a memory chunk that fits the JobInfo struct plus the command queue.
	struct JobInfo
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, JobInfo );

		JobInfo(int commandQueueCapacityInBytes) 
		{
			HK_ASSERT(0xaf51e231, (commandQueueCapacityInBytes & 0xf) == 0); // commandQueueCapacityInBytes has to be a multiple of 16
			m_commandQueue.init((hkpPhysicsCommand*)(this+1), commandQueueCapacityInBytes);
		}

		HK_ALIGN16( hkpConstraintInfo m_constraintInfo );
		hkpPhysicsCommandQueue m_commandQueue;
		// Located from here on is the actual command queue buffer.
	};

	HK_FORCE_INLINE  JobInfo*  getJobInfo (int index)	{	return (reinterpret_cast<JobInfo**>(this+1))[ HK_HINT_SIZE16(index) ];	}
	HK_FORCE_INLINE  JobInfo** getJobInfos()			{	return (reinterpret_cast<JobInfo**>(this+1));	}

	int						m_numTotalTasks;
	mutable int				m_openJobs;
	int						m_sizeOfJobInfo;

	//
	hkpShapeKeyTrack* m_shapeKeyTracks;
	int m_numShapeKeyTracks;
};


// Perform post narrowphase collision detection on one island
class hkpPostCollideJob : public hkpDynamicsJob
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hkpPostCollideJob);
		HK_FORCE_INLINE hkpPostCollideJob( const hkpAgentBaseJob& asj );

		HK_CPU_PTR(hkpAgentSectorHeader*) m_header;
};

#include <Physics2012/Dynamics/World/Simulation/Multithreaded/hkpDynamicsJobs.inl>

#endif // HK_DYNAMICS_JOBS_H

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
