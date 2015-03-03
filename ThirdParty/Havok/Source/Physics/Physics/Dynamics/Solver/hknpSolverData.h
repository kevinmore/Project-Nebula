/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_DATA_H
#define HKNP_SOLVER_DATA_H

#include <Physics/Physics/Dynamics/World/Deactivation/hknpCollisionPair.h>
#include <Physics/Physics/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianInfo.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>

class hkpConstraintData;
class hknpSimulationContext;


/// The output of the collide step, and input into the solve step.
class hknpSolverData : public hkBaseObject
{
	public:

		/// Per thread data.
		struct ThreadData
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, ThreadData );

			/// Stream which collects all pairs of bodies, which should deactivate together.
			hknpCdPairStream m_activePairStream;

			hknpConstraintSolverJacobianStream  m_jacMovingStream;
			hknpConstraintSolverJacobianStream  m_jacFixedStream;
			hknpConstraintSolverJacobianStream  m_jacConstraintsStream;
			hknpLiveJacobianInfoStream			m_liveJacInfoStream;
			hknpConstraintSolverJacobianStream	m_solverTempsStream;
		};

		/// A simple material that can be used as a parameter for adding immediate contact constraints.
		struct SimpleMaterial
		{
			SimpleMaterial()
			{
				m_friction = 1.0f;
				m_maxImpulse = HK_REAL_MAX;
				m_fractionOfClippedImpulseToApply = 1.0f;
			}

			hkReal m_friction;
			hkReal m_maxImpulse;
			hkReal m_fractionOfClippedImpulseToApply;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSolverData );

#if !defined(HK_PLATFORM_SPU)

		hknpSolverData( hkThreadLocalBlockStreamAllocator* allocator, int numThreads, int numCells, int numLinks );

		virtual ~hknpSolverData() {}

		virtual void clear( hkThreadLocalBlockStreamAllocator* allocator );


		/// Add a contact constraint which is valid for one frame only.
		/// This can only be called between collide() and solve().
		virtual void addImmediateContactConstraint(
			hknpBodyId bodyIdA, hknpBodyId bodyIdB,
			const hknpManifold& manifold,
			const hknpMaterial* childMaterialA = HK_NULL, const hknpMaterial* childMaterialB = HK_NULL,
			const SimpleMaterial& material = SimpleMaterial(),
			hknpMxContactJacobian** mxJacOut = HK_NULL, int* mxIdOut = HK_NULL );

		/// Add a contact constraint which is valid for one frame only.
		/// This can only be called between collide() and solve().
		virtual void addImmediateContactConstraintUsingJacobian( const hknpContactJacobian<1>& preComputedJacobianIn );
		virtual void addImmediateContactConstraintUsingJacobianBatch( const hknpMxContactJacobian& preComputedJacobianBatchIn );

		/// Add a constraint which is valid for one frame only.
		/// This can only be called between collide() and solve().
		/// Using a valid immediate ID will result in an immediate constraint that has it's impulses exported,
		/// this ID is used to call hknpModifier::postConstraintExport() if enabled.
		/// Use additionalFlags if you need to set additional flags in the temporary hknpConstraint.
		virtual void addImmediateConstraint(
			hknpBodyId bodyIdA, hknpBodyId bodyIdB,
			hkpConstraintData* data, hknpImmediateConstraintId immId, hkUint8 additionalFlags );

		/// Inlined wrapper function, calls addImmediateConstraint() with no constraint ID or additional flags.
		HK_FORCE_INLINE void addImmediateConstraint( hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data )
		{
			addImmediateConstraint( bodyIdA, bodyIdB, data, hknpImmediateConstraintId::invalid(), 0 );
		}

#else

		hknpSolverData() {}

#endif

	public:

		hknpConstraintSolverJacobianGrid	m_jacMovingGrid;
		hknpConstraintSolverJacobianGrid	m_jacFixedGrid;

	#if defined(HK_PLATFORM_HAS_SPU)
		/// Grids with the jacobians produced by PPU only caches.
		hknpConstraintSolverJacobianGrid	m_jacMovingPpuGrid;
		hknpConstraintSolverJacobianGrid	m_jacFixedPpuGrid;
	#endif

		hknpConstraintSolverJacobianGrid	m_jacConstraintsGrid;
		hknpLiveJacobianInfoGrid			m_liveJacInfoGrid;
		hknpConstraintSolverJacobianStream	m_contactSolverTempsStream;

		/// Thread specific data. The inline capacity must be equal to the maximum number of SPUs
		/// (hkSpuThreadPool::MAX_NUM_SPUS) to avoid extra DMAs.
		hkInplaceArray<ThreadData,6> m_threadData;

		hknpSimulationContext* m_simulationContext;
};


/// A solver data configured for a single threaded simulation.
class hknpSingleThreadedSolverData : public hknpSolverData
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSingleThreadedSolverData );

		hknpSingleThreadedSolverData( hkThreadLocalBlockStreamAllocator* allocator );
};


#endif // HKNP_SOLVER_DATA_H

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
