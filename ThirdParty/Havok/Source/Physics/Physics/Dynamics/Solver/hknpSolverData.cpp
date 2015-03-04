/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>
#include <Physics/Physics/Dynamics/Solver/MxJacobianSorter/hknpMxJacobianSorter.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverUtil.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolverSetup.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverSetup.h>
#include <Physics/Physics/Dynamics/World/Commands/hknpInternalCommands.h>

#include <Common/Base/Types/hkSignalSlots.h>


hknpSolverData::hknpSolverData( hkThreadLocalBlockStreamAllocator* allocator, int numThreads, int numCells, int numLinks )
{
	const bool zeroNewBlocks     = true;
	const bool dontZeroNewBlocks = false;

	m_threadData.setSize( numThreads );
	for( int i = 0; i < numThreads; i++ )
	{
		ThreadData& td = m_threadData[i];
		td.m_jacMovingStream.initBlockStream( allocator, zeroNewBlocks);
		td.m_jacFixedStream.initBlockStream( allocator, zeroNewBlocks );
		td.m_liveJacInfoStream.initBlockStream( allocator, dontZeroNewBlocks );
		td.m_activePairStream.initBlockStream( allocator, dontZeroNewBlocks );
		td.m_jacConstraintsStream.initBlockStream( allocator, zeroNewBlocks );
		// We do not zero temp stream because it is faster to zero during the first iteration of the solver.
		td.m_solverTempsStream.initBlockStream( allocator, dontZeroNewBlocks );
	}

	m_jacMovingGrid.setSize( numLinks);
	m_jacFixedGrid.setSize( numCells );
	m_liveJacInfoGrid.setSize( numLinks );
	m_jacConstraintsGrid.setSize( numLinks );
	m_contactSolverTempsStream.initBlockStream( allocator, dontZeroNewBlocks );

#if defined(HK_PLATFORM_HAS_SPU)
	m_jacMovingPpuGrid.setSize( numLinks );
	m_jacFixedPpuGrid.setSize( numCells );
#endif
}


hknpSingleThreadedSolverData::hknpSingleThreadedSolverData( hkThreadLocalBlockStreamAllocator* allocator )
	: hknpSolverData( allocator, 1, 1, 1 )
{

}


void hknpSolverData::clear( hkThreadLocalBlockStreamAllocator* allocator )
{
	int numThreads = m_threadData.getSize();
	for( int i = 0; i < numThreads; i++ )
	{
		ThreadData& td = m_threadData[i];
		td.m_jacMovingStream.clear( allocator );
		td.m_jacFixedStream.clear( allocator );
		td.m_liveJacInfoStream.clear( allocator );
		td.m_activePairStream.clear( allocator );
		td.m_jacConstraintsStream.clear( allocator );
		td.m_solverTempsStream.clear( allocator );
	}
	m_contactSolverTempsStream.clear( allocator );
}


void hknpSolverData::addImmediateContactConstraint(
	hknpBodyId bodyIdA, hknpBodyId bodyIdB,
	const hknpManifold& manifold,
	const hknpMaterial* childMaterialA, const hknpMaterial* childMaterialB,
	const SimpleMaterial& material,
	hknpMxContactJacobian** mxJacOut, int* mxIdOut )
{
	HK_ON_DEBUG( manifold.checkConsistency() );

	int currentThreadId				= m_simulationContext->getCurrentThreadNumber();
	hknpSimulationThreadContext* tl	= m_simulationContext->getThreadContext( currentThreadId );
	ThreadData* HK_RESTRICT td		= &m_threadData[currentThreadId];

	hknpWorld* world = tl->m_world;


	//
	// Check if the constraint should be added
	//
	const int cellA = world->m_bodyManager.getCellIndex( bodyIdA );
	const int cellB = world->m_bodyManager.getCellIndex( bodyIdB );

	// Only add the constraint if both bodies are not deactivated.
	if (cellA == HKNP_INVALID_CELL_IDX && cellB == HKNP_INVALID_CELL_IDX)
	{
		return;
	}

	const int linkIdx = world->m_spaceSplitter->getLinkIdx(cellA, cellB);

	//
	// Create the Jacobian in the stream
	//
	hknpConstraintSolverJacobianWriter jacWriter; jacWriter.setToEndOfStream( tl->m_tempAllocator, &td->m_jacMovingStream );

	hknpConstraintSolverJacobianRange2 jacobianRange;
	jacobianRange.initRange( hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER,
		hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS | hknpConstraintSolverJacobianRange2::SOLVER_TEMPS );
	jacobianRange.setStartPoint( &jacWriter );

	hknpMxJacobianSorter sorter( &jacWriter );

	const hknpBody* bodyA = &world->getSimulatedBody( bodyIdA );
	const hknpBody* bodyB = &world->getSimulatedBody( bodyIdB );

	const hknpMotion* motionA = &world->getMotion( bodyA->m_motionId );
	const hknpMotion* motionB = &world->getMotion( bodyB->m_motionId );

	hkUint32 bodyHash = sorter.calcBodyIdsHashCode( *bodyA, *bodyB );

	hknpMxContactJacobian* mxJac;
	int mxJacIdx = sorter.getJacobianLocation( bodyHash, &mxJac, &mxJac );

	hknpContactSolverSetup::BuildContactInput input;
	{
		input.m_friction = material.m_friction;
		input.m_maxImpulse = material.m_maxImpulse;
		input.m_fractionOfClippedImpulseToApply = material.m_fractionOfClippedImpulseToApply;
	}

	hknpContactSolverSetup::buildContactJacobian(
		manifold, world->m_solverInfo, input,
		bodyA, motionA, childMaterialA,
		bodyB, motionB, childMaterialB,
		mxJac, mxJac, mxJacIdx );

	if ( mxJacOut )
	{
		mxJacOut[0] = mxJac;
		mxIdOut[0]  = mxJacIdx;
	}

	jacobianRange.setEndPoint( &jacWriter );

	//
	// Add the resulting range to the grid
	//
	if ( m_threadData.getSize()==1 )
	{
		m_jacMovingGrid.addRange( jacWriter, linkIdx, jacobianRange );
	}
	else
	{
		void* buffer = tl->m_commandWriter->m_writer.reserve( sizeof(hknpAddConstraintRangeCommand));
		hknpAddConstraintRangeCommand* HK_RESTRICT acr = new (buffer) hknpAddConstraintRangeCommand( bodyIdA );
		acr->m_grid = &m_jacMovingGrid;
		acr->m_linkIndex = linkIdx;
		acr->m_range = jacobianRange;

		tl->m_commandWriter->m_writer.advance( sizeof(hknpAddConstraintRangeCommand) );
	}

	jacWriter.finalize();
}


void hknpSolverData::addImmediateContactConstraintUsingJacobian(const hknpContactJacobian<1>& preComputedJacobianIn)
{
	// Get thread data
	const int currentThreadId		= m_simulationContext->getCurrentThreadNumber();
	hknpSimulationThreadContext* tl	= m_simulationContext->getThreadContext( currentThreadId );
	ThreadData* HK_RESTRICT td		= &m_threadData[currentThreadId];
	hknpWorld* world				= tl->m_world;

	// Check if all pre-computed stuff is valid!
	const hknpContactJacobianTypes::HeaderData& manifold = preComputedJacobianIn.m_manifoldData[0];
	HK_ASSERT(0x392e4509, manifold.m_cellIndexA == world->m_bodyManager.getCellIndex(manifold.m_bodyIdA));
	HK_ASSERT(0x11432572, manifold.m_cellIndexB == world->m_bodyManager.getCellIndex(manifold.m_bodyIdB));
	HK_ASSERT(0x7dc82d9d, (manifold.m_cellIndexA != HKNP_INVALID_CELL_IDX) || (manifold.m_cellIndexB != HKNP_INVALID_CELL_IDX));
	const int linkIdx = world->m_spaceSplitter->getLinkIdx(manifold.m_cellIndexA, manifold.m_cellIndexB);

	// Create the Jacobian in the stream
	hknpConstraintSolverJacobianWriter jacWriter;
	jacWriter.setToEndOfStream(tl->m_tempAllocator, &td->m_jacMovingStream);

	hknpConstraintSolverJacobianRange2 jacobianRange;
	jacobianRange.initRange(hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER,
		hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS | hknpConstraintSolverJacobianRange2::SOLVER_TEMPS );
	jacobianRange.setStartPoint(&jacWriter);

	// Compute body hash
	const hknpBody* bodyA = &world->getSimulatedBody(manifold.m_bodyIdA);
	const hknpBody* bodyB = &world->getSimulatedBody(manifold.m_bodyIdB);
	const hkUint32 bodyHash = hknpMxJacobianSorter::calcBodyIdsHashCode(*bodyA, *bodyB);

	// Compute the index of the manifold where we can copy our Jacobian. Currently returns 0 as the MxJacobian is new!
	hknpMxContactJacobian* mxJac;
	hknpMxJacobianSorter sorter(&jacWriter);
	int mxJacIdx = sorter.getJacobianLocation(bodyHash, &mxJac, &mxJac);
	HK_ASSERT(0x5adaf997, mxJacIdx == 0);

	// Copy our Jacobian
	mxJac->copyFrom(preComputedJacobianIn, 0, mxJacIdx);

	
	// Need to set proper pointers to mxJac into the collision cache if any
/*	for (int k = HK_NP_NUM_MX_JACOBIANS - 1; k >= 0; k--)
	{
		mxJac->m_manifoldData[k].m_manifoldSolverInfo->m_contactJacobian = mxJac;
	}*/

	jacobianRange.setEndPoint(&jacWriter);

	// Add the resulting range to the grid
	if ( m_threadData.getSize() == 1 )
	{
		m_jacMovingGrid.addRange(jacWriter, linkIdx, jacobianRange);
	}
	else
	{
		void* buffer = tl->m_commandWriter->m_writer.reserve(sizeof(hknpAddConstraintRangeCommand));

		hknpAddConstraintRangeCommand* HK_RESTRICT acr = new (buffer) hknpAddConstraintRangeCommand(manifold.m_bodyIdA);
		acr->m_grid			= &m_jacMovingGrid;
		acr->m_linkIndex	= linkIdx;
		acr->m_range		= jacobianRange;

		tl->m_commandWriter->m_writer.advance( sizeof(hknpAddConstraintRangeCommand) );
	}

	jacWriter.finalize();
}


void hknpSolverData::addImmediateContactConstraintUsingJacobianBatch(const hknpMxContactJacobian& preComputedJacobianIn)
{
	// Get thread data
	const int currentThreadId		= m_simulationContext->getCurrentThreadNumber();
	hknpSimulationThreadContext* tl	= m_simulationContext->getThreadContext( currentThreadId );
	ThreadData* HK_RESTRICT td		= &m_threadData[currentThreadId];
	hknpWorld* world				= tl->m_world;

	// Check if all pre-computed stuff is valid!
	const hknpContactJacobianTypes::HeaderData& manifold = preComputedJacobianIn.m_manifoldData[0];
	HK_ASSERT(0x680f2973, manifold.m_cellIndexA == world->m_bodyManager.getCellIndex(manifold.m_bodyIdA));
	HK_ASSERT(0x5d2fe3d, manifold.m_cellIndexB == world->m_bodyManager.getCellIndex(manifold.m_bodyIdB));
	HK_ASSERT(0x45d05bae, (manifold.m_cellIndexA != HKNP_INVALID_CELL_IDX) || (manifold.m_cellIndexB != HKNP_INVALID_CELL_IDX));
	const int linkIdx = world->m_spaceSplitter->getLinkIdx(manifold.m_cellIndexA, manifold.m_cellIndexB);

	// Create the Jacobian in the stream
	hknpConstraintSolverJacobianWriter jacWriter;
	jacWriter.setToEndOfStream(tl->m_tempAllocator, &td->m_jacMovingStream);

	hknpConstraintSolverJacobianRange2 jacobianRange;
	jacobianRange.initRange(hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER,
		hknpConstraintSolverJacobianRange2::LIVE_JACOBIANS | hknpConstraintSolverJacobianRange2::SOLVER_TEMPS );
	jacobianRange.setStartPoint(&jacWriter);

	// Compute body hash
	const hknpBody* bodyA = &world->getSimulatedBody(manifold.m_bodyIdA);
	const hknpBody* bodyB = &world->getSimulatedBody(manifold.m_bodyIdB);
	const hkUint32 bodyHash = hknpMxJacobianSorter::calcBodyIdsHashCode(*bodyA, *bodyB);

	// Compute the index of the manifold where we can copy our Jacobian. Currently returns 0 as the MxJacobian is new!
	hknpMxContactJacobian* mxJac;
	hknpMxJacobianSorter sorter(&jacWriter);
	HK_ON_DEBUG(const int mxJacIdx = )sorter.getJacobianLocation(bodyHash, &mxJac, &mxJac);
	HK_ASSERT(0x42cdcd20, mxJacIdx == 0);

	// Copy our Jacobian
	*mxJac = preComputedJacobianIn;
	jacobianRange.setEndPoint(&jacWriter);

	// Add the resulting range to the grid
	if ( m_threadData.getSize() == 1 )
	{
		m_jacMovingGrid.addRange(jacWriter, linkIdx, jacobianRange);
	}
	else
	{
		void* buffer = tl->m_commandWriter->m_writer.reserve(sizeof(hknpAddConstraintRangeCommand));

		hknpAddConstraintRangeCommand* HK_RESTRICT acr = new (buffer) hknpAddConstraintRangeCommand(manifold.m_bodyIdA);
		acr->m_grid			= &m_jacMovingGrid;
		acr->m_linkIndex	= linkIdx;
		acr->m_range		= jacobianRange;

		tl->m_commandWriter->m_writer.advance( sizeof(hknpAddConstraintRangeCommand));
	}

	jacWriter.finalize();
}


void hknpSolverData::addImmediateConstraint(
	hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data,
	hknpImmediateConstraintId immId, hkUint8 additionalFlags )
{
	int currentThreadId				= m_simulationContext->getCurrentThreadNumber();
	hknpSimulationThreadContext* tl	= m_simulationContext->getThreadContext( currentThreadId );
	ThreadData* HK_RESTRICT td		= &m_threadData[currentThreadId];

	hknpWorld* world = tl->m_world;

	// Create the constraint
	hknpConstraint constraint;
	if (immId == hknpImmediateConstraintId::invalid())
	{
		constraint.initImmediate( bodyIdA, bodyIdB, data );
	}
	else
	{
		constraint.initImmediateExportable( bodyIdA, bodyIdB, data, immId );
	}
	constraint.m_flags.orWith(additionalFlags);
	HK_ASSERT2( 0xf0edc89a, !constraint.hasRuntime(),
		"Your constraint requires runtime data (caches). It cannot be added as an immediate constraint." );

	// Call the setup function
	hknpConstraintAtomSolverSetup::setupConstraintMt(
		*tl, world, &constraint,
		m_jacConstraintsGrid, &td->m_jacConstraintsStream, &td->m_solverTempsStream,
		world->isDeactivationEnabled() ? &td->m_activePairStream : HK_NULL );

	world->m_signals.m_immediateConstraintAdded.fire( world, &constraint );
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
