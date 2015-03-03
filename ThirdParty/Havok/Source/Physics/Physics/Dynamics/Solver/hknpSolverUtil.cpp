/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>
#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>
#include <Common/Base/Thread/SimpleScheduler/hkSimpleScheduler.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Physics/Physics/Dynamics/Solver/hknpSolverUtil.h>

#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>

#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobianUtil.h>
#include <Physics/Internal/Collide/Agent/ProcessCollision2DFast/hknpCollision2DFastProcessUtil.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverSetup.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactSolverLog.h>
#include <Physics/Internal/Dynamics/Solver/Contact/hknpContactJacobian.h>
#include <Physics/Internal/Dynamics/Solver/Integrator/hknpSolverIntegrator.h>
#include <Physics/Internal/Collide/NarrowPhase/LiveJacobian/hknpLiveJacobianUtil.h>

#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverStepInfo.h>
#include <Physics/Internal/Dynamics/Solver/Scheduler/hknpSolverScheduler.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>

#if defined HKNP_ENABLE_SOLVER_LOG
extern hkOstream* g_npDebugOstream;
#endif


void hknpSolverUtil::allocateSolverTemps(
	const hknpSimulationThreadContext& tl,
	const hknpConstraintSolverSchedulerGridInfo* jacGridInfos, hknpConstraintSolverJacobianGrid** jacGrids, const int jacGridCount,
	hknpConstraintSolverJacobianStream* HK_RESTRICT solverTempsStream, hknpConstraintSolverJacobianStream* HK_RESTRICT contactSolverTempsStream)
{
	hknpWorld* world = tl.m_world;
	hknpConstraintSolver** solvers = world->getModifierManager()->m_constraintSolvers;

	hknpConstraintSolverJacobianWriter sharedTempsWriter;
	sharedTempsWriter.setToEndOfStream(tl.m_tempAllocator, solverTempsStream);

	hknpConstraintSolverJacobianWriter contactTempsWriter;
	contactTempsWriter.setToEndOfStream(tl.m_tempAllocator, contactSolverTempsStream);

	for (int gi = 0; gi < jacGridCount; ++gi)
	{
		hknpConstraintSolverJacobianGrid& grid = *(jacGrids[gi]);

		for(int i=0; i< grid.m_entries.getSize(); i++)
		{
			hknpConstraintSolverJacobianRange2* entry = &(grid.m_entries[i]);

			if (!entry->isEmpty())
			{
				do
				{
					if (entry->m_flags.anyIsSet(hknpConstraintSolverJacobianRange2::SOLVER_TEMPS))
					{
						HK_ASSERT2(0xef459818, entry->m_solverTempRange.isEmpty(), "Solver temps were already allocated (probably during setup).");
						hknpConstraintSolverId::Type solverId = entry->m_solverId.value();
						// The contact solver has a special writer for it's temps that keeps them 16byte aligned and maximizes performance.
						// Mixing it with other solver temps would destory the alignment guarantees.
						hknpConstraintSolverJacobianWriter* tempsWriter = (solverId == hknpConstraintSolverType::CONTACT_CONSTRAINT_SOLVER ? &contactTempsWriter : &sharedTempsWriter);
						solvers[solverId]->allocateTemps(*entry, entry->m_solverTempRange, tempsWriter);
					}

					entry = (hknpConstraintSolverJacobianRange2*) entry->m_next;

				} while (entry != HK_NULL);
			}
			else
			{
				HK_ASSERT2(0xef5f13a5, entry->m_next == HK_NULL, "Empty entries should not be linked.");
			}
		}
	}

	contactTempsWriter.finalize();
	sharedTempsWriter.finalize();
}


#if defined HKNP_ENABLE_SOLVER_LOG
extern hkOstream* g_npDebugOstream;
#endif

void hknpSolverUtil::solveSt(
	hknpSimulationContext& stepData, hknpSimulationThreadContext& tl,
	const hknpConstraintSolverSchedulerGridInfo* jacGridInfos,
	hknpConstraintSolverJacobianGrid** jacGrids,
	hknpLiveJacobianInfoStream* liveJacInfos,
	hknpSolverVelocity* HK_RESTRICT solverVel, hknpSolverSumVelocity* HK_RESTRICT solverSumVel, int numSolverVel,
	hknpConstraintSolverJacobianStream* solverTempsStream, hknpConstraintSolverJacobianStream* contactSolverTempsStream)
{
	hknpWorld* world = tl.m_world;
	const hknpSolverInfo& solverInfo = world->m_solverInfo;
	const int jacGridCount = hknpJacobianGridType::NUM_TYPES;
	hknpConstraintSolver** solvers = world->getModifierManager()->m_constraintSolvers;

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& mStream = hkMonitorStream::getInstance();
#endif
	HK_TIMER_BEGIN_LIST2(mStream, "SolverUtil", "init" );

	//
	// Let solvers that need it allocate their temps.
	//
	allocateSolverTemps(tl, jacGridInfos, jacGrids, jacGridCount, solverTempsStream, contactSolverTempsStream);

#if defined HKNP_ENABLE_SOLVER_LOG
	if (!g_npDebugOstream){	g_npDebugOstream = new hkOfstream("npSolverLog.txt");	}
	int numSolverSteps = solverInfo.m_numSteps;
	hkReal invIntegrateVelocityFactor = solverInfo.m_invIntegrateVelocityFactor(0);
	{
		static int frameCount = 0;
		frameCount++;
		g_npDebugOstream->printf("****** Frame %3d *******\r\n\r\n", frameCount);
		g_npDebugOstream->printf("All Jacobians:\r\n\r\n", -1);

		hknpConstraintSolverJacobianReader jacReader;

		jacReader.setToStartOfStream( contactJacobians );		hknpContactSolverLog::debugPrintJacobians(solverVel, solverSumVel, jacReader);
		jacReader.setToStartOfStream( jacobiansHQ );	hknpContactSolverLog::debugPrintJacobians(solverVel, solverSumVel, jacReader);
		g_npDebugOstream->printf("All Motions at start:\r\n\r\n", -1);
		hknpContactSolverLog::debugPrintSolverVelocities(solverVel, solverSumVel, numSolverVel, 0, numSolverSteps, invIntegrateVelocityFactor);
	}
#endif

	// Sort the grids by ascending processing priority
	hkInplaceArray<hkUint8,jacGridCount> sortedGridIndices(jacGridCount);
	sortJacGrids(jacGridInfos, jacGridCount, sortedGridIndices);


	hknpSolverStepInfo solverStepInfo;
	solverStepInfo.m_solverInfo = &solverInfo;
	solverStepInfo.m_solverVelocities = solverVel;
	solverStepInfo.m_solverSumVelocities = solverSumVel;

	hknpSolverStep solverStep;
	hknpIdxRange dummyMotionEntry(0, numSolverVel);


	for (int stepIndex=0; stepIndex < solverInfo.m_numSteps; stepIndex++)
	{
		//
		// Iterate over all contactJacobians
		//
		HKNP_ON_SOLVER_LOG( g_npDebugOstream->printf("Impulses applied in %d/%d step:\r\n\r\n", stepIndex+1, solverInfo.m_numSteps) );

		if ( stepIndex>0 && !liveJacInfos->isEmpty())
		{
			hknpLiveJacobianInfoRange marker; marker.setEntireStream(liveJacInfos);

			HK_TIMER_SPLIT_LIST2( mStream, "generateLiveJacobians");
			// Generate motion transforms or something
			HKNP_ON_SOLVER_LOG( g_npDebugOstream->printf("___________________________________\r\n") );
			HKNP_ON_SOLVER_LOG( g_npDebugOstream->printf("Rebuilding Jacobians for %d/%d step:\r\n\r\n", stepIndex+2, solverInfo.m_numSteps) );

			// Process live contactJacobians: do collision detection and rebuild contactJacobians
			hknpLiveJacobianUtil::generateLiveJacobians(tl, solverStep, solverSumVel, solverSumVel, &marker);
		}


		for (int microStep= 0; microStep < solverInfo.m_numMicroSteps; microStep++)
		{
			solverStep.init( solverInfo, stepIndex, microStep);

			for (hkUint8 gpi = 0; gpi < jacGridCount; ++gpi)
			{
				int gridIndex = sortedGridIndices[gpi];
				const hknpConstraintSolverJacobianRange2* iJacEntry = & (jacGrids[gridIndex]->m_entries[0]);

				do
				{
					if (!iJacEntry->isEmpty())
					{
						switch(stepIndex)
						{
						case 0: HK_TIMER_SPLIT_LIST2( mStream, "SolveJacobians0"); break;
						case 1: HK_TIMER_SPLIT_LIST2( mStream, "SolveJacobians1"); break;
						case 3: HK_TIMER_SPLIT_LIST2( mStream, "SolveJacobians3"); break;
						default: HK_TIMER_SPLIT_LIST2( mStream, "SolveJacobians2"); break;
						}


						hknpConstraintSolver* solver = solvers[iJacEntry->m_solverId.value()];
						solver->solveJacobians( tl, solverStepInfo, solverStep, iJacEntry, solverVel, solverVel, dummyMotionEntry, dummyMotionEntry );
					}

					iJacEntry = (hknpConstraintSolverJacobianRange2*) iJacEntry->m_next;

				} while (iJacEntry != HK_NULL);
			}

		}	// stepIndex < numMicroSteps

		HKNP_ON_SOLVER_LOG( g_npDebugOstream->printf("\r\n") );

		// sub integrate
		if ( stepIndex < solverInfo.m_numSteps-1 )
		{
			HK_TIMER_SPLIT_LIST2( mStream, "SubIntegrate");
			hknpSolverIntegrator::subIntegrate( tl, solverStep, solverVel, solverSumVel, numSolverVel, solverInfo, world->getMotionPropertiesLibrary()->getBuffer() );

			HKNP_ON_SOLVER_LOG( g_npDebugOstream->printf("All Motions after %d/%d step:\r\n\r\n", stepIndex+1, solverInfo.m_numSteps) );
			HKNP_ON_SOLVER_LOG( hknpContactSolverLog::debugPrintSolverVelocities(solverVel, solverSumVel, numSolverVel, stepIndex+1, numSolverSteps, invIntegrateVelocityFactor) );
		}  // i < solverInfo.m_numSteps-1
	} // for m_solverInfo.m_numSteps

	HK_TIMER_SPLIT_LIST2( mStream, "SubIntegrateLast");

	hknpSolverStepInfo solverManager;
	solverManager.m_solverInfo = &solverInfo;
	solverManager.m_spaceSplitter = world->m_spaceSplitter;
	solverManager.m_intSpaceUtil = &world->m_intSpaceUtil;
	solverManager.m_motionGridEntriesStart = HK_NULL; // no motion grid here
	solverManager.m_simulationContext = &stepData;
	solverManager.m_solverData = HK_NULL;

	hknpMotion* motions = world->m_motionManager.accessMotionBuffer();
	hknpDeactivationManager* deactivationMgr = world->m_deactivationManager;


	if (world->isDeactivationEnabled())
	{
		const int solverVelOkToDeactivateOffset_1 = 0;

		hknpSolverIntegrator::subIntegrateLastStep(
			solverStep, solverVel, solverSumVel, numSolverVel, motions, deactivationMgr->getAllDeactivationStates(), world->getMotionPropertiesLibrary()->getBuffer(),
			&solverManager, tl,
			tl.m_spaceSplitterData,
			tl.m_deactivationData->m_solverVelOkToDeactivate.accessWords(), solverVelOkToDeactivateOffset_1, tl.m_deactivationData->m_solverVelOkToDeactivate.getSize() );
	}
	else
	{
		hknpSolverIntegrator::subIntegrateLastStep(
			solverStep, solverVel, solverSumVel, numSolverVel, motions, deactivationMgr->getAllDeactivationStates(), world->getMotionPropertiesLibrary()->getBuffer(),
			&solverManager, tl,
			tl.m_spaceSplitterData,
			HK_NULL, 0, 0 );
	}

	HKNP_ON_SOLVER_LOG( g_npDebugOstream->printf("All Motions after %d/%d step:\r\n\r\n", numSolverSteps, solverInfo.m_numSteps) );
	HKNP_ON_SOLVER_LOG( hknpContactSolverLog::debugPrintSolverVelocities(solverVel, solverSumVel, numSolverVel, numSolverSteps, numSolverSteps, invIntegrateVelocityFactor); )

	HK_TIMER_END_LIST2( mStream );
}


void hknpSolverUtil::calculateProjectedPointVelocitiesUsingIntegratedVelocities(
	const hkcdManifold4& manifold, const hknpMotion* motionA, const hknpMotion* motionB, hkVector4* HK_RESTRICT projectedPointVelocitiesOut )
{
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> position; position.moveLoad( &manifold.m_positions[0] );

	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> comA; comA.setVector(motionA->getCenterOfMassInWorld());
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> comB; comB.setVector(motionB->getCenterOfMassInWorld());
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> massCenterRelative0; massCenterRelative0.setSub(position, comA);
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> massCenterRelative1; massCenterRelative1.setSub(position, comB);

	hkVector4 angVelA; angVelA._setRotatedDir( motionA->m_orientation, motionA->m_previousStepAngularVelocity );
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> angA; angA.setVector(angVelA);

	hkVector4 angVelB; angVelB._setRotatedDir( motionB->m_orientation, motionB->m_previousStepAngularVelocity );
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> angB; angB.setVector(angVelB);

	hkVector4 normalInWorld = manifold.m_normal;
	hkVector4 deltaLin;	deltaLin.setSub(motionA->m_previousStepLinearVelocity, motionB->m_previousStepLinearVelocity); // linear part
	hkSimdReal dotLin = normalInWorld.dot<3>(deltaLin);

	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> tmp0; tmp0.setCross(angA, massCenterRelative0); 
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> tmp1; tmp1.setCross(angB, massCenterRelative1);

	tmp0.sub(tmp1);
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> nrm; nrm.setVector(normalInWorld);
	hkMxReal<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> deltaRhsMx;	tmp0.dot<3>(nrm, deltaRhsMx); // irrelevant points zeroed here
	hkVector4 deltaRhs; deltaRhsMx.storePacked(deltaRhs);
	hkVector4 dotLin4; dotLin4.setAll(dotLin);
	deltaRhs.add(dotLin4);
	projectedPointVelocitiesOut[0] = deltaRhs;
}

void hknpSolverUtil::calculateProjectedPointVelocities(
	const hkcdManifold4& manifold, const hknpMotion* motionA, const hknpMotion* motionB, hkVector4* HK_RESTRICT projectedPointVelocitiesOut )
{
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> position; position.moveLoad( &manifold.m_positions[0] );

	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> comA; comA.setVector(motionA->getCenterOfMassInWorld());
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> comB; comB.setVector(motionB->getCenterOfMassInWorld());
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> massCenterRelative0; massCenterRelative0.setSub(position, comA);
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> massCenterRelative1; massCenterRelative1.setSub(position, comB);

	hkVector4 angVelA; angVelA._setRotatedDir( motionA->m_orientation, motionA->m_angularVelocity );
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> angA; angA.setVector(angVelA);

	hkVector4 angVelB; angVelB._setRotatedDir( motionB->m_orientation, motionB->m_angularVelocity );
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> angB; angB.setVector(angVelB);

	hkVector4 normalInWorld = manifold.m_normal;
	hkVector4 deltaLin;	deltaLin.setSub(motionA->m_linearVelocity, motionB->m_linearVelocity); // linear part
	hkSimdReal dotLin = normalInWorld.dot<3>(deltaLin);


	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> tmp0; tmp0.setCross(angA, massCenterRelative0); 
	hkMxVector<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> tmp1; tmp1.setCross(angB, massCenterRelative1);

	tmp0.sub(tmp1);
	hkMxSingle<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> nrm; nrm.setVector(normalInWorld);
	hkMxReal<hknpMxContactJacobian::NUM_POINTS_PER_MANIFOLD> deltaRhsMx;	tmp0.dot<3>(nrm, deltaRhsMx); // irrelevant points zeroed here
	hkVector4 deltaRhs; deltaRhsMx.storePacked(deltaRhs);
	hkVector4 dotLin4; dotLin4.setAll(dotLin);
	deltaRhs.add(dotLin4);
	projectedPointVelocitiesOut[0] = deltaRhs;
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
