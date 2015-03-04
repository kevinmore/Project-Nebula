/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_USER_SOLVER_UTILITY_H
#define HKNP_USER_SOLVER_UTILITY_H

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Spu/Dma/Utils/hkSpuDmaUtils.h>
#include <Common/Base/Types/hkHandle.h>
#include <Common/Base/Math/Vector/hkIntVector.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/World/Deactivation/hknpCollisionPair.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverStepInfo.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>



namespace hknpUserSolverUtil
{
	enum BodyEnum
	{
		BODY_A = 0, BODY_B = 1
	};

	//
	class Constraint
	{
	public:

		Constraint(int type, hknpBodyId bodyA, hknpBodyId bodyB) : m_type(type) { m_body[BODY_A] = bodyA; m_body[BODY_B] = bodyB; }
		virtual ~Constraint() {}

		int				m_type;
		hknpBodyId		m_body[2];
	};

	//
	struct Schema
	{
	public:

		Schema(int type) : m_type(type) {}

		int				m_type;

		int				m_velStreamPick;
		int				m_selfSize;

		hknpSolverId	m_solverVelIndex[2];
		hkQTransform	m_motionFrame[2];
	};

	//
	struct DispatchPrototype
	{
		static int		getSizeOfSchema	(const Constraint& constraint);

		static void		initSchema		(const Constraint& constraint, Schema& schema,
											hknpWorld& world, hknpBodyId bodyIdA, hknpBodyId bodyIdB, const hknpMotion& motionA, const hknpMotion& motionB);

		static void		solve			(const Schema& schema, hknpWorld& world, const hknpSolverInfo& solverInfo, struct MotionState* HK_RESTRICT motionStateA, struct MotionState* HK_RESTRICT motionStateB);
	};


#ifdef HK_DEBUG
	namespace
	{
		void checkLegalConstraintActiveState(const hknpBody& bodyA, const hknpBody& bodyB)
		{
			if (bodyA.isDynamic() && bodyB.isDynamic())
			{
				bool bothActivated = bodyA.isActive() && bodyB.isActive();
				bool bothDeactivated = !bodyA.isActive() && !bodyB.isActive();
				HK_ASSERT2(0xef1113a7, bothActivated || bothDeactivated, "2 Constrained bodies should have the same active state");
			}
		}
	}
#endif

	//
	template <typename DispatchT>
	/*HK_FORCE_INLINE*/ void writeSchemaForConstraint(hknpWorld& world, Constraint& constraint,
		hknpConstraintSolverJacobianWriter& schemaWriter, hknpCdPairWriter* activePairWriter)
	{
		const hknpBody& bodyA = world.getBody( constraint.m_body[0] );
		const hknpBody& bodyB = world.getBody( constraint.m_body[1] );

		const hknpMotion& motionA = world.getMotion( bodyA.m_motionId );
		const hknpMotion& motionB = world.getMotion( bodyB.m_motionId );

		//
		// If any of the bodies is not active, skip.
		//
		if ( motionA.m_solverId.isValid() && motionB.m_solverId.isValid() )
		{
			//
			// Add as active pair for deactivation.
			//
			if (activePairWriter != HK_NULL && !bodyA.isStatic() && !bodyB.isStatic())
			{
				hknpCollisionPair* HK_RESTRICT activePair = activePairWriter->reserve(sizeof(hknpCollisionPair)) ;
				activePair->m_cell[0] = motionA.m_cellIndex;
				activePair->m_cell[1] = motionB.m_cellIndex;
				activePair->m_id[0] = motionA.m_solverId;
				activePair->m_id[1] = motionB.m_solverId;
				activePairWriter->advance(sizeof(hknpCollisionPair));
			}

			int size = DispatchT::getSizeOfSchema(constraint);
			Schema* schema = schemaWriter.reserve<Schema>(size);
			{
				schema->m_type = constraint.m_type;
				schema->m_selfSize = size;
				schema->m_velStreamPick = hknpSpaceSplitter::isLinkFlipped( motionA.m_cellIndex, motionB.m_cellIndex );

				schema->m_solverVelIndex[BODY_A] = motionA.m_solverId;
				schema->m_solverVelIndex[BODY_B] = motionB.m_solverId;

				schema->m_motionFrame[BODY_A].set(motionA.m_orientation, motionA.getCenterOfMassInWorld());
				schema->m_motionFrame[BODY_B].set(motionB.m_orientation, motionB.getCenterOfMassInWorld());

				DispatchT::initSchema(constraint, schema, world, constraint.m_body[0], constraint.m_body[1], motionA, motionB);
			}
			schemaWriter.advance( size );
		}
	}

	//
	template <typename DispatchT>
	void setupConstraintsST(hknpWorld* world, Constraint** instances, int numInstances,
		hknpConstraintSolverJacobianWriter& schemaWriter, hknpConstraintSolverJacobianWriter& solverTempsWriter,
		hknpCdPairWriter* activePairWriter)
	{
		for (int i = 0; i < numInstances; i++)
		{
			HK_ON_DEBUG( checkLegalConstraintActiveState(world->getBody(instances[i]->m_body[0]), world->getBody(instances[i]->m_body[1])) );

			hknpUserSolverUtil::writeSchemaForConstraint<DispatchT>(*world, *instances[i], schemaWriter, activePairWriter);
		}
	}

	//
	namespace
	{
		struct SortedConstraint
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS, SortedConstraint);

			HK_FORCE_INLINE SortedConstraint() {}
			HK_FORCE_INLINE SortedConstraint(int i, void* c) : m_linkIndex(i), m_constraint(c) {}

			int m_linkIndex;
			void* m_constraint;
		};

		class ByLinkIndex
		{
		public:

			HK_FORCE_INLINE hkBool32 operator() ( const SortedConstraint& a, const SortedConstraint& b ) { return ( a.m_linkIndex < b.m_linkIndex ); }
		};

		// Find constraint grid links.
		typedef hkArray<SortedConstraint> SortedConstraints;
	}

	//
	template <typename DispatchT>
	void setupConstraintsMT(const hknpSimulationThreadContext& tl, hknpWorld* world,
		hknpConstraintSolverType::Enum solverType, Constraint** constraints, int numConstraints,
		hknpConstraintSolverJacobianGrid& grid, hknpConstraintSolverJacobianStream* stream,
		hknpConstraintSolverJacobianStream* solverTempsStream,
		hknpCdPairStream* activePairStream)
	{
		SortedConstraints sortedConstraints;

		for (int i =0; i < numConstraints; i++)
		{
			Constraint* constraint = constraints[i];

			const int cellA = world->m_bodyManager.getCellIndex( constraint->m_body[0] );
			const int cellB = world->m_bodyManager.getCellIndex( constraint->m_body[1] );

			HK_ON_DEBUG( checkLegalConstraintActiveState(world->getBody(constraint->m_body[0]), world->getBody(constraint->m_body[1])) );

			// Only add the constraint if it is not deactivated.
			// This particular condition assumes that at least one of the bodies is dynamic and active
			//	and in that case, at least one has a valid cell index.
			// The situation where the 2 bodies are dynamic and only one of the is active is not legal
			//	and should be prevented by the code managing constraints.
			if (cellA != HKNP_INVALID_CELL_IDX || cellB != HKNP_INVALID_CELL_IDX)
			{
				const int linkIdx = world->m_spaceSplitter->getLinkIdx(cellA, cellB);
				sortedConstraints.pushBack(SortedConstraint(linkIdx, constraint));
			}
		}

		if (sortedConstraints.isEmpty())
		{
			return;
		}

		// sort into linkgrid and determine ranges
		hkAlgorithm::quickSort(sortedConstraints.begin(), sortedConstraints.getSize(), ByLinkIndex());
		numConstraints = sortedConstraints.getSize();

		// Do the rest.

		hknpConstraintSolverJacobianWriter jacConstraintWriter;
		jacConstraintWriter.setToEndOfStream( tl.m_tempAllocator, stream );

		hknpCdPairWriter activePairWriterHolder;
		hknpCdPairWriter* activePairWriter = HK_NULL;
		if (activePairStream != HK_NULL)
		{
			activePairWriterHolder.setToEndOfStream( tl.m_tempAllocator, activePairStream );
			activePairWriter = &activePairWriterHolder;
		}

		hknpConstraintSolverJacobianRange2 jacobianRange;
		jacobianRange.initRange( solverType, 0 );
		int currentLinkIndex = -1;

		for (int i =0; i < numConstraints; i++)
		{
			const SortedConstraint& sortedConstraint = sortedConstraints[i];

			if (sortedConstraint.m_linkIndex != currentLinkIndex)
			{
				if (currentLinkIndex != -1)
				{
					// Close current range
					jacobianRange.setEndPoint(&jacConstraintWriter);

					if (!jacobianRange.isEmpty())
					{
						hknpConstraintSolverJacobianRange2* presistentRange = jacConstraintWriter.reserve<hknpConstraintSolverJacobianRange2>();
						*presistentRange = jacobianRange;
						grid.m_entries[currentLinkIndex].appendPersistentRange(presistentRange);
						jacConstraintWriter.advance(sizeof(hknpConstraintSolverJacobianRange2));
					}
					else
					{
						HK_ASSERT2(0xef459817, jacobianRange.m_solverTempRange.isEmpty(), "Solver temps were allocated with no related jacobians");
					}
				}

				// Open new range
				jacobianRange.setStartPoint( &jacConstraintWriter);

				currentLinkIndex = sortedConstraint.m_linkIndex;
			}

			// Write the constraint
			{
				Constraint& constraint = * ((Constraint*) sortedConstraint.m_constraint);
				hknpUserSolverUtil::writeSchemaForConstraint<DispatchT>(
					*world, constraint, jacConstraintWriter, activePairWriter );
			}
		}

		// Close last range
		if (currentLinkIndex != -1)
		{
			// Close current range
			jacobianRange.setEndPoint( &jacConstraintWriter);

			if (!jacobianRange.isEmpty())
			{
				hknpConstraintSolverJacobianRange2* presistentRange = jacConstraintWriter.reserve<hknpConstraintSolverJacobianRange2>();
				*presistentRange = jacobianRange;
				grid.m_entries[currentLinkIndex].appendPersistentRange(presistentRange);
				jacConstraintWriter.advance(sizeof(hknpConstraintSolverJacobianRange2));
			}
			else
			{
				HK_ASSERT2(0xef459817, jacobianRange.m_solverTempRange.isEmpty(), "Solver temps were allocated with no related jacobians");
			}
		}

		if (activePairWriter != HK_NULL)
		{
			activePairWriter->finalize();
		}
		jacConstraintWriter.finalize();
	}

	//
	template <typename DispatchT>
	void setupConstraints(hknpWorld* world, hknpSolverData* solverData, hknpConstraintSolverType::Enum solverType, Constraint** instances, int numInstances)
	{
		if ( instances )
		{
			hknpSimulationThreadContext& tl = *solverData->m_simulationContext->getThreadContext();
			int currentThreadId = solverData->m_simulationContext->getCurrentThreadNumber();
			hknpSolverData::ThreadData* threadData	= &solverData->m_threadData[currentThreadId];

			if ( solverData->m_jacConstraintsGrid.m_entries.getSize() > 1 )
			{
				hknpUserSolverUtil::setupConstraintsMT<DispatchT>(tl, world, solverType, instances, numInstances,
					solverData->m_jacConstraintsGrid, &threadData->m_jacConstraintsStream,
					&threadData->m_solverTempsStream, &threadData->m_activePairStream );
			}
			else
			{
				bool isDeactivationEnabled = world->isDeactivationEnabled();

				hknpConstraintSolverJacobianWriter schemaWriter; schemaWriter.setToEndOfStream( tl.m_tempAllocator, &threadData->m_jacConstraintsStream );
				hknpConstraintSolverJacobianWriter solverTempsWriter; solverTempsWriter.setToEndOfStream( tl.m_tempAllocator, &threadData->m_solverTempsStream );
				hknpCdPairWriter activePairWriter;
				if (isDeactivationEnabled)
				{
					activePairWriter.setToEndOfStream( tl.m_tempAllocator, &threadData->m_activePairStream  );
				}

				hknpConstraintSolverJacobianRange2 range;
				range.initRange( solverType, 0);
				range.setStartPoint( &schemaWriter );
				range.m_solverTempRange.setStartPoint( &solverTempsWriter );

				hknpUserSolverUtil::setupConstraintsST<DispatchT>(world, instances, numInstances,
					schemaWriter, solverTempsWriter, isDeactivationEnabled ? &activePairWriter : HK_NULL);

				range.setEndPoint( &schemaWriter );
				range.m_solverTempRange.setEndPoint( &solverTempsWriter );

				if (!range.isEmpty())
				{
					solverData->m_jacConstraintsGrid.addRange( schemaWriter, 0, range );
				}

				if (isDeactivationEnabled)
				{
					activePairWriter.finalize();
				}
				solverTempsWriter.finalize();
				schemaWriter.finalize();
			}
		}
	}



	//
	template <typename DispatchT, typename MotionStateT>
	static HK_FORCE_INLINE void solveJacobiansImpl(const hknpSimulationThreadContext* tl, const hknpSolverInfo& npInfo, const hknpSolverStep& solverStepIn,
		const hknpConstraintSolverJacobianRange* jacobians, const hkBlockStreamBase::Range* tempsRange,
		const hknpSolverSumVelocity* HK_RESTRICT  solverSumVelAStream, hknpSolverVelocity* HK_RESTRICT  solverVelAStream,
		const hknpSolverSumVelocity* HK_RESTRICT  solverSumVelBStream, hknpSolverVelocity* HK_RESTRICT  solverVelBStream)
	{
		hknpConstraintSolverJacobianReader  schemaReader;
		schemaReader.initSpu( HK_SPU_DMA_GROUP_STALL, 2, "SolverConstraintJacReader" );
		schemaReader.setToRange( jacobians );

		MotionStateT stateA;
		MotionStateT stateB;

		hkPadSpu<hknpSolverVelocity*> solverVel[2];
		solverVel[0] = solverVelAStream;
		solverVel[1] = solverVelBStream;

		hkPadSpu<const hknpSolverSumVelocity*> solverSumVel[2];
		solverSumVel[0] = solverSumVelAStream;
		solverSumVel[1] = solverSumVelBStream;

		for ( const Schema* schema = schemaReader.access<Schema>();
			schema != HK_NULL;
			schema = schemaReader.advanceAndAccessNext<Schema>(schema->m_selfSize) )
		{
			hknpSolverVelocity* HK_RESTRICT solverVelA = solverVel[schema->m_velStreamPick].val() + schema->m_solverVelIndex[BODY_A].value();
			hknpSolverVelocity* HK_RESTRICT solverVelB = solverVel[1-schema->m_velStreamPick].val() + schema->m_solverVelIndex[BODY_B].value();

			const hknpSolverSumVelocity* HK_RESTRICT solverSumVelA = solverSumVel[schema->m_velStreamPick].val() + schema->m_solverVelIndex[BODY_A].value();
			const hknpSolverSumVelocity* HK_RESTRICT solverSumVelB = solverSumVel[1-schema->m_velStreamPick].val() + schema->m_solverVelIndex[BODY_B].value();

			stateA.load( npInfo, solverStepIn, schema->m_motionFrame[BODY_A], solverSumVelA, solverVelA );
			stateB.load( npInfo, solverStepIn, schema->m_motionFrame[BODY_B], solverSumVelB, solverVelB );

			DispatchT::solve(schema, *tl->m_world, npInfo, &stateA, &stateB);

			stateA.store( npInfo, solverSumVelA, solverVelA );
			stateB.store( npInfo, solverSumVelB, solverVelB );
		}
		schemaReader.exitSpu();
	}


	//
	template <typename DispatchT, typename MotionStateT>
	HK_FORCE_INLINE void solveJacobians( const hknpSimulationThreadContext& tl, const hknpSolverStepInfo& stepInfo, const hknpSolverStep& solverStep,
										const hknpConstraintSolverJacobianRange2* jacobians,
										hknpSolverVelocity* HK_RESTRICT solverVelA, hknpSolverVelocity* HK_RESTRICT solverVelB,
										const hknpIdxRange& motionEntryA, const hknpIdxRange& motionEntryB)
	{

#if defined(HK_PLATFORM_SPU)
		int numSolverSumVelB = HK_NEXT_MULTIPLE_OF(4, motionEntryB.m_numElements);
		hknpSolverSumVelocity* solverSumVelBPpu = stepInfo.m_solverSumVelocities + motionEntryB.m_start;
		int transferSizeSumVelB = numSolverSumVelB * sizeof(hknpSolverSumVelocity);
		int allocSizeSumVelB = HK_NEXT_MULTIPLE_OF(128, transferSizeSumVelB);


		hknpSolverSumVelocity* solverSumVelBSpu = (hknpSolverSumVelocity*)hkSpuStack::getInstance().allocateStack(allocSizeSumVelB, "JacConstrSolverVelB");
		hkSpuDmaManager::getFromMainMemoryLarge( solverSumVelBSpu, solverSumVelBPpu, transferSizeSumVelB, hkSpuDmaManager::READ_WRITE );

		hknpSolverSumVelocity* solverSumVelB = solverSumVelBSpu;
		hknpSolverSumVelocity* solverSumVelA = solverSumVelBSpu;

		int numSolverSumVelA = 0;
		hknpSolverSumVelocity* solverSumVelAPpu = HK_NULL;
		int transferSizeSumVelA = 0;
		int allocSizeSumVelA = 0;
		hknpSolverSumVelocity* solverSumVelASpu = HK_NULL;

		if (solverVelA != solverVelB)
		{

			numSolverSumVelA = HK_NEXT_MULTIPLE_OF(4, motionEntryA.m_numElements);
			solverSumVelAPpu = stepInfo.m_solverSumVelocities + motionEntryA.m_start;
			transferSizeSumVelA = numSolverSumVelA * sizeof(hknpSolverSumVelocity);
			allocSizeSumVelA = HK_NEXT_MULTIPLE_OF(128, transferSizeSumVelA);

			solverSumVelASpu = (hknpSolverSumVelocity*)hkSpuStack::getInstance().allocateStack(allocSizeSumVelA, "JacConstrSolverVelA");
			hkSpuDmaManager::getFromMainMemoryLarge( solverSumVelASpu, solverSumVelAPpu, transferSizeSumVelA, hkSpuDmaManager::READ_WRITE );

			solverSumVelA = solverSumVelASpu;
		}

		hkSpuDmaManager::waitForAllDmaCompletion();

#else
		hknpSolverSumVelocity* solverSumVelA = stepInfo.m_solverSumVelocities + motionEntryA.m_start;
		hknpSolverSumVelocity* solverSumVelB = stepInfo.m_solverSumVelocities + motionEntryB.m_start;
#endif

#if 1
		solveJacobiansImpl<DispatchT, MotionStateT>(&tl, *stepInfo.m_solverInfo, solverStep,
				jacobians, &jacobians->m_solverTempRange,
				solverSumVelA, solverVelA,
				solverSumVelB, solverVelB);
#else
		solverSumVelA; solverSumVelB;
#endif

#if defined(HK_PLATFORM_SPU)
		hkSpuDmaManager::putToMainMemoryLarge( solverSumVelBPpu, solverSumVelBSpu, transferSizeSumVelB, hkSpuDmaManager::WRITE_BACK );
		hkSpuDmaManager::deferFinalChecksUntilWait(solverSumVelBPpu, solverSumVelBSpu, transferSizeSumVelB);

		if (solverVelA != solverVelB)
		{
			hkSpuDmaManager::putToMainMemoryLarge( solverSumVelAPpu, solverSumVelASpu, transferSizeSumVelA, hkSpuDmaManager::WRITE_BACK );
			hkSpuDmaManager::deferFinalChecksUntilWait( solverSumVelAPpu, solverSumVelASpu, transferSizeSumVelA);
		}

		hkSpuDmaManager::waitForAllDmaCompletion();

		hkSpuStack::getInstance().deallocateStack(solverSumVelBSpu, allocSizeSumVelB);

		if (solverVelA != solverVelB)
		{
			hkSpuStack::getInstance().deallocateStack(solverSumVelASpu, allocSizeSumVelA);
		}

#endif
	}

	//
	template <typename DispatchT, typename MotionStateT = struct hknpUserSolverUtil::MotionState>
	class ConstraintSolver : public hknpConstraintSolver
	{
	public:

		hknpConstraintSolverType::Enum m_solverType;

		//
		ConstraintSolver(hknpConstraintSolverType::Enum solverType)
			: m_solverType(solverType)
		{
		}

		//
		void setupConstraints(hknpWorld* world, hknpSolverData* solverData, Constraint** instances, int numInstances)
		{
			hknpUserSolverUtil::setupConstraints<DispatchT>(world, solverData, m_solverType, instances, numInstances);
		}

		//
		virtual void solveJacobians(const hknpSimulationThreadContext& tl, const hknpSolverStepInfo& stepInfo, const hknpSolverStep& solverStep,
			const hknpConstraintSolverJacobianRange2* jacobians,
			hknpSolverVelocity* HK_RESTRICT solverVelA, hknpSolverVelocity* HK_RESTRICT solverVelB,
			const hknpIdxRange& motionEntryA, const hknpIdxRange& motionEntryB)
		{
			hknpUserSolverUtil::solveJacobians<DispatchT, MotionStateT>(tl, stepInfo, solverStep, jacobians, solverVelA, solverVelB, motionEntryA, motionEntryB);
		}
	};
}



namespace hknpUserSolverUtil
{
	//
	struct MotionState
	{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, MotionState );

		HK_FORCE_INLINE void load(const hknpSolverInfo& solverInfo, const hknpSolverStep& step,
								const hkQTransform& firstFrame, const hknpSolverSumVelocity* solverSumVel, const hknpSolverVelocity* solverVel)
		{
			solverVel->getInvInertias(m_invInteriasLocal);
			solverVel->getLinearVelocity(m_solverNLinVel);
			solverVel->getAngularVelocity(m_solverNAngVelLocal);

			hkVector4 solverSumLinVel, solverSumAngVelLocal;
			solverSumVel->getLinearVelocity(solverSumLinVel);
			solverSumVel->getAngularVelocity(solverSumAngVelLocal);

			hkVector4 angVelLocal;

			// Calculate real current velocities.
			m_linVel.setSub(m_solverNLinVel, solverSumLinVel);
			angVelLocal.setSub(m_solverNAngVelLocal, solverSumAngVelLocal);

			// Calculate current frame.
			{
				hkSimdReal dt = solverInfo.m_subStepDeltaTime;

				hkVector4 slv; slv.setMul(solverSumLinVel, solverInfo.m_invIntegrateVelocityFactor);
				// Translation.
				{
					m_currFrame.m_translation.setAddMul(firstFrame.getTranslation(), slv, dt);
				}

				hkVector4 savl; savl.setMul(solverSumAngVelLocal, solverInfo.m_invIntegrateVelocityFactor);
				// Orientation.
				{
					hkVector4 vAngVelLocal = savl;
					hkQuaternion dq;

					dq.m_vec.setMul(dt * hkSimdReal_Half, vAngVelLocal);
					dq.m_vec.setComponent<3>(hkSimdReal_1);

					hkQuaternion currOrient; currOrient.setMul(firstFrame.getRotation(), dq);
					currOrient.normalize<HK_ACC_23_BIT, HK_SQRT_IGNORE>();

					m_currFrame.setRotation(currOrient);
				}
			}

			// Calculate real current world angular velocity.
			toWorldAngular(m_currFrame, angVelLocal, m_angVel);
		}

		HK_FORCE_INLINE void store(const hknpSolverInfo& solverInfo, const hknpSolverSumVelocity* HK_RESTRICT solverSumVel, hknpSolverVelocity* HK_RESTRICT solverVel)
		{
			solverVel->setVelocity(m_solverNLinVel, m_solverNAngVelLocal);
		}

		HK_FORCE_INLINE hkVector4Parameter getLinVel() const { return m_linVel; }
		HK_FORCE_INLINE hkVector4Parameter getAngVel() const { return m_angVel; }
		HK_FORCE_INLINE const hkQTransform& getCurrFrame() const { return m_currFrame; }

		HK_FORCE_INLINE hkVector4 getVelocityAtPoint(const hknpSolverInfo& solverInfo, const hkQTransform& frame, hkVector4Parameter arm)
		{
			hkVector4 vel;

			vel.setCross(m_angVel, arm);
			vel.add(m_linVel);

			return vel;
		}

		HK_FORCE_INLINE hkVector4 getVelocityAtPoint(const hknpSolverInfo& solverInfo, hkVector4Parameter arm)
		{
			return getVelocityAtPoint(solverInfo, m_currFrame, arm);
		}

		HK_FORCE_INLINE hkSimdReal getInvMass() const { return m_invInteriasLocal.getComponent<3>(); }
		HK_FORCE_INLINE hkVector4 getLocalInvInertia() const { return m_invInteriasLocal; }
		HK_FORCE_INLINE hkSimdReal getMass() const { return m_invInteriasLocal.getComponent<3>().reciprocal(); }
		HK_FORCE_INLINE hkVector4 getLocalInertia() const { hkVector4 res; res.setReciprocal(m_invInteriasLocal); return res; }

		static HK_FORCE_INLINE void computeInvInertiaTensor(const hkQTransform& frame, hkVector4Parameter invInteriaLocal, hkMatrix3& invI)
		{
			hkRotation rotation; rotation.set(frame.getRotation());

			hkMatrix3 in;

			hkVector4 a; a.setBroadcast<0>( invInteriaLocal );
			hkVector4 b; b.setBroadcast<1>( invInteriaLocal );
			hkVector4 c; c.setBroadcast<2>( invInteriaLocal );
			in.getColumn(0).setMul( a, rotation.getColumn(0) );
			in.getColumn(1).setMul( b, rotation.getColumn(1) );
			in.getColumn(2).setMul( c, rotation.getColumn(2) );

			invI.setMulInverse( in, rotation );
		}

		HK_FORCE_INLINE void computeInvInertiaTensor(const hkQTransform& frame, hkMatrix3& invI) const
		{
			computeInvInertiaTensor(frame, m_invInteriasLocal, invI);
		}

		HK_FORCE_INLINE void computeInvInertiaTensor(hkMatrix3& invI) const
		{
			computeInvInertiaTensor(m_currFrame, m_invInteriasLocal, invI);
		}

		HK_FORCE_INLINE void computeInvI(hkMatrix3& invI) const
		{
			return computeInvInertiaTensor(invI);
		}

		HK_FORCE_INLINE hkMatrix3 computeInvInertiaTensor(const hkQTransform& frame) const
		{
			hkMatrix3 invI; computeInvInertiaTensor(frame, m_invInteriasLocal, invI); return invI;
		}

		HK_FORCE_INLINE hkMatrix3 computeInvInertiaTensor() const
		{
			hkMatrix3 invI; computeInvInertiaTensor(m_currFrame, m_invInteriasLocal, invI); return invI;
		}

		HK_FORCE_INLINE hkMatrix3 computeInvI() const
		{
			return computeInvInertiaTensor();
		}

		HK_FORCE_INLINE void computeInvInertiaTensor(const hkQTransform& frame, hkSimdRealParameter scale, hkMatrix3& invI) const
		{
			hkVector4 ils; ils.setMul(m_invInteriasLocal, scale);
			computeInvInertiaTensor(frame, ils, invI);
		}

		static HK_FORCE_INLINE void toWorldAngular(const hkQTransform& frame, hkVector4Parameter angVelLocal, hkVector4& angVelWorld)
		{
			angVelWorld._setRotatedDir(frame.getRotation(), angVelLocal);
		}

		static HK_FORCE_INLINE hkVector4 toWorldAngular(const hkQTransform& frame, hkVector4Parameter angVelLocal)
		{
			hkVector4 angVelWorld; toWorldAngular(frame, angVelLocal, angVelWorld); return angVelWorld;
		}

		static HK_FORCE_INLINE void toLocalAngular(const hkQTransform& frame, hkVector4Parameter angVel, hkVector4& angVelLocal)
		{
			angVelLocal._setRotatedInverseDir(frame.getRotation(), angVel);
		}

		HK_FORCE_INLINE void toWorldAngular(hkVector4Parameter angVelLocal, hkVector4& angVelWorld) const
		{
			toWorldAngular(m_currFrame, angVelLocal, angVelWorld);
		}

		HK_FORCE_INLINE hkVector4 toWorldAngular(hkVector4Parameter angVelLocal) const
		{
			hkVector4 angVelWorld; toWorldAngular(angVelLocal, angVelWorld); return angVelWorld;
		}

		HK_FORCE_INLINE void toLocalAngular(hkVector4Parameter angVel, hkVector4& angVelLocal) const
		{
			angVelLocal._setRotatedInverseDir(m_currFrame.getRotation(), angVel);
		}

		static HK_FORCE_INLINE void computeFutureFrame(const hkQTransform& frame, hkVector4Parameter linDisp, hkQuaternionParameter angDisp, hkQTransform& futureFrame)
		{
			futureFrame.m_translation.setAdd(frame.getTranslation(), linDisp);

			hkQuaternion futureOrientation; futureOrientation.setMul(angDisp, frame.getRotation());
			futureOrientation.normalize<HK_ACC_23_BIT, HK_SQRT_IGNORE>();

			futureFrame.setRotation(futureOrientation);
		}

		static HK_FORCE_INLINE void computeFutureFrameL(const hkQTransform& frame, hkVector4Parameter linDisp, hkQuaternionParameter angDispLocal, hkQTransform& futureFrame)
		{
			futureFrame.m_translation.setAdd(frame.getTranslation(), linDisp);

			hkQuaternion futureOrientation; futureOrientation.setMul(frame.getRotation(), angDispLocal);
			futureOrientation.normalize<HK_ACC_23_BIT, HK_SQRT_IGNORE>();

			futureFrame.setRotation(futureOrientation);
		}

		HK_FORCE_INLINE void computeNextFrame(const hknpSolverInfo& solverInfo, hkQTransform& futureFrame) const
		{
			hkSimdReal dt = solverInfo.m_subStepDeltaTime;

			hkVector4 dl; dl.setMul(m_linVel, dt);
			hkQuaternion dq;
			dq.m_vec.setMul(dt * hkSimdReal_Half, m_angVel);
			dq.m_vec.setComponent<3>(hkSimdReal_1);

			computeFutureFrame(m_currFrame, dl, dq, futureFrame);
		}

		HK_FORCE_INLINE void calculateAngularImpulse(const hkQTransform& frame, hkVector4Parameter angVelDiff, hkVector4& impulse) const
		{
			hkVector4 dVelLocal; toLocalAngular(frame, angVelDiff, dVelLocal);

			hkVector4Parameter invInertiaLocal = m_invInteriasLocal;
			hkVector4 impLocal; impLocal.setDiv<HK_ACC_23_BIT, HK_DIV_SET_ZERO>(dVelLocal, invInertiaLocal);

			toWorldAngular(frame, impLocal, impulse);
		}

		HK_FORCE_INLINE void calculateAngularImpulse(hkVector4Parameter angVelDiff, hkVector4& impulse) const
		{
			return calculateAngularImpulse(m_currFrame, angVelDiff, impulse);
		}

		HK_FORCE_INLINE void applyLinearImpulse(hkVector4Parameter impulse)
		{
			const hkSimdReal invMass = m_invInteriasLocal.getComponent<3>();
			hkVector4 diff; diff.setMul(invMass, impulse);
			m_solverNLinVel.add(diff);
			m_linVel.add(diff);
		}

		HK_FORCE_INLINE void applyAngularImpulse(const hkQTransform& frame, hkVector4Parameter impulse)
		{
			hkVector4 angImpLocal; toLocalAngular(frame, impulse, angImpLocal);

			hkVector4 diffLocal; diffLocal.setMul(m_invInteriasLocal, angImpLocal);
			m_solverNAngVelLocal.add(diffLocal);

			hkVector4 diff;
			toWorldAngular(frame, diffLocal, diff);
			m_angVel.add(diff);
		}

		HK_FORCE_INLINE void applyAngularImpulse(hkVector4Parameter impulse)
		{
			return applyAngularImpulse(m_currFrame, impulse);
		}

		HK_FORCE_INLINE void applyImpulseUsingArm(const hkQTransform& frame, hkVector4Parameter arm, hkVector4Parameter impulse)
		{
			applyLinearImpulse(impulse);

			hkVector4 angImp; angImp.setCross(arm, impulse);
			applyAngularImpulse(frame, angImp);
		}

		HK_FORCE_INLINE void applyImpulseUsingArm(hkVector4Parameter arm, hkVector4Parameter impulse)
		{
			applyImpulseUsingArm(m_currFrame, arm, impulse);
		}

		static HK_FORCE_INLINE void computeArmAndPivot(const hkQTransform& frame, hkVector4Parameter localArm, hkVector4& arm, hkVector4& pivot)
		{
			arm._setRotatedDir(frame.getRotation(), localArm);
			pivot.setAdd(arm, frame.getTranslation());
		}

		HK_FORCE_INLINE void computeArmAndPivot(hkVector4Parameter localArm, hkVector4& arm, hkVector4& pivot)
		{
			computeArmAndPivot(m_currFrame, localArm, arm, pivot);
		}

		HK_FORCE_INLINE static void computeMassMatrixForArm(const hkQTransform& frame, hkVector4Parameter arm, const hkMatrix3& inverse_inertia_world, hkSimdRealParameter inverse_mass, hkMatrix3& effMassMatrixOut)
		{
			hkMatrix3 rhat;		rhat.setCrossSkewSymmetric(arm);

			hkMatrix3Util::_setDiagonal( inverse_mass, inverse_mass, inverse_mass, effMassMatrixOut );

			// calculate: effMassMatrixOut -= (rhat * inertialInvWorld * rhat)
			hkMatrix3 temp;		temp._setMul(rhat, inverse_inertia_world);
			hkMatrix3 temp2;	temp2._setMul(temp, rhat);
			effMassMatrixOut.sub(temp2);
			effMassMatrixOut._invertSymmetric();
		}

		HK_FORCE_INLINE void computeMassMatrixForArm(const hkQTransform& frame, hkVector4Parameter arm, hkMatrix3& effMassMatrixOut) const
		{
			hkMatrix3 inverse_inertia_world;
			computeInvInertiaTensor(frame,inverse_inertia_world);
			computeMassMatrixForArm(frame,arm,inverse_inertia_world,getInvMass(),effMassMatrixOut);
		}

		HK_FORCE_INLINE void computeMassMatrixForArm(hkVector4Parameter arm, hkMatrix3& effMassMatrixOut) const
		{
			computeMassMatrixForArm(m_currFrame,arm,effMassMatrixOut);
		}

		hkQTransform m_currFrame;

		hkVector4 m_invInteriasLocal;

		hkVector4 m_linVel;
		hkVector4 m_angVel;

		// Warning: these are not the real velocities.
		hkVector4 m_solverNLinVel;
		hkVector4 m_solverNAngVelLocal;
	};
}

#endif // HKNP_USER_SOLVER_UTILITY_H

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
