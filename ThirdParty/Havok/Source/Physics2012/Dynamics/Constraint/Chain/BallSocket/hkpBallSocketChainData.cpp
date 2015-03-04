/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Chain/BallSocket/hkpBallSocketChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstanceAction.h>

#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>

#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics/ConstraintSolver/Constraint/Chain/hkpChainConstraintInfo.h>
#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics2012/Dynamics/Motion/hkpMotion.h>
#include <Physics2012/Dynamics/Motion/hkpMotion.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

hkpBallSocketChainData::hkpBallSocketChainData() : m_tau(hkReal(0.6f)), m_damping(hkReal(1)), m_cfm(hkReal(0.1f) * HK_REAL_EPSILON) 
{
	m_maxErrorDistance = 0.1f;
	m_useStabilizedCode = false;
	m_atoms.m_bridgeAtom.init( this );
}

hkpBallSocketChainData::hkpBallSocketChainData(hkFinishLoadedObjectFlag f) : hkpConstraintChainData(f), m_atoms(f), m_infos(f)
{
	if( f.m_finishing )
	{
		m_atoms.m_bridgeAtom.init( this );
	}
}


hkpBallSocketChainData::~hkpBallSocketChainData()
{
}


int hkpBallSocketChainData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN;
}

void hkpBallSocketChainData::addConstraintInfoInBodySpace(const hkVector4& pivotInA, const hkVector4& pivotInB)
{
	ConstraintInfo& info = m_infos.expandOne();
	info.m_pivotInA = pivotInA;
	info.m_pivotInB = pivotInB;
}

void hkpBallSocketChainData::useStabilizedCode(bool useIt)
{
	// todo: assert not in world
	m_useStabilizedCode = useIt;
}



void hkpBallSocketChainData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
	info.clear();
	info.addHeader();

	const int numConstraints = this->m_infos.getSize();


	int schemaSize  = hkpJacobianSchemaInfo::BallSocketChain::Sizeof
	                + 3 * numConstraints    * sizeof(hkp1Lin2AngJacobian)
					+ numConstraints        * sizeof(hkpConstraintChainMatrixTriple)
					+ (1 + numConstraints ) * sizeof(hkVector4) // temp buff
					+ (1 + numConstraints)  * sizeof(hkpVelocityAccumulatorOffset); // accumulators
	    schemaSize  = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, schemaSize);
	int resultsSize = hkpJacobianSchemaInfo::StableBallSocket::Results * numConstraints;

	info.add(schemaSize, resultsSize, resultsSize );
}


void hkpBallSocketChainData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const 
{
	if ( wantRuntime )
	{
		int childSolverResultMax = hkpBallAndSocketConstraintData::SOLVER_RESULT_MAX;
		infoOut.m_numSolverResults = m_infos.getSize() * childSolverResultMax;
		infoOut.m_sizeOfExternalRuntime = infoOut.m_numSolverResults * sizeof(hkpSolverResults);
	}
	else
	{
		infoOut.m_numSolverResults = 0;
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}



extern "C"
{
	hkp1Lin2AngJacobian* HK_CALL hkJacobianBallSocketChainSchema_getJacobians(hkpJacobianSchema* schema);
	hkpJacobianSchema*	 HK_CALL hkJacobianStabilizedBallSocketChainSchema_init(hkpJacobianSchema* sIn, int numConstraints, hkpVelocityAccumulatorOffset* accumulatorsIn, hkReal tau, hkReal damping, hkReal cfm, hkReal virtualMassFactor);
}

void hkpBallSocketChainData::buildJacobian( const hkpConstraintQueryIn &inChain, hkpConstraintQueryOut &out )
{
	if (!m_useStabilizedCode)
	{
		buildJacobian_Unstabilized(inChain, out);
	}
	else
	{
		buildJacobian_Stabilized(inChain, out);
	}
}

void hkpBallSocketChainData::buildJacobian_Unstabilized( const hkpConstraintQueryIn &inChain, hkpConstraintQueryOut &out )
{
	hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() ) ;

	inChain.beginConstraints( out, solverResults, sizeof(hkpSolverResults) );

	hkpConstraintQueryIn newIn = inChain;
	{
		newIn.m_constraintInstance = HK_NULL;
		out.m_constraintRuntime  = HK_NULL;
	}

	hkInplaceArray<hkpVelocityAccumulatorOffset,32> accumulators;

	{
		const hkpVelocityAccumulator * baseAccum;
		hkpConstraintChainInstance* chainInstance;
		{
			// Calculate base address of accumulators
			HK_ASSERT2(0xad677d6d, inChain.m_constraintInstance, "internal error");
			chainInstance = reinterpret_cast<hkpConstraintChainInstance*>(inChain.m_constraintInstance.val());
			const hkpEntity* chainCA = chainInstance->getEntityA();
			baseAccum = hkAddByteOffsetConst(inChain.m_bodyA.val(), -int(chainCA->m_solverData) );
		}

		{
			// Check whether the accompanying action is properly added to the world
			HK_ASSERT2(0xad5677dd, chainInstance->m_action->getWorld() == static_cast<hkpSimulationIsland*>(reinterpret_cast<hkpConstraintChainInstance*>(inChain.m_constraintInstance.val())->getOwner())->getWorld(), "The action and the chain instance must be both added to the world before running simulation.");
		}

		const hkArray<hkpEntity*>& entities = chainInstance->m_chainedEntities;
		int numConstraints = entities.getSize() - 1;

		HK_ASSERT2(0xad567755, numConstraints <= m_infos.getSize(), "Not enough pivot sets are specified in the hkChainConstraintData to handle all entities referenced by the hkpConstraintChainInstance.");


		// Initialize first body info
		hkpEntity* rB = entities[0]; // yes, body A
		hkpMotion* cB = rB->getMotion();
		newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );
		accumulators.pushBackUnchecked( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );

		newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
		HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

			// set the tau to use the user tau functions
		newIn.m_rhsFactor = newIn.m_subStepInvDeltaTime;
		newIn.m_virtMassFactor = hkReal(1);

		hkp1Lin2AngJacobian* jacobiansBase = hkJacobianBallSocketChainSchema_getJacobians(out.m_jacobianSchemas.val());
		hkp1Lin2AngJacobian* jacobians = jacobiansBase;
		//HK_TIMER_BEGIN_LIST( "hkBallSocketChainBuildJacobian", "jac");
		for (int i = 0; i < numConstraints; i++)
		{
			newIn.m_bodyA = newIn.m_bodyB;
			newIn.m_transformA = newIn.m_transformB;

			rB = entities[i+1];// i.e. m_constraintInstances[i]->getEntityB();
			cB = rB->getMotion();
			newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );

			accumulators.pushBack( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );


			newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
			HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

			hkVector4 posA; posA._setTransformedPos( *newIn.m_transformA, m_infos[i].m_pivotInA );
			hkVector4 posB; posB._setTransformedPos( *newIn.m_transformB, m_infos[i].m_pivotInB );

			// we're ignoring the shema generated by the ball-and-socket.
			// xxx use lower level function and remove newIn
			if(0)
			{
				//?? we shoudl build those stabilized jacobians after the constraint matrix is build on the base of normal ones..
				hkStabilizedBallSocketConstraintBuildJacobian_noSchema( posA, posB, m_maxErrorDistance, newIn, jacobians );
			}
			else
			{
				hkBallSocketConstraintBuildJacobian_noSchema_Proj( posA, posB, newIn, jacobians);
			}
			jacobians += 3;
		}
	//HK_TIMER_SPLIT_LIST("MassMatrix and Lu-demp");
		//
		// Initialize the schema and build the constraint matrix
		//
		HK_ASSERT2(0xad674d4d, numConstraints == accumulators.getSize() - 1, "Number of chained constraints and number of velocity accumulators don't match.");
		hkBallSocketChainBuildJacobian( numConstraints, m_tau, m_damping, m_cfm, accumulators.begin(), baseAccum, jacobians, inChain, out );
	}

	//HK_TIMER_END_LIST();
	hkEndConstraints();

}


extern void HK_CALL setupStabilizationFromAtom_outOfLine(	const struct hkpSetupStabilizationAtom& atom,
															   const hkpConstraintQueryIn& in,
															   const hkTransform& baseA, const hkTransform& baseB,
															   hkVector4& vLocalArmA, hkVector4& vLocalArmB,
															   hkSimdReal& maxAngularImpulse, hkSimdReal& maxLinearImpulse,
															   const hkpConstraintQueryOut& noOut);

extern void HK_CALL buildSchemaFromBallSocketAtom(	const struct hkpBallSocketConstraintAtom& atom,
														 const hkpConstraintQueryIn &in, 
														 const hkTransform& baseA, const hkTransform& baseB,
														 hkVector4Parameter vLocalArmA, hkVector4Parameter vLocalArmB,
														 hkSimdRealParameter maxAngularImpulse, hkSimdRealParameter maxLinearImpulse,
														 hkpConstraintQueryOut &out);


void hkpBallSocketChainData::buildJacobian_Stabilized( const hkpConstraintQueryIn &inChain, hkpConstraintQueryOut &out )
{
	hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() ) ;

	inChain.beginConstraints( out, solverResults, sizeof(hkpSolverResults) );

	hkpConstraintQueryIn newIn = inChain;
	{
		newIn.m_constraintInstance = HK_NULL;
		out.m_constraintRuntime  = HK_NULL;
	}

	hkInplaceArray<hkpVelocityAccumulatorOffset,32> accumulators;

	{
		const hkpVelocityAccumulator * baseAccum;
		hkpConstraintChainInstance* chainInstance;
		{
			// Calculate base address of accumulators
			HK_ASSERT2(0xad677d6d, inChain.m_constraintInstance, "internal error");
			chainInstance = reinterpret_cast<hkpConstraintChainInstance*>(inChain.m_constraintInstance.val());
			const hkpEntity* chainCA = chainInstance->getEntityA();
			baseAccum = hkAddByteOffsetConst(inChain.m_bodyA.val(), -int(chainCA->m_solverData) );
		}

		{
			// Check whether the accompanying action is properly added to the world
			HK_ASSERT2(0xad5677dd, chainInstance->m_action->getWorld() == static_cast<hkpSimulationIsland*>(reinterpret_cast<hkpConstraintChainInstance*>(inChain.m_constraintInstance.val())->getOwner())->getWorld(), "The action and the chain instance must be both added to the world before running simulation.");
		}

		const hkArray<hkpEntity*>& entities = chainInstance->m_chainedEntities;
		int numConstraints = entities.getSize() - 1;

		HK_ASSERT2(0xad567755, numConstraints <= m_infos.getSize(), "Not enough pivot sets are specified in the hkChainConstraintData to handle all entities referenced by the hkpConstraintChainInstance.");


		// Initialize first body info
		hkpEntity* rB = entities[0]; // yes, body A
		hkpMotion* cB = rB->getMotion();
		newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );
		accumulators.pushBackUnchecked( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );

		newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
		HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

		// set the tau to use the user tau functions
		newIn.m_rhsFactor = newIn.m_subStepInvDeltaTime;
		newIn.m_virtMassFactor = hkReal(1);

		hkpConstraintQueryOut newOut = out;
		newOut.m_jacobianSchemas = hkJacobianStabilizedBallSocketChainSchema_init(out.m_jacobianSchemas.val(), numConstraints, accumulators.begin(), m_tau, m_damping, m_cfm, inChain.m_virtMassFactor );

		for (int i = 0; i < numConstraints; i++)
		{
			newIn.m_bodyA = newIn.m_bodyB;
			newIn.m_transformA = newIn.m_transformB;

			rB = entities[i+1];// i.e. m_constraintInstances[i]->getEntityB();
			cB = rB->getMotion();
			newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );

			accumulators.pushBack( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );

			newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
			HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

			// Locally cached base vectors
			hkTransform baseA = (*newIn.m_transformA.val());
			hkTransform baseB = (*newIn.m_transformB.val());

			// Locally cached max angular / linear impulse values. Initially set to FLT_MAX
			hkSimdReal maxAngularImpulse = hkSimdReal_Max;
			hkSimdReal maxLinearImpulse = maxAngularImpulse;

			// Locally cached arm values in local space. These are updated by the set transform calls and used inside the stable schemas
			hkVector4 vLocalArmA; vLocalArmA.setZero();
			hkVector4 vLocalArmB; vLocalArmB.setZero();

			// Apply pivots (as in set translations atom)
			baseA.getTranslation()._setTransformedPos(*newIn.m_transformA.val(), m_infos[i].m_pivotInA );
			baseB.getTranslation()._setTransformedPos(*newIn.m_transformB.val(), m_infos[i].m_pivotInB );

			// we're ignoring the shema generated by the ball-and-socket.
			// xxx use lower level function and remove newIn
			//if(0)
			//{
			//	hkLoadVelocityAccumulators( newIn );
			//	//?? we should build those stabilized jacobians after the constraint matrix is build on the base of normal ones..
			//	hkStabilizedBallSocketConstraintBuildJacobian_noSchema( posA, posB, m_maxErrorDistance, newIn, jacobians );
			//}
			//else
			{
				//hkBallSocketConstraintBuildJacobian_noSchema_noProj( posA, posB, newIn, jacobians);

				hkpSetupStabilizationAtom stabilizationAtom;
				stabilizationAtom.m_enabled = true;
				hkpBallSocketConstraintAtom ballSocketAtom;
				ballSocketAtom.m_solvingMethod = hkpConstraintAtom::METHOD_STABILIZED;

				setupStabilizationFromAtom_outOfLine(	stabilizationAtom, newIn, baseA, baseB, vLocalArmA, vLocalArmB, maxAngularImpulse, maxLinearImpulse, newOut);
				buildSchemaFromBallSocketAtom( ballSocketAtom, newIn, baseA, baseB, vLocalArmA, vLocalArmB, maxAngularImpulse, maxLinearImpulse, newOut);
			}
		}

		//
		// Initialize the schema and build the constraint matrix
		//
		HK_ASSERT2(0xad674d4d, numConstraints == accumulators.getSize() - 1, "Number of chained constraints and number of velocity accumulators don't match.");
		hkStabilizedBallSocketChainBuildJacobian( numConstraints, m_tau, m_damping, m_cfm, accumulators.begin(), baseAccum, reinterpret_cast<hkpChainLinkData*>(newOut.m_jacobianSchemas.val()), inChain, out );
	}


	hkEndConstraints();

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
