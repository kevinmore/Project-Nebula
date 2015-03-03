/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Chain/Powered/hkpPoweredChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstanceAction.h>

#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>

#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics/ConstraintSolver/Constraint/Motor/hkpMotorConstraintInfo.h>
#include <Physics/ConstraintSolver/Constraint/Chain/hkpPoweredChainSolverUtil.h>

#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics2012/Dynamics/Motion/hkpMotion.h>


#include <Physics/Constraint/Motor/hkpConstraintMotor.h>
#include <Physics/Constraint/Motor/Position/hkpPositionConstraintMotor.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>


hkpPoweredChainData::hkpPoweredChainData() : m_tau(hkReal(0.6f)), m_damping(hkReal(1))
{
	m_cfmLinAdd = hkReal(0.1f) * HK_REAL_EPSILON;
	m_cfmLinMul = hkReal(1);
	m_cfmAngAdd = hkReal(0.1f) * HK_REAL_EPSILON;
	m_cfmAngMul = hkReal(1);

	m_maxErrorDistance = hkReal(0.1f);

	m_atoms.m_bridgeAtom.init( this );
}

hkpPoweredChainData::hkpPoweredChainData(hkFinishLoadedObjectFlag f) : hkpConstraintChainData(f), m_atoms(f), m_infos(f)
{
	if( f.m_finishing )
	{
		m_atoms.m_bridgeAtom.init( this );
	}
}


hkpPoweredChainData::~hkpPoweredChainData()
{
	for (int i = 0; i < m_infos.getSize(); i++)
	{
		for (int m = 0; m < 3; m++)
		{
			if (m_infos[i].m_motors[m])
			{
				m_infos[i].m_motors[m]->removeReference();
			}
		}
	}
}


int hkpPoweredChainData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN;
}

void hkpPoweredChainData::addConstraintInfoInBodySpace(const hkVector4& pivotInA, const hkVector4& pivotInB, const hkQuaternion& aTc, 
														hkpConstraintMotor* xMotor, hkpConstraintMotor* yMotor, hkpConstraintMotor* zMotor)
{
	ConstraintInfo& info = m_infos.expandOne();
	info.m_pivotInA = pivotInA;
	info.m_pivotInB = pivotInB;
	info.m_aTc = aTc;

	info.m_motors[0] = xMotor; xMotor->addReference();
	info.m_motors[1] = yMotor; yMotor->addReference();
	info.m_motors[2] = zMotor; zMotor->addReference();

	info.m_switchBodies = false;
	info.m_bTc.setIdentity();

	//info.m_cfmLinear = m_cfm;
	//info.m_cfmAngular = m_cfm;
}





void hkpPoweredChainData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
	info.clear();
	info.addHeader();

	const int numConstraints = this->m_infos.getSize();

	int schemaSize  = hkpJacobianSchemaInfo::PoweredChain::Sizeof
	                + 3 * numConstraints    * sizeof(hkp1Lin2AngJacobian)
					+ 3 * numConstraints    * sizeof(hkp2AngJacobian)
					+ numConstraints        * sizeof(hkpConstraintChainMatrix6Triple)
					+ (1 + numConstraints ) * sizeof(hkVector8) // temp buff
					+ numConstraints        * sizeof(hkVector8) // velocity buffer
					+ (1 + numConstraints)  * sizeof(hkpVelocityAccumulatorOffset) // accumulators
					+ numConstraints        * sizeof(hkp3dAngularMotorSolverInfo); // child constraint status
	    schemaSize  = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, schemaSize);

	int resultsSize = 6 * numConstraints;

	info.add(schemaSize, resultsSize, resultsSize);
}


void hkpPoweredChainData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const 
{
	const int numConstraints = m_infos.getSize();
	const int childSolverResultMax = 6;

	{
		infoOut.m_numSolverResults = numConstraints * childSolverResultMax;
		infoOut.m_sizeOfExternalRuntime = infoOut.m_numSolverResults * sizeof(hkpSolverResults)
										+ HK_NEXT_MULTIPLE_OF(4, numConstraints * sizeof(hkp3dAngularMotorSolverInfo::Status)) 
										+ numConstraints * sizeof(hkQuaternion); 
	}
}

void HK_CALL hkpPoweredChainData::enableMotor(hkpConstraintChainInstance* instance, int constraintIndex, int motorIndex)
{
	HK_ASSERT2(0xad7899dd, motorIndex >=0 && motorIndex < 3, "motorIndex must be in [1,3] range");
	// xxx remove the data dependency
	hkpPoweredChainData* data = static_cast<hkpPoweredChainData*>(instance->getData());
	hkp3dAngularMotorSolverInfo::Status* statuses = data->getConstraintFlags( instance->getRuntime() );
	HK_ASSERT2(0xad8bdd9d, instance->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN && statuses && constraintIndex < instance->getNumConstraints(), "Wrong constraint type or no runtime or constraintIndex out of range.");

	hkp3dAngularMotorSolverInfo::Status zeroMask = static_cast<hkp3dAngularMotorSolverInfo::Status>(~(hkp3dAngularMotorSolverInfo::ANGULAR_1ST << (2 * motorIndex)));
	hkp3dAngularMotorSolverInfo::Status onStatus = static_cast<hkp3dAngularMotorSolverInfo::Status>(hkp3dAngularMotorSolverInfo::MOTOR_NOT_BROKEN << (2 * motorIndex));

	statuses[constraintIndex] = statuses[constraintIndex] & zeroMask;
	statuses[constraintIndex] = statuses[constraintIndex] | onStatus;
}

void HK_CALL hkpPoweredChainData::disableMotor(hkpConstraintChainInstance* instance, int constraintIndex, int motorIndex)
{
	HK_ASSERT2(0xad7899dd, motorIndex >=0 && motorIndex < 3, "motorIndex must be in [1,3] range");

	// xxx remove the data dependency
	hkpPoweredChainData* data = static_cast<hkpPoweredChainData*>(instance->getData());
	hkp3dAngularMotorSolverInfo::Status* statuses = data->getConstraintFlags( instance->getRuntime() );
	HK_ASSERT2(0xad8bdd9d, instance->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN && statuses && constraintIndex < instance->getNumConstraints(), "Wrong constraint type or no runtime or constraintIndex out of range.");

	hkp3dAngularMotorSolverInfo::Status zeroMask = static_cast<hkp3dAngularMotorSolverInfo::Status>(~(hkp3dAngularMotorSolverInfo::ANGULAR_1ST << (2 * motorIndex)));
	hkp3dAngularMotorSolverInfo::Status offStatus = static_cast<hkp3dAngularMotorSolverInfo::Status>(hkp3dAngularMotorSolverInfo::MOTOR_DISABLED << (2 * motorIndex));

	statuses[constraintIndex] = statuses[constraintIndex] & zeroMask;
	statuses[constraintIndex] = statuses[constraintIndex] | offStatus;
}


	// returns ( from-1 * to ) * 2.0f
static inline void HK_CALL estimateAngleToLs(const hkQuaternion& from, const hkQuaternion& to, hkVector4& angleOut)
{
	angleOut.setCross(from.getImag(),   to.getImag());
	angleOut.addMul(to.getRealPart(),   from.getImag());
	angleOut.subMul(from.getRealPart(), to.getImag());
	angleOut.add(angleOut);
	angleOut.setFlipSign(angleOut, to.getImag().dot<4>( from.getImag() ));
}

static HK_FORCE_INLINE void HK_CALL hk1dAngularVelocityMotorCommitJacobianInMotorInfo( hkp1dConstraintMotorInfo& info, const hkpConstraintQueryIn &in, hkp2AngJacobian& jac, hkp1dMotorSolverInfo* motorInfoOut )
{
	hkp1dMotorSolverInfo* si = motorInfoOut;

	si->m_maxImpulsePerSubstep = info.m_maxForce * in.m_microStepDeltaTime;
	si->m_minImpulsePerSubstep = info.m_minForce * in.m_microStepDeltaTime;
	si->m_velocity = info.m_targetVelocity;
	si->m_tau = info.m_tau;
	si->m_damping = info.m_damping;

	const hkReal rhs = info.m_targetPosition * in.m_subStepInvDeltaTime;
	jac.setAngularRhs( rhs );
}

extern "C"
{
	hkp1Lin2AngJacobian* HK_CALL hkJacobianPoweredChainSchema_getLinearJacobians ( hkpJacobianSchema* schema );
	hkp2AngJacobian*	 HK_CALL hkJacobianPoweredChainSchema_getAngularJacobians( hkpJacobianSchema* schema, int numConstraints );
}

void hkpPoweredChainData::buildJacobian( const hkpConstraintQueryIn &inNotValid, hkpConstraintQueryOut &out )
{
	{
		hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() ) ;
		inNotValid.beginConstraints( out, solverResults, sizeof(hkpSolverResults) );
	}

	hkpConstraintRuntime* origConstraintRuntime = out.m_constraintRuntime;
	hkpConstraintQueryIn newIn = inNotValid;
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
			HK_ASSERT2(0xad677d6d, inNotValid.m_constraintInstance, "internal error");
			chainInstance = reinterpret_cast<hkpConstraintChainInstance*>(inNotValid.m_constraintInstance.val());
			const hkpEntity* chainCA = chainInstance->getEntityA();
			baseAccum = hkAddByteOffsetConst(inNotValid.m_bodyA.val(), - int(chainCA->m_solverData) );
		}

		{
			// Check whether the accompanying action is properly added to the world
			HK_ASSERT2(0xad5677dd, chainInstance->m_action->getWorld() == static_cast<hkpSimulationIsland*>(reinterpret_cast<hkpConstraintChainInstance*>(inNotValid.m_constraintInstance.val())->getOwner())->getWorld(), "The action and the chain instance must be both added to the world before running simulation.");
		}

		const hkArray<hkpEntity*>& entities = chainInstance->m_chainedEntities;
		int numConstraints = entities.getSize() - 1;
		HK_ASSERT2(0xad567755, numConstraints <= m_infos.getSize(), "Not enough pivot sets are specified in the hkChainConstraintData to handle all entities referenced by the hkpConstraintChainInstance.");

		hkInplaceArray<hkp3dAngularMotorSolverInfo, 32> motorsState; motorsState.setSize(numConstraints);


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

		hkp1Lin2AngJacobian* const  linearJacobiansBase = hkJacobianPoweredChainSchema_getLinearJacobians(out.m_jacobianSchemas.val());
		hkp2AngJacobian*	const      angularJacobiansBase = hkJacobianPoweredChainSchema_getAngularJacobians(out.m_jacobianSchemas.val(), numConstraints);
		hkp1Lin2AngJacobian* linearJacobians = linearJacobiansBase;
		hkp2AngJacobian*     angularJacobians = angularJacobiansBase;


		for (int i = 0; i < numConstraints; i++)
		{
			newIn.m_bodyA = newIn.m_bodyB;
			newIn.m_transformA = newIn.m_transformB;

			rB = entities[i+1];//  m_constraintInstances[i]->getEntityB();
			cB = rB->getMotion();
			newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );

			accumulators.pushBack( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );


			newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
			HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

			hkVector4 posA; posA._setTransformedPos( *newIn.m_transformA, m_infos[i].m_pivotInA );
			hkVector4 posB; posB._setTransformedPos( *newIn.m_transformB, m_infos[i].m_pivotInB );

			// we're ignoring the shema generated by the pulley.
			hkStabilizedBallSocketConstraintBuildJacobian_noSchema( posA, posB, m_maxErrorDistance, newIn, linearJacobians );
			linearJacobians += 3;

			//
			// And here angular parts
			//
			{
				hkQuaternion rotA; rotA.set( newIn.m_transformA->getRotation() );
				hkQuaternion rotB; rotB.set( newIn.m_transformB->getRotation() );

				if (m_infos[i].m_switchBodies)
				{
					hkAlgorithm::swap(rotA, rotB);
				}

				// Apply constraint space offset here
				{
					hkQuaternion wTc; wTc.setMul(rotB, m_infos[i].m_bTc);
					rotB = wTc;
				}

				const hkQuaternion& aTc = m_infos[i].m_aTc;

				// copy values, due to SIMD alignment
				hkReal* oldTargetFrameF = &getMotorRuntimeQuaternions(origConstraintRuntime)[i*4];
				hkQuaternion oldTargetFrame;
				oldTargetFrame.m_vec.load<4,HK_IO_NATIVE_ALIGNED>(oldTargetFrameF);

				// <todo: have a boolean flag in runtime marking the runtime as [un]initialized
				if (oldTargetFrame.m_vec.lengthSquared<4>().isEqualZero())
				{
					oldTargetFrame = aTc;
				}

				// Convention:
				//    a a space
				//    ob current b-space
				//	  nb new/target b-space

				hkQuaternion target_wTnb; target_wTnb.setMul(rotA, aTc);
				hkQuaternion target_wTob; target_wTob.setMul(rotA, oldTargetFrame);
				aTc.m_vec.store<4,HK_IO_NATIVE_ALIGNED>(oldTargetFrameF);

				hkVector4 deltaTarget;    estimateAngleToLs( rotB, target_wTnb, deltaTarget); // == target_wTob^1 * target_wTnb 
				hkVector4 positionError;  estimateAngleToLs( rotB, target_wTob, positionError);
				deltaTarget.sub( positionError );

				if (m_infos[i].m_switchBodies)
				{
					deltaTarget.setNeg<3>(deltaTarget);
					positionError.setNeg<3>(positionError);
				}

				hkRotation constraintSpace; constraintSpace.set( rotB );

				for (int j = 0; j < 3; j++)
				{
					const hkVector4 constrainedDofW = constraintSpace.getColumn(j);

					//////////////////////////////////////////////////////////////////////////
					hkpSolverResults* solverResults = getSolverResults( chainInstance->getRuntime() );
					
					{
						HK_ASSERT2( 0xf032ef45, m_infos[i].m_motors[j] != HK_NULL, "You must supply motors for this constraint to work" );

						hkpConstraintMotorInput motorIn;

						hk1dAngularVelocityMotorBeginJacobian( constrainedDofW, newIn, angularJacobians, motorIn ); 

						motorIn.m_stepInfo    = &newIn;
						motorIn.m_lastResults = solverResults[i*6/*6solverresults per constraint*/ + 3 + j];

						motorIn.m_deltaTarget   = deltaTarget(j);
						motorIn.m_positionError = positionError(j);

						hkpConstraintMotorOutput motorOut;
						hkCalcMotorData(m_infos[i].m_motors[j], &motorIn, &motorOut );

						//make it inline
						hk1dAngularVelocityMotorCommitJacobianInMotorInfo( motorOut, newIn, *angularJacobians, &motorsState[i].m_motorInfos[j] );
						angularJacobians++;
					}
				}
			}
		}

		HK_ASSERT2(0xad6777dd, hkUlong(angularJacobiansBase) == hkUlong(linearJacobians), "Internal error: angular vs linear jacobians not properly placed.");
		//HK_ASSERT2(0xad6777dd, start of matrix  == angularJacobians, "Internal error: angular vs linear jacobians not properly placed.");

		//
		// Initialize the schema and build the constraint matrix
		//
		{
			HK_ASSERT2(0xad674d4d, numConstraints == accumulators.getSize() - 1, "Number of chained constraints and number of velocity accumulators don't match.");
			hkp3dAngularMotorSolverInfo::Status *const  childConstraintStatusFlags = getConstraintFlags( chainInstance->getRuntime() );

			for (int c = 0; c < numConstraints; c++)
			{
				motorsState[c].m_broken = childConstraintStatusFlags[c];
			}

			hkpPoweredChainBuildJacobianParams params;
			params.m_numConstraints = numConstraints;
			params.m_chainTau = m_tau;
			params.m_chainDamping = m_damping;
			params.m_cfm.m_linAdd = m_cfmLinAdd;
			params.m_cfm.m_linMul = m_cfmLinMul;
			params.m_cfm.m_angAdd = m_cfmAngAdd;
			params.m_cfm.m_angMul = m_cfmAngMul;
			params.m_accumulators = accumulators.begin();
			params.m_accumsBase = baseAccum;
			params.m_motorsState = motorsState.begin();
			params.m_maxTorqueHysterisys = hkReal(0);
			params.m_childConstraintStatusWriteBackBuffer = childConstraintStatusFlags;
			params.m_jacobiansEnd   = reinterpret_cast<hkp2AngJacobian*>(angularJacobians);

			hkPoweredChainBuildJacobian( params, inNotValid, out );
		}
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
