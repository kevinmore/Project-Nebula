/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/Chain/StiffSpring/hkpStiffSpringChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstanceAction.h>

#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics/ConstraintSolver/Constraint/Chain/hkpChainConstraintInfo.h>
#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.h>

#include <Physics2012/Dynamics/Motion/hkpMotion.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

hkpStiffSpringChainData::hkpStiffSpringChainData() : m_tau(hkReal(0.6f)), m_damping(hkReal(1)), m_cfm(hkReal(0.1f) * HK_REAL_EPSILON) 
{
	m_atoms.m_bridgeAtom.init( this );
}

hkpStiffSpringChainData::hkpStiffSpringChainData(hkFinishLoadedObjectFlag f) : hkpConstraintChainData(f), m_atoms(f), m_infos(f)
{
	if( f.m_finishing )
	{
		m_atoms.m_bridgeAtom.init( this );
	}
}


hkpStiffSpringChainData::~hkpStiffSpringChainData()
{
}

int hkpStiffSpringChainData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN;
}


void hkpStiffSpringChainData::addConstraintInfoInBodySpace(const hkVector4& pivotInA, const hkVector4& pivotInB, hkReal springLength)
{
	ConstraintInfo& info = m_infos.expandOne();
	info.m_pivotInA = pivotInA;
	info.m_pivotInB = pivotInB;
	info.m_springLength = springLength;
}


void hkpStiffSpringChainData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
	info.clear();
	info.addHeader();

	const int numConstraints = this->m_infos.getSize();

	int schemaSize  = hkpJacobianSchemaInfo::StiffSpringChain::Sizeof
					+ numConstraints        * sizeof(hkp1Lin2AngJacobian)
					+ numConstraints        * sizeof(hkpConstraintChainTriple)
					+ (1 + numConstraints)  * sizeof(hkVector4) // temp buff
					+ (1 + numConstraints)  * sizeof(hkpVelocityAccumulatorOffset); // accumulators
	schemaSize		= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, schemaSize);

	int resultsSize = numConstraints;

	info.add(schemaSize, resultsSize, resultsSize);
}


void hkpStiffSpringChainData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const 
{
	if ( wantRuntime )
	{
		int childSolverResultMax = hkpStiffSpringConstraintData::SOLVER_RESULT_MAX;
		infoOut.m_numSolverResults = m_infos.getSize() * childSolverResultMax;
		infoOut.m_sizeOfExternalRuntime = infoOut.m_numSolverResults * sizeof(hkpSolverResults); //* sizeof( Runtime );
	}
	else
	{
		infoOut.m_numSolverResults = 0;
		infoOut.m_sizeOfExternalRuntime = 0;
	}
}

extern "C"
{
	hkp1Lin2AngJacobian* hkJacobianStiffSpringChainSchema_getJacobians(hkpJacobianSchema* schema);
}

void hkpStiffSpringChainData::buildJacobian( const hkpConstraintQueryIn &inNotValid, hkpConstraintQueryOut &out )
{
	hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() ) ;

	inNotValid.beginConstraints( out, solverResults, sizeof(hkpSolverResults) );

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

		// Initialize first body info
		hkpEntity* rB = entities[0]; // yes, body A
		hkpMotion* cB = rB->getMotion();
		newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );
		accumulators.pushBackUnchecked( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );

		newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
		HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

		newIn.m_rhsFactor = newIn.m_subStepInvDeltaTime;
		newIn.m_virtMassFactor = hkReal(1);

		hkp1Lin2AngJacobian* jacobians = hkJacobianStiffSpringChainSchema_getJacobians(out.m_jacobianSchemas.val());

		for (int i = 0; i < numConstraints; i++)
		{
			newIn.m_bodyA      = newIn.m_bodyB;
			newIn.m_transformA = newIn.m_transformB;

			rB = entities[i+1];//  m_constraintInstances[i]->getEntityB();
			cB = rB->getMotion();
			newIn.m_bodyB = hkAddByteOffsetConst(baseAccum, rB->m_solverData );

			accumulators.pushBack( hkpVelocityAccumulatorOffset(baseAccum, newIn.m_bodyB) );


			newIn.m_transformB = &(static_cast<const hkpMotion*>(cB)->getTransform());
			HK_ASSERT( 0xf0140201, &rB->getCollidable()->getTransform() == newIn.m_transformB );

			hkVector4 posA; posA._setTransformedPos( *newIn.m_transformA, m_infos[i].m_pivotInA );
			hkVector4 posB; posB._setTransformedPos( *newIn.m_transformB, m_infos[i].m_pivotInB );

			//
			// Code copied from stiff-spring constraint .cpp
			//

			hkp1dLinearBilateralConstraintInfo bp;
			hkSimdReal springLength;
			{
				bp.m_pivotA._setTransformedPos( *newIn.m_transformA, m_infos[i].m_pivotInA );
				bp.m_pivotB._setTransformedPos( *newIn.m_transformB, m_infos[i].m_pivotInB );

				hkVector4 sepDist;	sepDist.setSub( bp.m_pivotA, bp.m_pivotB );
				springLength = sepDist.normalizeWithLength<3>();

				const hkVector4Comparison springLengthGreaterZero = springLength.greaterZero();
				bp.m_constrainedDofW.setSelect(springLengthGreaterZero, sepDist, hkVector4::getConstant<HK_QUADREAL_1000>());
				springLength.setMax(springLength, hkSimdReal_0);
			}
			{
				// we're ignoring the shema generated by the pulley.
				const hkReal customRhs = m_infos[i].m_springLength - springLength.getReal();
				hk1dLinearBilateralConstraintBuildJacobianWithCustomRhs_noSchema( bp, newIn, jacobians, customRhs); 
				jacobians++;
			}			
		}

		HK_ASSERT2(0xad674d4d, numConstraints == accumulators.getSize() - 1, "Number of chained constraints and number of velocity accumulators don't match.");
		hkStiffSpringChainBuildJacobian( numConstraints, m_tau, m_damping, m_cfm, accumulators.begin(), baseAccum, jacobians, inNotValid, out );
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
