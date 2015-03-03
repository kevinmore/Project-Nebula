/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_POWERED_CHAIN_BUILD_AND_UPDATE_JACOBIAN_H
#define HKP_POWERED_CHAIN_BUILD_AND_UPDATE_JACOBIAN_H

#include <Physics/ConstraintSolver/Solve/hkpSolverInfo.h>
#include <Physics/ConstraintSolver/Constraint/Chain/hkpChainConstraintInfo.h>

class hkp1Lin2AngJacobian;
class hkp2AngJacobian;
class hkpVelocityAccumulator;
struct hkSolverStepTemp;


struct hkpConstraintChainMatrix6Triple
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpConstraintChainMatrix6Triple );

	hkMatrix6 m_lower;
	hkMatrix6 m_diagonalInv;
	hkMatrix6 m_upper;

	// Diagonal block of the original constraint matrix.
	hkMatrix6 m_mtxDiag;
	// Upper diagonal block of the original constraint matrix. (This block is located to the right of m_mtxDiag, and it is zero for the last row of the matrix.)
	hkMatrix6 m_mtxNextOffdiag;
};


struct hkpChainSolverInfo
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpChainSolverInfo );

	hkpChainSolverInfo( const hkpSolverInfo& solverInfo ) : m_solverInfo(solverInfo) {}

	const hkpSolverInfo& m_solverInfo;

	hkPadSpu<int> m_numConstraints;
	hkSimdReal m_schemaTau;
	hkSimdReal m_schemaDamping;

	hkPadSpu<hkp1Lin2AngJacobian*> m_j;
	hkPadSpu<hkp2AngJacobian*> m_jAng;
	hkPadSpu<hkpVelocityAccumulatorOffset*> m_va;
	hkPadSpu<hkpVelocityAccumulator*> m_accumsBase;
	hkPadSpu<hkpConstraintChainMatrix6Triple*> m_triples;
	hkPadSpu<hkp3dAngularMotorSolverInfo*> m_motorsState;


	hkPadSpu<hkVector8*> m_gTempBuffer;
	hkPadSpu<hkVector8*> m_velocitiesBuffer;
};


/// This structure holds a list of cfm parameters for a hkPoweredChain link
struct hkpCfmParam
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpCfmParam );

		// Additive cfm parameter for the linear velocity equations
	hkReal m_linAdd;

		// Multiplicative cfm parameter for the linear velocity equations
	hkReal m_linMul;

		// Additive cfm parameter for the angular velocity equations
	hkReal m_angAdd;

		// Multiplicative cfm parameter for the angular velocity equations
	hkReal m_angMul;
};


// This structure holds a list of parameters required to build chain's jacobians.
struct hkpPoweredChainBuildJacobianParams
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpPoweredChainBuildJacobianParams );

	int m_numConstraints;
	hkReal m_chainTau;
	hkReal m_chainDamping;
	hkpCfmParam m_cfm;
	hkpVelocityAccumulatorOffset* m_accumulators;
	const hkpVelocityAccumulator* m_accumsBase;
	hkp3dAngularMotorSolverInfo* m_motorsState;
	hkReal m_maxTorqueHysterisys;
	hkp3dAngularMotorSolverInfo::Status* m_childConstraintStatusWriteBackBuffer;
	hkp2AngJacobian* m_jacobiansEnd;
};


extern "C"
{
	void HK_CALL hkPoweredChainBuildJacobian( const hkpPoweredChainBuildJacobianParams& params, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	//
	//  Building constraint matrix and computing LU decomposition.
	//
	//  also updating LU decomposition from pre-calculated constraint matrix.
	//
	void hkPoweredChain_BuildConstraintMatrixAndLuDecomposition(int numConstraints, const hkpCfmParam& cfm, hkp3dAngularMotorSolverInfo* motorsState, hkp1Lin2AngJacobian* j, hkp2AngJacobian* jAng, hkpVelocityAccumulatorOffset* va, const hkpVelocityAccumulator* accumsBase, hkpConstraintChainMatrix6Triple* triples, int bufferSize);

	void hkPoweredChain_UpdateLuDecomposition(int firstConsraintToProcess, int numConstraints, hkp3dAngularMotorSolverInfo* motorsState, hkpConstraintChainMatrix6Triple* triple);

	/*HK_FORCE_INLINE*/
	void hkPoweredChain_ComputeConstraintMatrixLuDecomposition_ForOneRow( hkBool isNotLastRow, hkpConstraintChainMatrix6Triple& triple, hkp3dAngularMotorSolverInfo* motorsState, hkMatrix6& mtxPrevOffdiag, hkMatrix6*& triplePrevUpper );

	//
	//  Computing constraint matrix blocks
	//
	HK_FORCE_INLINE void hkPoweredChain_ComputeConstraintMatrix_DiagonalAtRow(int row, const hkpCfmParam& cfm, hkp1Lin2AngJacobian* j, hkp2AngJacobian* jAng, hkpVelocityAccumulatorOffset* va, const hkpVelocityAccumulator* accumsBase, hkMatrix6& mtxDiag );

	HK_FORCE_INLINE void hkPoweredChain_ComputeConstraintMatrix_NextOffDiagonalAtRow(int row, hkp1Lin2AngJacobian* j, hkp2AngJacobian* jAng, hkpVelocityAccumulatorOffset* va, const hkpVelocityAccumulator* accumsBase, hkMatrix6& mtxNextOffdiag );

	//
	//  Disabling of angular parts of constraints in the constraint matrix blocks
	//
	void HK_CALL hkPoweredChain_DisableMotorInMatrixRow_ThisConstraint(const hkp3dAngularMotorSolverInfo& motorsState, hkBool isNotLastRow, hkMatrix6& mtxDiag, hkMatrix6& mtxNextOffdiag);

	void HK_CALL hkPoweredChain_DisableMotorInMatrixRow_NextConstraint(const hkp3dAngularMotorSolverInfo& motorsState, hkMatrix6& mtxNextOffdiag);

	//
	// Constraint solving
	//
	void HK_CALL hkPoweredChain_CalculateVelocities(const hkSolverStepTemp& temp, const hkpChainSolverInfo& info, hkVector8* velocities);

	void HK_CALL hkPoweredChain_OverwriteVelocityWithExplicitImpulse(int modifiedConstraintIndex, int modifiedCoordinateIndex, const hkp3dAngularMotorSolverInfo* motorsState, hkVector8* velocities);

	void HK_CALL hkPoweredChain_RestoreVelocityValue(int modifiedConstraintIndex, int modifiedCoordinateIndex, const hkpChainSolverInfo& info, hkVector8* velocities);

	void HK_CALL hkPoweredChain_SolveConstraintMatrix_ComputeVectorG(const hkpChainSolverInfo& info, hkVector8* tempBufferG );

	void HK_CALL hkPoweredChain_ScanAndDisableMotors(const hkpChainSolverInfo& info, int& modifiedChildConstraintIndex, int& modifiedCoordianteIndex, hkReal& impulse, hkp3dAngularMotorSolverInfo* /*output*/ motorsState );

	void HK_CALL hkPoweredChain_ScanAndEnableMotors(const hkpChainSolverInfo& info, int& modifiedChildConstraintIndex, int& modifiedCoordianteIndex, hkReal& impulse, hkp3dAngularMotorSolverInfo* /*output*/ motorsState );

	HK_ON_DEBUG(void HK_CALL hkPoweredChain_VerifyVelocities(const hkpChainSolverInfo& info, hkVector8* velocities2 ));
}


#endif // HKP_POWERED_CHAIN_BUILD_AND_UPDATE_JACOBIAN_H

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
