/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Solver/hknpConstraintSolver.h>

#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>


void hknpConstraintSolver::allocateTemps(const hknpConstraintSolverJacobianRange& jacobians, hkBlockStreamBase::Range& temps,  hknpConstraintSolverJacobianWriter* tempsWriter)
{
	HK_ASSERT2( 0xef22eedd, false, "allocateTemps should not be called if the solver does not implement it" );
}


void hknpSolverStep::init( const hknpSolverInfo& solverInfo, int iStep, int iMicroStep )
{
	hkSimdReal isr; isr.setFromInt32( iStep );
	m_currentStep = iStep;
	m_intregratePositionFactor = hkSimdReal_1 + isr * solverInfo.m_integrateVelocityFactor.getComponent<0>();

	m_flags = 0;
	m_flags |= ( (iStep == 0) && (iMicroStep == 0) ) ? hknpSolverStep::FIRST_ITERATION : 0;
	m_flags |= ( (iStep == solverInfo.m_numSteps-1) && (iMicroStep == solverInfo.m_numMicroSteps-1) ) ? hknpSolverStep::LAST_ITERATION : 0;
}

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
