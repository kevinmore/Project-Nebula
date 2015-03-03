/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

// this
#include <Physics/Physics/Dynamics/Solver/hknpSolverInfo.h>


#if !defined(HK_PLATFORM_SPU)

hknpSolverInfo::hknpSolverInfo()
{
	setTauAndDamping( 0.5f, 0.9f );
	hkVector4 gravity; gravity.set( 0,-9.81f, 0);
	setStepInfo( gravity, 0.1f, 0.016f, 4, 1 );
	m_maxApproachSpeedForHighQualitySolver = 0.1f;		// meters per second
	setMaxTimDistance(100.0f);
	m_stepSolveCount = 0;
}

#endif


void hknpSolverInfo::setMaxTimDistance(hkReal d)
{
	m_distanceToLinearTim.setFromFloat( hkReal(0x7fff) / d );
	m_linearTimToDistance.setReciprocal( m_distanceToLinearTim );
}

void hknpSolverInfo::setTauAndDamping( hkReal tau, hkReal damping )
{
	m_tau = tau;
	m_frictionTau = 0.5f * tau;
	m_damping.setFromFloat(damping);
	m_dampDivTau = damping / tau;
	m_dampDivFrictionTau = damping / m_frictionTau;
	m_tauDivDamp = tau/damping;
	m_frictionTauDivDamp = m_frictionTau/damping;
	m_integrateVelocityFactor.setAll( m_tauDivDamp );
	m_integrateVelocityFactor.zeroComponent<3>();
	m_invIntegrateVelocityFactor.setAll( m_dampDivTau );
	m_invIntegrateVelocityFactor.zeroComponent<3>();

	// Precompute helper variables for hknpContactSolverSetup::buildContactJacobianForSingleManifold().
	m_rhsFactor.setFromFloat(m_tauDivDamp * m_subStepInvDeltaTime.getReal());
	m_frictionRhsFactor.setFromFloat( -m_frictionTauDivDamp * m_subStepInvDeltaTime.getReal());
}

void hknpSolverInfo::setStepInfo( hkVector4Parameter gravity, hkReal collisionTolerance, hkReal dt, int numSubsteps, int numMicrosteps )
{
	m_collisionTolerance.setFromFloat( collisionTolerance );
	m_numSteps = numSubsteps;
	m_invNumSteps = 1.0f / hkReal(numSubsteps);

	m_numMicroSteps = numMicrosteps;
	m_invNumMicroSteps = 1.0f / hkReal(numMicrosteps);

	m_expectedDeltaTime = dt;

	hkSimdReal dtSr; dtSr.setFromFloat(dt);

	m_deltaTime.setFromFloat(dt);
	m_invDeltaTime.setFromFloat(1.0f/dt);

	m_subStepDeltaTime.setFromFloat(dt * m_invNumSteps);

	hkSimdReal numSteps; numSteps.setFromInt32(m_numSteps);
	m_subStepInvDeltaTime = m_invDeltaTime * numSteps;

	m_globalAccelerationPerSubStep.setMul( m_subStepDeltaTime, gravity );
	m_globalAccelerationPerStep.setMul( m_deltaTime, gravity );
	m_globalAccelerationPerStep.setW( m_globalAccelerationPerStep.length<3>() );

	// Precompute helper variables for hknpContactSolverSetup::buildContactJacobianForSingleManifold().
	m_rhsFactor.setFromFloat( m_tauDivDamp * m_subStepInvDeltaTime.getReal() );
	m_frictionRhsFactor.setFromFloat( -m_frictionTauDivDamp * m_subStepInvDeltaTime.getReal());

	hkSimdReal gravityMagnitude = gravity.length<3>();
	gravityMagnitude.store<1>( &m_nominalGravityLength );
	if ( gravityMagnitude.isEqualZero() )
	{
		m_nominalGravityLength = 9.81f;
	}
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
