/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Utilities/Dynamics/TimeSteppers/hkpVariableTimestepper.h>

hkpVariableTimestepper::hkpVariableTimestepper( hkReal timePerSubstep, hkReal minSubstepCount ) :
	m_timePerSubstep( timePerSubstep ), m_minSubstepCount( minSubstepCount )
{
	
}


int hkpVariableTimestepper::step( hkpWorld* world, hkReal timestep )
{

	// The idea here is to try to keep the size of the substeps a constant. We vary the number of substeps.
	// We scale tau and damping to account for fractional substeps. In the end the substep size may vary
	// when stepDeltaTime is called, but with tau/damping scaling the effect should be the same as if
	// we took substeps of a size that remains constant over frames.

	// Take ceil to make sure we always scale tau down, which is safer.
	hkReal numSubsteps = hkMath::ceil(timestep / m_timePerSubstep);

	hkReal timePerSubstep = m_timePerSubstep;
	
	if( numSubsteps < m_minSubstepCount )
	{
		numSubsteps = m_minSubstepCount;
		timePerSubstep = timestep/m_minSubstepCount;
	}

	HK_ASSERT2(0x34ca5d09,  (timestep / (numSubsteps*timePerSubstep)) <= 1.01f , "increasing tau is dangerous!" );

	// scale Tau and Damping to interpolate stiffness factors
	hkReal originalTau = world->m_dynamicsStepInfo.m_solverInfo.m_tau;
	hkReal scaledTau = originalTau + ( timestep / (numSubsteps*timePerSubstep) - 1.0f );

	hkReal originalDamping = world->m_dynamicsStepInfo.m_solverInfo.m_damping;
	hkReal scaledDamping = originalDamping + ( timestep / (numSubsteps*timePerSubstep) - 1.0f );

	world->m_dynamicsStepInfo.m_solverInfo.m_tau = scaledTau;
	world->m_dynamicsStepInfo.m_solverInfo.m_damping = scaledDamping;
	
	world->m_dynamicsStepInfo.m_solverInfo.m_numSteps = hkMath::hkFloatToInt( numSubsteps );
	world->m_dynamicsStepInfo.m_solverInfo.m_invNumSteps = 1.0f/world->m_dynamicsStepInfo.m_solverInfo.m_numSteps;

	// step it
	world->stepDeltaTime(timestep);

	world->m_dynamicsStepInfo.m_solverInfo.m_tau = originalTau;
	world->m_dynamicsStepInfo.m_solverInfo.m_damping = originalDamping;

	return world->m_dynamicsStepInfo.m_solverInfo.m_numSteps;
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
