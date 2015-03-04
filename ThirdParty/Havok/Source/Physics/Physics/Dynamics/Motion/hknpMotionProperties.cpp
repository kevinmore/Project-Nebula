/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionProperties.h>

#define hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE hkReal(0.05f)
#define hknpMotionProperties_MAX_ROTATION_IN_PI_RAD hkReal(0.6f)
#define hknpMotionProperties_DEFAULT_MINIMUM_PATHING_VELOCITY hkReal(0.035f) // 2*hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE * 60/128 * 1.5/2
#define hknpMotionProperties_REFERENCE_DIST_TO_WRAPPING_BLOCK_SIZE_RATIO hkReal(20)

HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) % 16 == 0);
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotionProperties) % 16 == 0);
#if !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) <= 128);
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotion) >= 128);
#endif


void hknpMotionProperties::setPreset( hknpMotionPropertiesId::Preset preset, hkReal gravityHint, hkReal unitScaleHint )
{
	HK_ASSERT2(0x7f3f0515, preset >= 0 && preset < hknpMotionPropertiesId::NUM_PRESETS, "Invalid preset");

	HK_ON_DEBUG( const hkBool32 isExclusive = m_isExclusive );

	// Clear everything except the exclusivity flag
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF( hknpMotionProperties, m_isExclusive ) == 0 );
	hkString::memSet( hkAddByteOffset(this, hkSizeOf(m_isExclusive)), 0, hkSizeOf(*this) - hkSizeOf(m_isExclusive) );

	switch(preset)
	{
	case hknpMotionPropertiesId::STATIC:
		{
			m_linearDamping = 0.0f;
			m_angularDamping = 0.0f;
			m_maxLinearSpeed = 0.0f;
			m_maxAngularSpeed = 0.0f;
			m_gravityFactor = 0.0f;
			setSolverStabilization( hknpMotionProperties::SOLVER_STABILIZATION_OFF, gravityHint );
		}
		break;

	case hknpMotionPropertiesId::DYNAMIC:
		{
			m_linearDamping = 0.0f;
			m_angularDamping = 0.0f;
			m_maxLinearSpeed =  200.0f * unitScaleHint;
			m_maxAngularSpeed = 100.0f;
			m_gravityFactor = 1.0f;
			setSolverStabilization( hknpMotionProperties::SOLVER_STABILIZATION_OFF, gravityHint );
		}
		break;

	case hknpMotionPropertiesId::KEYFRAMED:
		{
			m_linearDamping = 0.0f;
			m_angularDamping = 0.0f;
			m_maxLinearSpeed = 200.0f * unitScaleHint;
			m_maxAngularSpeed = 100.0f;
			m_gravityFactor = 0.0f;
			setSolverStabilization( hknpMotionProperties::SOLVER_STABILIZATION_OFF, gravityHint );
		}
		break;

	case hknpMotionPropertiesId::FROZEN:
		{
			m_linearDamping  = 30.0f;
			m_angularDamping = 30.0f;
			m_maxLinearSpeed = 0.0f;
			m_maxAngularSpeed = 0.0f;
			m_gravityFactor = 0.0f;
			setSolverStabilization( hknpMotionProperties::SOLVER_STABILIZATION_OFF, gravityHint );
		}
		break;

	case hknpMotionPropertiesId::DEBRIS:
		{
			m_linearDamping = 0.01f;
			m_angularDamping = 0.5f;
			m_maxLinearSpeed  = 100.0f * unitScaleHint;
			m_maxAngularSpeed = 10.0f;
			m_gravityFactor = 1.0f;
			setSolverStabilization( hknpMotionProperties::SOLVER_STABILIZATION_HIGH, gravityHint );
		}
		break;

	default:
		// Should not get here.
		HK_ASSERT(0x5601b82d, false);
		break;
	}

	setDeactivationParameters();

	HK_ASSERT(0x21f96db1, isExclusive == m_isExclusive );
}

void hknpMotionProperties::setSolverStabilization( SolverStabilizationType type, hkReal gravity )
{
	gravity = hkMath::fabs( gravity );
	if (gravity == 0.0f)
	{
		gravity = 10.0f;
	}

	hkReal relVelocityThres;  // relative to gravity*1sec
	hkReal relDeceleration;	  // factor of the gravity at relVelocityThres
	switch (type)
	{
	default:
	case SOLVER_STABILIZATION_OFF:
		relVelocityThres   = 0.0f;
		relDeceleration    = 0.0f;
		break;
	case SOLVER_STABILIZATION_LOW:
		relVelocityThres   = 0.1f;   // = 10cm/sec
		relDeceleration    = 0.03f;
		break;
	case SOLVER_STABILIZATION_MEDIUM:
		relVelocityThres   = 0.17f;   // = 17cm/sec
		relDeceleration    = 0.05f;
		break;
	case SOLVER_STABILIZATION_HIGH:
		relVelocityThres   = 0.2f;   // = 20cm/sec
		relDeceleration    = 0.10f;
		break;
	case SOLVER_STABILIZATION_AGGRESSIVE:
		relVelocityThres   = 0.4f;   // = 25cm/sec
		relDeceleration    = 0.2f;
		break;
	}

	m_solverStabilizationSpeedThreshold = relVelocityThres;
	m_solverStabilizationSpeedReduction = gravity * relDeceleration;
}

void hknpMotionProperties::setDeactivationParameters(
	hkReal referenceDistance, hkReal referenceRotation, DeactivationStrategy deactivationStrategy )
{
	//static const hkReal referenceDistances[] = { 2.0f, 1.0f, 0.5f, 0.1f, 0.05f, 0.03f };
	//hkReal referenceDistance = referenceDistances[deactivationJitterScale];

	const hkReal deactivationReferenceDistance = referenceDistance;
	hkReal q = deactivationReferenceDistance;

	m_maxDistSqrd = q * q;
	hkReal angleFraction = hkMath::min2( referenceRotation / HK_REAL_PI, hknpMotionProperties_MAX_ROTATION_IN_PI_RAD ); //assumption: 180 degrees rotation corresponds to a quaternion difference of 1 after subtraction (hugely approximate: it does not scale linearly and it is even incorrect for most pairs of 180-degree rotated quaternions)
	m_maxRotSqrd = angleFraction * angleFraction;
	m_invBlockSize = 1.0f / (q * hknpMotionProperties_REFERENCE_DIST_TO_WRAPPING_BLOCK_SIZE_RATIO);

	m_numDeactivationFrequencyPasses = hkUint8(deactivationStrategy);

	// To obtain the minimumPathingVelocity, multiply DEFAULT_DEACTIVATION_VELOCITY by the same factor as hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE.
	// Additionally, increase DEFAULT_DEACTIVATION_VELOCITY by a factor of 2 for every increase in the aggressiveness of deactivation.
	hkReal minimumPathingVelocity;
	{
		minimumPathingVelocity = hknpMotionProperties_DEFAULT_MINIMUM_PATHING_VELOCITY * (referenceDistance / hknpMotionProperties_DEFAULT_REFERENCE_DISTANCE);
		int numDeactivationFrequencyPassesToMaxAccuracy = DEACTIVATION_STRATEGY_BALANCED - m_numDeactivationFrequencyPasses;
		if( numDeactivationFrequencyPassesToMaxAccuracy >= 0 ) //only increase the pathing velocity, do not decrease it - in order to increase accuracy instead
		{
			hkReal twoPowerToMaxAccuracy = hkReal( 1 << numDeactivationFrequencyPassesToMaxAccuracy );
			minimumPathingVelocity *= twoPowerToMaxAccuracy;
		}

		hkReal minimumPathingVelocitySquare = minimumPathingVelocity * minimumPathingVelocity;
		m_minimumPathingVelocityScaleSquare = hkUint8((*((hkUint32*)&minimumPathingVelocitySquare) & 0x7F800000) >> 23);
	}

	setAdvancedDeactivationParameters(minimumPathingVelocity*0.5f);
}

void hknpMotionProperties::setAdvancedDeactivationParameters(
	hkReal deactivationVelocity, hkReal pathingUpperThreshold, hkReal pathingLowerThreshold,
	hkUint8 spikingVelocityThreshold, hkReal minimumSpikingVelocity )
{
	HK_ASSERT(0x6644296D, pathingUpperThreshold >= -1.0f && pathingUpperThreshold <= 1.0f);

	hkReal deactivationVelocitySquare = deactivationVelocity * deactivationVelocity;
	m_deactivationVelocityScaleSquare = hkUint8((*((hkUint32*)&deactivationVelocitySquare) & 0x7F800000) >> 23);
	m_deactivationVelocityScaleSquare = hkMath::min2(m_deactivationVelocityScaleSquare, m_minimumPathingVelocityScaleSquare);

	hkReal pathingThresholdScale = hkReal(0x7FFF);

	m_pathingUpperThreshold = hkUint16(pathingUpperThreshold * pathingThresholdScale);
	m_pathingLowerThreshold = hkUint16(pathingLowerThreshold * pathingThresholdScale);

	m_spikingVelocityScaleThresholdSquared = spikingVelocityThreshold * 2; //exponent of the squared velocity, so multiply by 2
	hkReal minimumSpikingVelocitySquared = minimumSpikingVelocity * minimumSpikingVelocity;
	hkUint32 currentVelocityScaleSquared = (*((hkUint32*)&minimumSpikingVelocitySquared) & 0x7F800000) >> 23;
	m_minimumSpikingVelocityScaleSquared = hkUint8(currentVelocityScaleSquared);
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
