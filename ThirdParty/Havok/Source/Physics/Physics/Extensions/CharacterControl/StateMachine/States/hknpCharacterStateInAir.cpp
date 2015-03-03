/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateInAir.h>

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/Utils/hknpCharacterMovementUtil.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterContext.h>


hknpCharacterStateInAir::hknpCharacterStateInAir()
	: m_gain(0.05f)
	, m_airSpeed(10.0f)
	// maximal linear acceleration inAir state is set to 25% of max linear acceleration in onGround state
	, m_maxLinearAcceleration(50.0f)
{

}

/// Return the state type
hknpCharacterState::hknpCharacterStateType hknpCharacterStateInAir::getType() const
{
	return hknpCharacterState::HK_CHARACTER_IN_AIR;
}

/// Process the user input - causes state transitions.
void hknpCharacterStateInAir::change(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output)
{
	// Grab details about the surface I'm currently on
	if ( input.m_surfaceInfo.m_supportedState == hknpCharacterSurfaceInfo::SUPPORTED )
	{
		context.setState(hknpCharacterState::HK_CHARACTER_ON_GROUND, input, output);
		return;
	}

	// Check if I should climb
	if (input.m_atLadder)
	{
		context.setState(hknpCharacterState::HK_CHARACTER_CLIMBING, input, output);
		return;
	}

}

/// Process the user input - causes state actions.
void hknpCharacterStateInAir::update(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output)
{
	//
	// Move character relative to the surface we're standing on
	//
	{
		hknpCharacterMovementUtil::hknpMovementUtilInput muInput;
		muInput.m_gain = m_gain;
		muInput.m_forward = input.m_forward;
		muInput.m_up = input.m_up;
		muInput.m_surfaceNormal = input.m_up;
		muInput.m_currentVelocity = input.m_velocity;
		muInput.m_desiredVelocity.set( input.m_inputUD * m_airSpeed, input.m_inputLR * m_airSpeed, 0.f);
		muInput.m_maxVelocityDelta = m_maxLinearAcceleration*input.m_stepInfo.m_deltaTime;
		muInput.m_surfaceVelocity.setZero();

		hknpCharacterMovementUtil::calculateMovement(muInput, output.m_velocity);

		// Restore to original vertical component
		output.m_velocity.addMul(-output.m_velocity.dot<3>(input.m_up), input.m_up);
		output.m_velocity.addMul(  input.m_velocity.dot<3>(input.m_up) , input.m_up);

		hkSimdReal deltaTime; deltaTime.setFromFloat( input.m_stepInfo.m_deltaTime );
		output.m_velocity.addMul(deltaTime, input.m_characterGravity );
	}
		// Hacky workaround for http://gcc.gnu.org/bugzilla/show_bug.cgi?id=44578
#if defined(HK_PLATFORM_LINUX) && defined(HK_COMPILER_GCC) && (HK_COMPILER_GCC_VERSION <= 45000) && HK_CONFIG_SIMD==HK_CONFIG_SIMD_ENABLED
	_mm_empty();
#endif
}

hkReal hknpCharacterStateInAir::getGain() const
{
	return m_gain;
}

void hknpCharacterStateInAir::setGain(hkReal newGain)
{
	m_gain = newGain;
}

hkReal hknpCharacterStateInAir::getSpeed() const
{
	return m_airSpeed;
}

void hknpCharacterStateInAir::setSpeed(hkReal newSpeed)
{
	m_airSpeed = newSpeed;
}

hkReal hknpCharacterStateInAir::getMaxLinearAcceleration() const
{
	return m_maxLinearAcceleration;
}

void hknpCharacterStateInAir::setMaxLinearAcceleration(hkReal newMaxAcceleration)
{
	m_maxLinearAcceleration = newMaxAcceleration;
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
