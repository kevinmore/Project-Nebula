/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateOnGround.h>

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterContext.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/Utils/hknpCharacterMovementUtil.h>


hknpCharacterStateOnGround::hknpCharacterStateOnGround()
	: m_gain(0.95f)
	, m_walkSpeed(10.0f)
	, m_maxLinearAcceleration(200.0f)
	, m_killVelocityOnLaunch(true)
	, m_limitVerticalVelocity(false)
	, m_disableHorizontalProjection(false)
{

}

	// Return the state type
hknpCharacterState::hknpCharacterStateType hknpCharacterStateOnGround::getType() const
{
	return hknpCharacterState::HK_CHARACTER_ON_GROUND;
}

void hknpCharacterStateOnGround::change(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output)
{
	// Check if I should jump
	if (input.m_wantJump)
	{
		context.setState(hknpCharacterState::HK_CHARACTER_JUMPING, input, output);
		return;
	}

	// Check if I should climb
	if (input.m_atLadder)
	{
		context.setState(hknpCharacterState::HK_CHARACTER_CLIMBING, input, output);
		return;
	}

	// Check that I'm supported
	if ( input.m_surfaceInfo.m_supportedState != hknpCharacterSurfaceInfo::SUPPORTED )
	{
		if (m_killVelocityOnLaunch)
		{
			// Suddenly the character is no longer supported.
			// Remove all vertical velocity if required.
			{
				// Remove velocity in vertical component
				output.m_velocity.subMul(input.m_velocity.dot<3>(input.m_up), input.m_up);
			}
		}
		context.setState(hknpCharacterState::HK_CHARACTER_IN_AIR, input, output);
		return;
	}
}

	// Process the user input - causes state transitions etc.
void hknpCharacterStateOnGround::update(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output)
{
	//
	// Move character relative to the surface we're standing on
	//
	{
		// Correct input velocity to apply instantly any changes in the velocity of the standing surface and this way
		// avoid artifacts caused by filtering of the output velocity when standing on moving objects.
		hkVector4& inputVelocity = output.m_inputVelocity;
		{
			hkVector4 previousSurfaceVel = context.getPreviousSurfaceVelocity();
			inputVelocity.setSub(input.m_velocity, previousSurfaceVel);
			inputVelocity.setAdd(input.m_surfaceInfo.m_surfaceVelocity, inputVelocity);
		}

		hknpCharacterMovementUtil::hknpMovementUtilInput muInput;
		muInput.m_gain = m_gain;
		muInput.m_forward = input.m_forward;
		muInput.m_up = input.m_up;
		muInput.m_surfaceNormal = input.m_surfaceInfo.m_surfaceNormal;
		muInput.m_currentVelocity = inputVelocity;
		muInput.m_desiredVelocity.set( input.m_inputUD * m_walkSpeed, input.m_inputLR * m_walkSpeed, 0.f );
		muInput.m_maxVelocityDelta = m_maxLinearAcceleration*input.m_stepInfo.m_deltaTime;
		muInput.m_surfaceVelocity = input.m_surfaceInfo.m_surfaceVelocity;

		// This will stick the character to the surface
		hknpCharacterMovementUtil::calculateMovement(muInput, output.m_velocity);

		hkSimdReal deltaTime; deltaTime.setFromFloat(input.m_stepInfo.m_deltaTime);

		// If the character is too far from the surface, apply gravity along the surface normal.
		if ( input.m_surfaceInfo.m_surfaceDistanceExcess > 0.0f )
		{
			hkSimdReal normalGravity; normalGravity.setMul(input.m_surfaceInfo.m_surfaceNormal.dot<3>( input.m_characterGravity ), deltaTime);
			output.m_velocity.addMul( normalGravity, input.m_surfaceInfo.m_surfaceNormal );
		}

		// Do not let the character fall faster than gravity
		if ( m_limitVerticalVelocity )
		{
			hkSimdReal maxGravityVelocityInc; maxGravityVelocityInc.setMul(input.m_characterGravity.dot<3>(input.m_up), deltaTime);
			hkSimdReal actualVelocityInc; actualVelocityInc.setSub(output.m_velocity.dot<3>(input.m_up), input.m_velocity.dot<3>(input.m_up));

			if (output.m_velocity.dot<3>(input.m_up).isLessZero() && actualVelocityInc.isLess(maxGravityVelocityInc))
			{
				output.m_velocity.addMul((maxGravityVelocityInc - actualVelocityInc), input.m_up);
			}
		}
	}

	if (!m_disableHorizontalProjection)
	{
		output.m_velocity.sub(input.m_surfaceInfo.m_surfaceVelocity);
		hkSimdReal inv1k; inv1k.setFromFloat(.001f);
		if (output.m_velocity.dot<3>(input.m_up).isGreater(inv1k))
		{
			HK_ASSERT2(0x3f96452a, input.m_surfaceInfo.m_surfaceNormal.dot<3>(input.m_up).isGreater(hkSimdReal_Eps), "Surface is vertical and the character is in the walk state.");

			hkSimdReal velLen = output.m_velocity.normalizeWithLength<3>();

				// Get the desired length in the horizontal direction
			hkSimdReal horizLen = velLen / input.m_surfaceInfo.m_surfaceNormal.dot<3>(input.m_up);

				// Re project the velocity onto the horizontal plane
			hkVector4 c; c.setCross(input.m_surfaceInfo.m_surfaceNormal, output.m_velocity);
			output.m_velocity.setCross(c, input.m_up);

				// Scale the velocity to maintain the speed on the slope
			output.m_velocity.mul(horizLen);
		}
		output.m_velocity.add(input.m_surfaceInfo.m_surfaceVelocity);
	}


	// HVK-1362 : The following code is also used in the inAir state.
	// Its purpose is to allow gravity to affect the character by stopping
	// the hknpCharacterMovementUtil culling velocity in the direction of m_up.
	// However leaving this code in means that characters can start sliding
	// down gentle slopes.
	// The more common use case is that a character should not slide at all
	// on slopes that are shallow enough to support them. Therefore the
	// following lines have been commented out.

	// Restore to original vertical component
	//output.m_velocity.addMul4(-output.m_velocity.dot3(input.m_up), input.m_up);
	//output.m_velocity.addMul4(  input.m_velocity.dot3(input.m_up) , input.m_up);
}

hkReal hknpCharacterStateOnGround::getGain() const
{
	return m_gain;
}

void hknpCharacterStateOnGround::setGain(hkReal newGain)
{
	m_gain = newGain;
}

hkReal hknpCharacterStateOnGround::getSpeed() const
{
	return m_walkSpeed;
}

void hknpCharacterStateOnGround::setSpeed(hkReal newSpeed)
{
	m_walkSpeed = newSpeed;
}

hkReal hknpCharacterStateOnGround::getMaxLinearAcceleration() const
{
	return m_maxLinearAcceleration;
}

void hknpCharacterStateOnGround::setMaxLinearAcceleration(hkReal newMaxAcceleration)
{
	m_maxLinearAcceleration = newMaxAcceleration;
}


hkBool hknpCharacterStateOnGround::getGroundHugging() const
{
	return m_killVelocityOnLaunch;
}

void hknpCharacterStateOnGround::setGroundHugging(hkBool newVal)
{
	m_killVelocityOnLaunch = newVal;
}


hkBool hknpCharacterStateOnGround::getLimitDownwardVelocity() const
{
	return m_limitVerticalVelocity;
}

void hknpCharacterStateOnGround::setLimitDownwardVelocity( hkBool newVal )
{
	m_limitVerticalVelocity = newVal;
}

hkBool hknpCharacterStateOnGround::getDisableHorizontalProjection() const
{
	return m_disableHorizontalProjection;
}

void hknpCharacterStateOnGround::setDisableHorizontalProjection( hkBool newVal )
{
	m_disableHorizontalProjection = newVal;
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
