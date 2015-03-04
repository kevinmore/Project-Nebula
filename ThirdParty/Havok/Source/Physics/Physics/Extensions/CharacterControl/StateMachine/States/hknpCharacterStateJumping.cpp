/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateJumping.h>

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterContext.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/Utils/hknpCharacterMovementUtil.h>


hknpCharacterStateJumping::hknpCharacterStateJumping()
	: m_jumpHeight(1.5f)
{

}

// Return the state type
hknpCharacterState::hknpCharacterStateType hknpCharacterStateJumping::getType() const
{
	return hknpCharacterState::HK_CHARACTER_JUMPING;
}

	/// Process the user input - causes state transition.
void hknpCharacterStateJumping::change(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output)
{

}

	/// Process the user input - causes state actions.
void hknpCharacterStateJumping::update(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output)
{
	//
	// Jump - v^2 = u^2 + 2 a s
	// At apex of jump v^2 = 0 so u = hkMath::sqrt(2 a s);
	//
	if (context.getNumUpdatesInCurrentState() == 1)
	{
		hkVector4 impulse = input.m_characterGravity;
		const hkSimdReal a = impulse.normalizeWithLength<3>();
		hkSimdReal s; s.setFromFloat(hkMath::fabs( m_jumpHeight ));
		hkSimdReal u = hkSimdReal_2*a*s; u = u.sqrt();
		impulse.mul( -u );

		hkVector4 relVelocity; relVelocity.setSub( input.m_velocity, input.m_surfaceInfo.m_surfaceVelocity );
		hkSimdReal curRelVel = relVelocity.dot<3>( input.m_up );

		if (curRelVel.isLess( u ))
		{
			output.m_velocity.setAddMul(input.m_velocity, input.m_up, u-curRelVel );
		}
	}
	else
	{
		context.setState(hknpCharacterState::HK_CHARACTER_IN_AIR, input, output);
	}
}

hkReal hknpCharacterStateJumping::getJumpHeight() const
{
	return m_jumpHeight;
}

void hknpCharacterStateJumping::setJumpHeight( hkReal jumpHeight )
{
	m_jumpHeight = jumpHeight;
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
