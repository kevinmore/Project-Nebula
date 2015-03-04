/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterState.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterContext.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/Jumping/hkpCharacterStateJumping.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/Util/hkpCharacterMovementUtil.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>

hkpCharacterStateJumping::hkpCharacterStateJumping()
{
	m_jumpHeight = 1.5f;
}

	// Return the state type
hkpCharacterStateType hkpCharacterStateJumping::getType() const 
{
	return HK_CHARACTER_JUMPING;
}
	/// Process the user input - causes state transition.
void hkpCharacterStateJumping::change(hkpCharacterContext& context, const hkpCharacterInput& input, hkpCharacterOutput& output)
{

}

	/// Process the user input - causes state actions.
void hkpCharacterStateJumping::update(hkpCharacterContext& context, const hkpCharacterInput& input, hkpCharacterOutput& output)
{
	//
	// Jump - v^2 = u^2 + 2 a s
	// At apex of jump v^2 = 0 so u = hkMath::sqrt(2 a s);
	//	
	if (context.getNumFramesInCurrentState() == 1)
	{
		hkVector4 impulse = input.m_characterGravity;
		const hkSimdReal	a = impulse.normalizeWithLength<3>();
		hkSimdReal	s; s.setFromFloat( m_jumpHeight ); s.setAbs(s);
		const hkSimdReal	u = ( hkSimdReal_2 * a * s ).sqrt();
		impulse.mul( -u );

		hkVector4 relVelocity; relVelocity.setSub( input.m_velocity, input.m_surfaceInfo.m_surfaceVelocity );
		hkSimdReal curRelVel = relVelocity.dot<3>( input.m_up );

		if (curRelVel < u)
		{
			output.m_velocity.setAddMul(input.m_velocity, input.m_up, (u-curRelVel) );
		}
	}
	else
	{	
		context.setState(HK_CHARACTER_IN_AIR, input, output);
	}
}

hkReal hkpCharacterStateJumping::getJumpHeight() const
{
	return m_jumpHeight;
}

void hkpCharacterStateJumping::setJumpHeight( hkReal jumpHeight )
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
