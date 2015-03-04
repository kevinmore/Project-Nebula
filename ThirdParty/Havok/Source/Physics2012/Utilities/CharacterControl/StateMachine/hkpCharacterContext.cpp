/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterState.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterStateManager.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpCharacterContext.h>


hkpCharacterContext::hkpCharacterContext(const hkpCharacterStateManager* Manager, hkpCharacterStateType initialState)
:	m_stateManager(Manager),
	m_currentState(initialState),
	m_previousState(initialState),
	m_numFramesInCurrentState(0)
{
	m_characterType = HK_CHARACTER_PROXY;

	m_previousSurfaceVelocity.setZero();

	m_filterEnable = true;
	
	m_maxLinearVelocity = 20.0f;

	m_maxLinearAcceleration = 625.0f;

	m_gain = 1.0f;

	m_stateManager->addReference();
}

hkpCharacterContext::~hkpCharacterContext()
{
	m_stateManager->removeReference();
}

hkpCharacterStateType hkpCharacterContext::getState() const
{
	return m_currentState;
}

void hkpCharacterContext::setState(hkpCharacterStateType state, const hkpCharacterInput& input, hkpCharacterOutput& output )
{
	hkpCharacterStateType prevState = m_currentState;

	HK_ASSERT3(0x6d4abe44, m_stateManager->getState(state) != HK_NULL , "Bad state transition to " << state << ". Has this state been registered?");

	// Leave the old state
	m_stateManager->getState(m_currentState)->leaveState(*this, state, input, output);

	// Transition to the new state
	m_currentState = state;

	// Enter the new state
	m_stateManager->getState(m_currentState)->enterState(*this, prevState, input, output);

	m_numFramesInCurrentState = 0;	
}

void hkpCharacterContext::update(const hkpCharacterInput& input, hkpCharacterOutput& output)
{
	HK_ASSERT2(0x6d4abe44, input.isValid(), "Invalid character input");	

	// Give sensible initialized values for output
	output.m_velocity = input.m_velocity;
	output.m_inputVelocity = input.m_velocity;
	
	// ORIGINAL IMPLEMENTATION
	//m_stateManager->getState(m_currentState)->update(*this, input, output);

	// NEW IMPLEMENTATION A
	// Do state transition logic
	// m_stateManager->getState(m_currentState)->change(*this, input, output);
	// Do update with potential new state (straight after the state transition )
	// m_stateManager->getState(m_currentState)->update(*this, input, output);

	// NEW IMPLEMENTATION B
	// Do state transition logic
	m_stateManager->getState(m_currentState)->change(*this, input, output);
	// Do update with potential new state (skip one frame - similar to original implementation)
	if (m_currentState == m_previousState)
	{
		m_stateManager->getState(m_currentState)->update(*this, input, output);			
	}	
	
	m_previousState = m_currentState;

	// Keep track of the previous surface velocity in case any state needs it (e.g to calculate the 
	// variation in the velocity relative to the surface)
	m_previousSurfaceVelocity = input.m_surfaceInfo.m_surfaceVelocity;

	// Apply output velocity filtering
	if (m_filterEnable)
	{
		// limit output velocity
		const hkSimdReal currentVel = output.m_velocity.length<3>();
		const hkSimdReal maxLinVel = hkSimdReal::fromFloat(m_maxLinearVelocity);
		if (currentVel > maxLinVel)
		{
			const hkSimdReal cv = maxLinVel / currentVel;	

			output.m_velocity.mul(cv);
		}
	
		// limit maximal linear acceleration and smooth velocity
		hkVector4 currentAcceleration; currentAcceleration.setSub(output.m_velocity, output.m_inputVelocity);
		hkReal invDt = input.m_stepInfo.m_invDeltaTime;
		currentAcceleration.mul(hkSimdReal::fromFloat(invDt));
		
		const hkSimdReal ca = currentAcceleration.length<3>() / hkSimdReal::fromFloat(m_maxLinearAcceleration);
		
		output.m_velocity = output.m_inputVelocity;

		const hkSimdReal cc = hkSimdReal::fromFloat(m_gain*input.m_stepInfo.m_deltaTime);
		if ( ca.isGreater(hkSimdReal_1) && !(m_currentState == HK_CHARACTER_JUMPING))
		{
			output.m_velocity.addMul(cc/ca,currentAcceleration);		
		}
		else
		{
			output.m_velocity.addMul(cc,currentAcceleration);
		}
	}

	m_numFramesInCurrentState++;
}

void hkpCharacterContext::setFilterEnable(const hkBool status)
{
	m_filterEnable = status;
}

void hkpCharacterContext::setFilterParameters(const hkReal gain, const hkReal maxVelocity, const hkReal maxAcceleration)
{
	m_gain = gain;
	
	m_maxLinearVelocity = maxVelocity;

	m_maxLinearAcceleration = maxAcceleration;
}

void hkpCharacterContext::setCharacterType(const CharacterType newType)
{
	m_characterType = newType;
}

const hkVector4& hkpCharacterContext::getPreviousSurfaceVelocity() const
{
	return m_previousSurfaceVelocity;
}

int hkpCharacterContext::getNumFramesInCurrentState() const
{
	return m_numFramesInCurrentState;
}

hkBool hkpCharacterInput::isValid() const
{		
#define CHECK(X) if( !(X) ) return false
	CHECK( hkMath::isFinite(m_inputLR) );
	CHECK( hkMath::isFinite(m_inputUD) );
	CHECK( m_up.isOk<3>() );
	CHECK( m_up.isNormalized<3>() );
	CHECK( m_forward.isOk<3>() );
	CHECK( m_surfaceInfo.isValid() );
	CHECK( hkMath::isFinite(m_stepInfo.m_deltaTime) );
	CHECK( hkMath::isFinite(m_stepInfo.m_invDeltaTime) );
	CHECK( m_position.isOk<3>() );
	CHECK( m_velocity.isOk<3>() );
	CHECK( m_characterGravity.isOk<3>() );
#undef CHECK
	return true;
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
