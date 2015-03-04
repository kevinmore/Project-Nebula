/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterContext.h>

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterStateManager.h>


hknpCharacterContext::hknpCharacterContext(const hknpCharacterStateManager* Manager, hknpCharacterState::hknpCharacterStateType initialState)
	: m_characterType(HK_CHARACTER_PROXY)
	, m_stateManager(Manager)
	, m_currentState(initialState)
	, m_previousState(initialState)
	, m_filterEnable(true)
	, m_maxLinearAcceleration(625.0f)
	, m_maxLinearSpeed(20.0f)
	, m_gain(1.0f)
	, m_numUpdatesInCurrentState(0)
{
	m_previousSurfaceVelocity.setZero();
	m_stateManager->addReference();
}

hknpCharacterContext::~hknpCharacterContext()
{
	m_stateManager->removeReference();
}

hknpCharacterState::hknpCharacterStateType hknpCharacterContext::getState() const
{
	return m_currentState;
}

void hknpCharacterContext::setState(hknpCharacterState::hknpCharacterStateType state, const hknpCharacterInput& input, hknpCharacterOutput& output )
{
	hknpCharacterState::hknpCharacterStateType prevState = m_currentState;

	HK_ASSERT3(0x6d4abe44, m_stateManager->getState(state) != HK_NULL , "Bad state transition to " << state << ". Has this state been registered?");

	// Leave the old state
	m_stateManager->getState(m_currentState)->leaveState(*this, state, input, output);

	// Transition to the new state
	m_currentState = state;
	// Reset counter
	m_numUpdatesInCurrentState = 0;

	// Enter the new state
	m_stateManager->getState(m_currentState)->enterState(*this, prevState, input, output);

}

void hknpCharacterContext::update(const hknpCharacterInput& input, hknpCharacterOutput& output)
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
#if 0
	if (m_filterEnable)
	{
		// limit output velocity
		const hkReal currentVel = output.m_velocity.length<3>().getReal();
		if (currentVel > m_maxLinearVelocity)
		{
			hkSimdReal cv; cv.setFromFloat(m_maxLinearVelocity/currentVel);
			output.m_velocity.mul(cv);
		}

		// limit maximal linear acceleration and smooth velocity
		hkVector4 currentAcceleration; currentAcceleration.setSub(output.m_velocity, output.m_inputVelocity);
		hkSimdReal invDt; invDt.setFromFloat(input.m_stepInfo.m_invDeltaTime);
		currentAcceleration.mul(invDt);

		hkSimdReal ca; ca.setFromFloat(m_maxLinearAcceleration); ca = currentAcceleration.length<3>()/ca;
		hkSimdReal fac; fac.setFromFloat(m_gain*input.m_stepInfo.m_deltaTime);
		if(m_currentState != hknpCharacterState::HK_CHARACTER_JUMPING && ca.isGreater(hkSimdReal_1))
		{
			ca.setReciprocal(ca);
			fac.mul(ca);
		}
		output.m_velocity.setAddMul(output.m_inputVelocity, currentAcceleration, fac);
	}
#else
	if (m_filterEnable)
	{
		// limit output velocity
		const hkReal currentSpeed = output.m_velocity.length<3>().getReal();
		if (currentSpeed > m_maxLinearSpeed)
		{
			hkSimdReal cv; cv.setFromFloat( m_maxLinearSpeed/currentSpeed );

			output.m_velocity.mul(cv);
		}

		// limit maximal linear acceleration and smooth velocity
		hkVector4 currentAcceleration; currentAcceleration.setSub(output.m_velocity, output.m_inputVelocity);
		hkSimdReal invDt; invDt.setFromFloat(input.m_stepInfo.m_invDeltaTime);
		currentAcceleration.mul(invDt);

		hkReal ca = currentAcceleration.length<3>().getReal()/m_maxLinearAcceleration;

		output.m_velocity = output.m_inputVelocity;

		if ( (ca > 1.0f) && !(m_currentState == hknpCharacterState::HK_CHARACTER_JUMPING))
		{
			hkSimdReal tmp; tmp.setFromFloat(m_gain*input.m_stepInfo.m_deltaTime/ca);
			output.m_velocity.addMul(tmp,currentAcceleration);
		}
		else
		{
			hkSimdReal tmp; tmp.setFromFloat(m_gain*input.m_stepInfo.m_deltaTime);
			output.m_velocity.addMul(tmp,currentAcceleration);
		}
	}
#endif

	++m_numUpdatesInCurrentState;
}

void hknpCharacterContext::setFilterEnable(const hkBool status)
{
	m_filterEnable = status;
}

void hknpCharacterContext::setFilterParameters(const hkReal gain, const hkReal maxSpeed, const hkReal maxAcceleration)
{
	m_gain = gain;

	m_maxLinearSpeed = maxSpeed;

	m_maxLinearAcceleration = maxAcceleration;
}

void hknpCharacterContext::setCharacterType(const CharacterType newType)
{
	m_characterType = newType;
}

int hknpCharacterContext::getNumUpdatesInCurrentState() const
{
	return m_numUpdatesInCurrentState;
}

const hkVector4& hknpCharacterContext::getPreviousSurfaceVelocity() const
{
	return m_previousSurfaceVelocity;
}

hkBool hknpCharacterInput::isValid() const
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
