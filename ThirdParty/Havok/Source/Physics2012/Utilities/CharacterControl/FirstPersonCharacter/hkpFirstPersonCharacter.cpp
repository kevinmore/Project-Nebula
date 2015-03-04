/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Visualize/hkDebugDisplay.h>

#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpAllRayHitCollector.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Utilities/CharacterControl/CharacterRigidBody/hkpCharacterRigidBody.h>
#include <Physics2012/Utilities/CharacterControl/StateMachine/hkpDefaultCharacterStates.h>
#include <Physics2012/Utilities/Weapons/hkpFirstPersonGun.h>

// this
#include <Physics2012/Utilities/CharacterControl/FirstPersonCharacter/hkpFirstPersonCharacter.h>


void hkpFirstPersonCharacter::createDefaultCharacterController( const hkVector4& position, const hkVector4& up, hkReal capsuleHeight, hkReal capsuleRadius )
{
	//  Create character
	hkSimdReal h = hkSimdReal::fromFloat( capsuleHeight * 0.5f - capsuleRadius );
	hkVector4 vertexA; vertexA.setMul( h, up );
	hkVector4 vertexB; vertexB.setMul( -h, up);
	hkpShape* characterShape = new hkpCapsuleShape(vertexA, vertexB, capsuleRadius);

	// Construct a character rigid body
	hkpCharacterRigidBodyCinfo info;
	info.m_shape = characterShape;
	info.m_position = position;
	info.m_up = up;
	
	m_characterRb = new hkpCharacterRigidBody( info );
	characterShape->removeReference();
}

void hkpFirstPersonCharacter::createDefaultCharacterContext()
{
	hkpCharacterState* state;
	hkpCharacterStateManager* manager = new hkpCharacterStateManager();

	state = new hkpCharacterStateOnGround();
	manager->registerState( state,	HK_CHARACTER_ON_GROUND);
	state->removeReference();

	state = new hkpCharacterStateInAir();
	manager->registerState( state,	HK_CHARACTER_IN_AIR);
	state->removeReference();

	state = new hkpCharacterStateJumping();
	manager->registerState( state,	HK_CHARACTER_JUMPING);
	state->removeReference();

	state = new hkpCharacterStateClimbing();
	manager->registerState( state,	HK_CHARACTER_CLIMBING);
	state->removeReference();

	m_characterRbContext = new hkpCharacterContext(manager, HK_CHARACTER_ON_GROUND);
	manager->removeReference();
}


void hkpFirstPersonCharacter::setCharacterInputFromCharacterRb( hkpCharacterInput& input )
{
	//
	// Copy some values from the character controller
	//
	input.m_velocity = m_characterRb->getLinearVelocity();
	input.m_position = m_characterRb->getPosition();
	input.m_up = m_characterRb->m_up;
	if (m_specialGravity)
	{
		input.m_characterGravity = m_gravity;
	}
	else
	{
		hkSimdReal g; g.setFromFloat(m_gravityStrength);
		input.m_characterGravity.setMul(g, m_world->m_gravity);
	}

	//
	// Call get support and set the relevant values in the character input
	//

	hkStepInfo stepInfo;
	stepInfo.m_deltaTime = 1.f / 60.f;
	stepInfo.m_invDeltaTime = 60.f;
	m_characterRb->checkSupport( stepInfo, input.m_surfaceInfo );
}


void hkpFirstPersonCharacter::update( hkReal timestep, const CharacterControls& controls, bool showForwardDirection )
{
	hkpCharacterInput input;
	hkpCharacterOutput output;

	input.m_atLadder = false;
	input.m_stepInfo.m_deltaTime = timestep;
	input.m_stepInfo.m_invDeltaTime = 1.0f / timestep;

	input.m_forward = controls.m_forward;
	input.m_inputLR = controls.m_straffeLeftRight;
	input.m_inputUD = controls.m_forwardBack;
	input.m_wantJump = controls.m_wantJump;

	setCharacterInputFromCharacterRb(input);

	m_characterRbContext->update(input, output);		

	m_characterRb->setLinearVelocity(output.m_velocity, timestep);

	if (showForwardDirection)
	{
		hkVector4 start; start.setAdd( m_characterRb->getPosition(), m_characterRb->m_up );
		hkVector4 end; end.setAdd( start, controls.m_forward );
		HK_DISPLAY_LINE( start, end, hkColor::BLUE );
	}
}



hkpFirstPersonCharacterCinfo::hkpFirstPersonCharacterCinfo()
{
	m_position.set(0, 3, 0);
	m_direction.set(1,0,0);
	m_up.set(0,1,0);
	m_gravityStrength = 16;
	m_capsuleHeight = 2.0f;
	m_capsuleRadius = 0.6f;
	m_eyeHeight	= 0.7f;

	m_flags = hkpFirstPersonCharacter::CAN_DETACH_FROM_CHAR;

	m_verticalSensitivity = .1f;
	m_horizontalSensitivity = .1f;
	m_sensivityPadX = 0.2f;
	m_sensivityPadY = 0.2f;
	m_maxUpDownAngle = HK_REAL_PI / 6.f;
	m_forwardBackwardSpeedModifier = 1.0f;
	m_leftRightSpeedModifier = 1.0f;

	m_numFramesPerShot = 5;
	m_characterRb = HK_NULL;
	m_context = HK_NULL;
	m_world = HK_NULL;
}


hkpFirstPersonCharacter::hkpFirstPersonCharacter( const hkpFirstPersonCharacterCinfo& info )
{
	HK_ASSERT(0x123496, info.m_world);
	m_world = info.m_world;

	if ( info.m_characterRb == HK_NULL )
	{
		createDefaultCharacterController( info.m_position, info.m_up, info.m_capsuleHeight, info.m_capsuleRadius );
	}
	else
	{
		m_characterRb = info.m_characterRb;
		m_characterRb->addReference();
	}

	if (info.m_context == HK_NULL)
	{
		createDefaultCharacterContext();
	}
	else
	{
		m_characterRbContext = info.m_context;
		m_characterRbContext->addReference();
	}

	HK_ASSERT(0x32436503, info.m_up.isNormalized<3>() );
	m_currentAngle = HK_REAL_PI * 0.5f; 
	m_currentElevation = 0.0f;
	hkVector4 direction = info.m_direction;
	hkSimdReal dirLen = direction.normalizeWithLength<3>();
	if ( dirLen.isNotEqualZero() )
	{
		hkSimdReal dot; dot.setClamped(info.m_up.dot<3>(direction), hkSimdReal_Minus1, hkSimdReal_1 );

		direction.addMul( -dot, info.m_up);
		direction.normalize<3>();
		const hkVector4 forward = hkVector4::getConstant<HK_QUADREAL_1000>();
		hkSimdReal d2 = forward.dot<3>(direction);
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		hkVector4 acosDots; acosDots.set(d2,dot,hkSimdReal_0,hkSimdReal_0);
#if defined(HK_REAL_IS_DOUBLE)
		acosDots.m_quad.xy = hkMath::twoAcos(acosDots.m_quad.xy);
#else
		acosDots.m_quad = hkMath::quadAcos(acosDots.m_quad);
#endif
		const hkSimdReal maxUD = hkSimdReal::fromFloat(info.m_maxUpDownAngle);
		hkSimdReal currentEl;
		currentEl.setClamped(hkSimdReal_PiOver2 - acosDots.getComponent<1>(), -maxUD, maxUD);
		currentEl.store<1>(&m_currentElevation);
		acosDots.store<1>(&m_currentAngle);
#else
		m_currentElevation = HK_REAL_PI * 0.5f - hkMath::acos(dot.getReal());
		m_currentElevation = hkMath::clamp(m_currentElevation, -info.m_maxUpDownAngle, info.m_maxUpDownAngle);
		m_currentAngle = hkMath::acos(d2.getReal());
#endif
	}
	else
	{
		m_currentAngle = 0.0f;
	}

	m_gravityStrength = info.m_gravityStrength;
	hkSimdReal g; g.setFromFloat(info.m_gravityStrength);
	m_gravity.setMul(-g, info.m_up);
	m_specialGravity = false;
	m_gunCounter = 0;
	m_gunCounterRmb = 0;

	m_flags = info.m_flags | HAS_USER_CONTROL | MAKE_OCCLUDING_OBJECTS_TRANSPARENT;
	// make occluding transparent only used when view is behind character (currently code is disabled)

	m_verticalSensitivity = info.m_verticalSensitivity;
	m_horizontalSensitivity = info.m_horizontalSensitivity;
	m_sensivityPadX = info.m_sensivityPadX;
	m_sensivityPadY = info.m_sensivityPadY;
	m_eyeHeight = info.m_eyeHeight;

	m_maxUpDownAngle = info.m_maxUpDownAngle;
	m_numFramesPerShot = info.m_numFramesPerShot;
	m_forwardBackwardSpeedModifier = info.m_forwardBackwardSpeedModifier;
	m_leftRightSpeedModifier = info.m_leftRightSpeedModifier;

	m_currentGun = HK_NULL;
}

hkpFirstPersonCharacter::~hkpFirstPersonCharacter()
{
	m_characterRb->removeReference();
	m_characterRbContext->removeReference();

	if (m_currentGun)
	{
		m_currentGun->quitGun( m_world );
		m_currentGun->removeReference();
	}
}

void hkpFirstPersonCharacter::getForwardDir( hkVector4& forward ) const
{
	forward = hkVector4::getConstant<HK_QUADREAL_1000>();
	hkQuaternion currentOrient;
	currentOrient.setAxisAngle(m_characterRb->m_up, m_currentAngle);
	forward.setRotatedDir( currentOrient, forward );

	hkVector4 rotateAxis;
	rotateAxis.setCross(forward, m_characterRb->m_up);
	hkQuaternion upOrient;
	upOrient.setAxisAngle( rotateAxis, m_currentElevation );
	forward.setRotatedDir( upOrient, forward );
}

void hkpFirstPersonCharacter::setForwardDir( hkVector4Parameter forward )
{
	hkSimdReal sinUp = forward.dot<3>( m_characterRb->m_up );
	hkVector4 remaining = forward; 
	remaining.subMul( sinUp, m_characterRb->m_up );
	hkSimdReal cosUp = remaining.length<3>();
	hkVector4Util::linearAtan2Approximation( sinUp, cosUp ).store<1>(&m_currentElevation);

	const hkVector4 X = hkVector4::getConstant<HK_QUADREAL_1000>();
	hkVector4 R; R.setCross( m_characterRb->m_up, X );
	hkSimdReal sinX = X.dot<3>(forward);
	hkSimdReal cosX = R.dot<3>(forward);

	hkVector4Util::linearAtan2Approximation( cosX, sinX ).store<1>(&m_currentAngle);
}


hkpFirstPersonGun* hkpFirstPersonCharacter::setGun( hkpFirstPersonGun* gun )
{
	if (gun)
	{
		gun->addReference();
	}
	if (m_currentGun)
	{
		m_currentGun->quitGun( m_world );
		m_currentGun->removeReference();
	}
	m_currentGun = gun;
	if (gun)
	{
		gun->initGun( m_world );
	}
	return gun;
}

void hkpFirstPersonCharacter::getViewTransform(hkTransform& viewTransformOut) const
{
	hkSimdReal e; e.setFromFloat(m_eyeHeight);
	viewTransformOut.getTranslation().setAddMul( m_characterRb->getPosition(), m_characterRb->m_up, e );

	// get rotation
	hkVector4 x; getForwardDir(x); x.normalize<3>();
	hkVector4 y; y = m_characterRb->m_up; //y.normalize3();
	hkVector4 z; z.setCross(x, y); z.normalize<3>();
	             y.setCross(z, x); y.normalize<3>();
	viewTransformOut.getRotation().setCols(x, y, z);
}

hkpRigidBody* hkpFirstPersonCharacter::getRigidBody() const
{ 
	return m_characterRb->m_character; 
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
