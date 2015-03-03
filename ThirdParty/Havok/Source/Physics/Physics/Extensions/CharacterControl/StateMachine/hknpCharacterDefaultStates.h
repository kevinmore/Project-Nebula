/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_DEFAULT_STATES_H
#define HKNP_CHARACTER_DEFAULT_STATES_H

// State Machine
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterState.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterStateManager.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterContext.h>

// Default States
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateOnGround.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateClimbing.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateFlying.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateInAir.h>
#include <Physics/Physics/Extensions/CharacterControl/StateMachine/States/hknpCharacterStateJumping.h>

#endif // HKNP_CHARACTER_DEFAULT_STATES_H

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
