/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_STATE_H
#define HKNP_CHARACTER_STATE_H

#include <Common/Base/hkBase.h>

class hknpCharacterContext;
struct hknpCharacterInput;
struct hknpCharacterOutput;

/// This class represents the behavior for a typical character state.
/// The state machine is built implicitly from a collection of states.
/// Most of the real work is done in the update call where the state performs actions
/// and transitions to other states through a given character context.
class hknpCharacterState : public hkReferencedObject
{
	public:

		/// The character state types.
		enum hknpCharacterStateType
		{
			// Default states
			HK_CHARACTER_ON_GROUND = 0,
			HK_CHARACTER_JUMPING,
			HK_CHARACTER_IN_AIR,
			HK_CHARACTER_CLIMBING,
			HK_CHARACTER_FLYING,

			// User states
			HK_CHARACTER_USER_STATE_0,
			HK_CHARACTER_USER_STATE_1,
			HK_CHARACTER_USER_STATE_2,
			HK_CHARACTER_USER_STATE_3,
			HK_CHARACTER_USER_STATE_4,
			HK_CHARACTER_USER_STATE_5,

			HK_CHARACTER_MAX_STATE_ID
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Return the state type.
		virtual hknpCharacterStateType getType() const = 0;

		/// Process the input - causes only state actions.
		virtual void update( hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output ) = 0;

		/// Process the input - causes only state transitions.
		virtual void change( hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output ) = 0;

		/// Do something before we transition to this state - default do nothing.
		virtual void enterState( hknpCharacterContext& context, hknpCharacterStateType prevState, const hknpCharacterInput& input, hknpCharacterOutput& output ) {}

		/// Do something before we leave this state - default do nothing.
		virtual void leaveState( hknpCharacterContext& context, hknpCharacterStateType nextState, const hknpCharacterInput& input, hknpCharacterOutput& output ) {}
};

#endif // HKNP_CHARACTER_STATE_H

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
