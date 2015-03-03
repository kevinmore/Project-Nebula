/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_STATE_FLYING_H
#define HKNP_CHARACTER_STATE_FLYING_H

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterState.h>

/// A state to represent flying.
/// This state is currently unused - no other states transition to it.
class hknpCharacterStateFlying : public hknpCharacterState
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpCharacterStateFlying() {}

		/// Return the state type.
		virtual hknpCharacterStateType getType() const;

		/// Process the user input - causes state actions.
		virtual void update(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output);

		///	Process the user input - causes state transitions.
		virtual void change(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output);
};

#endif // HKNP_CHARACTER_STATE_FLYING_H

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
