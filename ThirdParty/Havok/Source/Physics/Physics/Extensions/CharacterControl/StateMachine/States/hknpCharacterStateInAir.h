/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CHARACTER_STATE_IN_AIR_H
#define HKNP_CHARACTER_STATE_IN_AIR_H

#include <Physics/Physics/Extensions/CharacterControl/StateMachine/hknpCharacterState.h>

class hknpCharacterStateInAir : public hknpCharacterState
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpCharacterStateInAir();

		/// Return the state type.
		virtual hknpCharacterStateType getType() const;

		/// Process the user input - causes state actions.
		virtual void update(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output);

		///	Process the user input - causes state transitions.
		virtual void change(hknpCharacterContext& context, const hknpCharacterInput& input, hknpCharacterOutput& output);

		/// gets the gain for this feedback controller.
		hkReal getGain() const;

		/// Sets the gain for this feedback controller.
		void setGain(hkReal newGain);

		/// gets the air speed.
		hkReal getSpeed() const;

		/// Sets the air speed.
		void setSpeed(hkReal newGain);

		/// Gets the maximal linear acceleration for this state.
		hkReal getMaxLinearAcceleration() const;

		/// Set the maximal linear acceleration for this state.
		void setMaxLinearAcceleration(hkReal newMaxAcceleration);

	protected:

		hkReal	m_gain;
		hkReal	m_airSpeed;
		hkReal  m_maxLinearAcceleration;
};

#endif // HKNP_CHARACTER_STATE_IN_AIR_H

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
