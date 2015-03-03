/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_ACTIVE_BODY_SET_H
#define HKNP_ACTIVE_BODY_SET_H

#include <Physics/Physics/Dynamics/World/Deactivation/hknpActivationListener.h>


/// This is a set of bodies which are very efficiently sorted into active and inactive bodies.
/// (Note: There is a very small CPU cost for active bodies and zero CPU cost for inactive bodies).
///
/// This class can be used to elegantly implement force fields which ignore deactivated bodies efficiently.
/// To understand deactivation, please read the reference manual on the hknpDeactivationManager.
///
/// How to use it:
/// Create an instance of this class and call addToWorld() to activate activation tracking.
/// Then simply add all bodies to this class. Before you access the m_activeBodies array, you need to call updateSet()
/// to make sure the m_activeBodies array is up to date.
///
/// Implementation details: If the engine deactivates a body, it will just flag the body and will not
/// inform this hknpActiveBodySet, so the implementation of updateSet which check the current list of active bodies.
/// If a deactivated body is detected, it will insert a callback in the hknpDeactivatedIsland. So should an
/// island get activated, this class will receive a callback (through through the hknpActivationListener interface)
/// and update the m_activeBodies and m_inactiveBodies arrays.
class hknpActiveBodySet : public hkReferencedObject, protected hknpActivationListener
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor
		hknpActiveBodySet();

		/// Destructor
		~hknpActiveBodySet();

		/// Add a body.
		/// This body will be present either in the m_activeBodies or m_inactiveBodies array. It is OK to add already
		/// inactive bodies (in this case the body will be added to the active list as soon as the body becomes active).
		/// If the hknpActiveBodySet is not added to the world yet, this is will simply be added to the activeBodies array.
		void addBody( hknpBodyId id );

		/// Remove a body.
		/// Removing an active body is much cheaper than removing an inactive body.
		void removeBody( hknpBodyId id );

		/// Add this body set to the world. This is needed so that the hknpActiveBodySet can track deactivation.
		/// Should there already be any bodies in the active of inactive lists, those lists will be rechecked
		/// and the body IDs sorted into the correct list.
		virtual void addToWorld( hknpWorld* world );

		/// Remove this body set from the world. This will put all bodies in the m_activeBodies array.
		virtual void removeFromWorld( hknpWorld* world );

		/// This updates the m_activeBodies and m_inactiveBodies.
		/// Once this function is called it is safe to modify the motions of the active bodies directly instead of using
		/// the hknpWorld functions to improve performance. However this also means that you bypass the visual debugger
		/// and extra checking.
		virtual void updateSet() { sortBodiesIntoActiveAndInactiveSets(); }

	protected:

		// Implementation of updateSet().
		void sortBodiesIntoActiveAndInactiveSets();

		HK_FORCE_INLINE void addBodyToInactiveList( hknpWorld* world, hknpBodyId id );
		HK_FORCE_INLINE bool removeBodyFromInactiveList( hknpWorld* world, hknpBodyId id );

		HK_FORCE_INLINE void removeBodyFromLists( hknpWorld* world, hknpBodyId id );

		// hknpActivationListener interface implementation
		virtual void activateCallback( hknpDeactivatedIsland* island, void* userData );

	public:

		/// A list of active bodies.
		hkArray<hknpBodyId> m_activeBodies;

		/// A list of inactive bodies.
		hkArray<hknpBodyId> m_inactiveBodies;

		/// A pointer to the world if this set is added to the world.
		hknpWorld* m_world;
};


/// An active body set holding all active bodies intersecting a trigger volume.
class hknpTriggerVolumeFilteredBodySet : public hknpActiveBodySet
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		hknpTriggerVolumeFilteredBodySet( hknpBodyId triggerVolumeBodyId = hknpBodyId::invalid() );

		/// Set the trigger volume body id. This only works if the material of the body has the m_triggerVolumeAccuracy
		/// parameter set to != 0.0f. See hknpMaterial for details.
		void setTriggerVolumeBodyId( hknpBodyId id );

		/// Get the trigger volume body id.
		hknpBodyId getTriggerVolumeBodyId();

		// hknpActiveBodySet implementation.
		virtual void addToWorld( hknpWorld* world );

		// hknpActiveBodySet implementation.
		virtual void removeFromWorld( hknpWorld* world );

	protected:

		// Trigger event handler
		void onTriggerVolumeEvent( const hknpEventHandlerInput& input, const hknpEvent& event );

	public:

		/// If set, bodies entering the trigger volume will be activated.
		/// Setting this flag only makes sense if the trigger volume is static and is moved by setting its position.
		hkBool m_activateEnteringBodies;

		/// If set, bodies leaving the trigger volume will be activated.
		/// Setting this flag only makes sense if the trigger volume is static and is moved by setting its position.
		hkBool m_activateLeavingBodies;

		/// The body id of the trigger volume.
		hknpBodyId m_triggerVolumeBodyId;
};


#endif // HKNP_ACTIVE_BODY_SET_H

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
