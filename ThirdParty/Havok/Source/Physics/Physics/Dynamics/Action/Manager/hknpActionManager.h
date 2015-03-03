/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_ACTION_MANAGER_H
#define HK_ACTION_MANAGER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/Action/hknpAction.h>
#include <Physics/Physics/Dynamics/World/Deactivation/hknpActivationListener.h>

class hknpAction;
class hknpCdPairWriter;


/// Class that manages actions.
/// It keeps a list of all active actions.
/// If an action requests to be deactivated, it will be removed from the active list and inserted into the hknpDeactivatedIsland::m_activationListeners.
class hknpActionManager : public hknpActivationListener
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpActionManager );

		/// Destructor.
		virtual ~hknpActionManager() {}

		/// Internal method: Adds an action, you should use the hknpWorld::addAction().
		void addAction( hknpWorld* world, hknpAction* action, hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );

		/// Internal method: Removes an action, you should use the hknpWorld::removeAction().
		void removeAction( hknpWorld* world, hknpAction* action, hknpActivationMode::Enum activationMode = hknpActivationMode::ACTIVATE );

		/// Check if an action is present in the manager.
		bool isActionAdded( hknpAction* action ) const;

		//
		//	Internal methods
		//

		/// Execute all actions.
		void executeActions( const hknpSimulationThreadContext& tl, const hknpSolverInfo& solverInfo, hknpCdPairWriter* HK_RESTRICT pairWriter, hknpDeactivatedIsland* deactivatedIsland );

		/// Find the deactivated island of an action.
		static hknpIslandId HK_CALL findDeactivatedIsland( hknpWorld* world, const hknpAction* action );

		// Signal handler
		void onWorldShifted( hknpWorld* world, hkVector4Parameter offset );

	protected:

		/// Called by the hknpDeactivatedIsland when it gets activated
		virtual void activateCallback( hknpDeactivatedIsland* island, void* userData );

	public:

		/// All active actions.
		hkArray<hknpAction*> m_activeActions;
};


#endif // HK_ACTION_MANAGER_H

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
