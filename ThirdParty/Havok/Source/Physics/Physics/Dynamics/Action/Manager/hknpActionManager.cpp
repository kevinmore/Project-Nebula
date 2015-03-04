/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Action/hknpAction.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>

#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Physics/Physics/Dynamics/Solver/hknpSolverData.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulation.h>

#include <Physics/Physics/Dynamics/Action/Manager/hknpActionManager.h>


void hknpActionManager::executeActions( const hknpSimulationThreadContext& tl, const hknpSolverInfo& solverInfo, hknpCdPairWriter* HK_RESTRICT pairWriter, hknpDeactivatedIsland* deactivatedIsland )
{
	int numActions = m_activeActions.getSize();
	int d = 0;
	for( int i = 0; i < numActions; i++ )
	{
		hknpAction* action = m_activeActions[i];
		hknpAction::ApplyActionResult result = action->applyAction( tl, solverInfo, pairWriter );
		if( result == hknpAction::RESULT_OK )
		{
			m_activeActions[d] = action;
			d++;
			continue;
		}
		if( result == hknpAction::RESULT_DEACTIVATE )
		{
			hknpDeactivatedIsland::ActivationInfo* HK_RESTRICT ai = &deactivatedIsland->m_activationListeners.expandOne();
			ai->m_activationListener = this;
			ai->m_userData = action;
		}
		else
		{
			// remove is silent, nothing to do
		}
	}
	m_activeActions.setSize(d);
}

void hknpActionManager::addAction( hknpWorld* world, hknpAction* action, hknpActivationMode::Enum activationMode )
{
	activateCallback( HK_NULL, action );	// add the action to the list

	bool forceActivation = ( activationMode == hknpActivationMode::ACTIVATE );
	hkInplaceArray<hknpBodyId,16> bodyIds;
	action->getBodies( &bodyIds );

	if( !forceActivation )
	{
		hknpIslandId islandId = hknpIslandId::invalid();

		// if we don't enforce activation, we need to check if at least one body is active,
		// if yes we have to enforce activation
		for( int i = 0; i < bodyIds.getSize(); i++ )
		{
			hknpBodyId id = bodyIds[i];
			const hknpBody& body = world->getBody( id );
			if( body.isDynamic() )
			{
				forceActivation = true;
				break;
			}
			if( body.isInactive() )
			{
				if( islandId.isValid() )
				{
					if( islandId.value() != body.getDeactivatedIslandIndex() )
					{
						// Now the action connects 2 separate deactivated islands, force activation
						forceActivation = true;
						break;
					}
				}
				else
				{
					islandId = hknpIslandId( body.getDeactivatedIslandIndex() );
				}
			}
		}
	}

	if( forceActivation )
	{
		for( int i = 0; i < bodyIds.getSize(); i++ )
		{
			hknpBodyId id = bodyIds[i];
			const hknpBody& body = world->getBody( id );
			if( body.isInactive() && body.isAddedToWorld() )
			{
				world->m_deactivationManager->markBodyForActivation( id );
			}
		}
	}
}

hknpIslandId HK_CALL hknpActionManager::findDeactivatedIsland( hknpWorld* world, const hknpAction* action )
{
	hkInplaceArray<hknpBodyId,16> bodyIds;
	action->getBodies( &bodyIds );

	for( int i = 0; i < bodyIds.getSize(); i++ )
	{
		hknpBodyId id = bodyIds[i];
		const hknpBody& body = world->getBody( id );
		if( body.isInactive() && body.isAddedToWorld() )
		{
			hknpIslandId islandId = hknpIslandId( body.getDeactivatedIslandIndex() );
			return islandId;
		}
	}

	return hknpIslandId::invalid();
}


void hknpActionManager::removeAction( hknpWorld* world, hknpAction* action, hknpActivationMode::Enum activationMode )
{
	// Look for it in the active actions
	{
		const int index = m_activeActions.indexOf( action );
		if( index >= 0 )
		{
			m_activeActions.removeAtAndCopy( index );
			return;
		}
	}

	// Otherwise look for it in the inactive islands
	{
		hknpIslandId islandId = findDeactivatedIsland( world, action );
		if( islandId.isValid() )
		{
			hknpDeactivatedIsland* island = world->m_deactivationManager->m_deactivatedIslands[ islandId.value() ];
			hknpDeactivatedIsland::ActivationInfo ai; ai.m_activationListener = this; ai.m_userData = action;
			int aiIndex = island->findActivationInfo( ai );
			if( aiIndex < 0 )
			{
				HK_ASSERT2( 0xf03ddf5f, false, "Cannot find action" );
				return;
			}
			island->m_activationListeners.removeAt( aiIndex );
			if( activationMode == hknpActivationMode::ACTIVATE )
			{
				world->m_deactivationManager->markIslandForActivation( islandId );
			}
		}
		else
		{
			HK_ASSERT2( 0xf03ddf5f, false, "Cannot find action" );
		}
	}
}

bool hknpActionManager::isActionAdded( hknpAction* action ) const
{
	if( m_activeActions.indexOf(action) != -1 )
	{
		return true;
	}

	return false;
}

void hknpActionManager::activateCallback( hknpDeactivatedIsland* island, void* userData )
{
	hknpAction* action = (hknpAction*)userData;
	HK_ASSERT2(0x19bbac34, !isActionAdded(action), "Action already added!");
	m_activeActions.pushBack( action );
}

void hknpActionManager::onWorldShifted( hknpWorld* world, hkVector4Parameter offset )
{
	const int numActiveActions = m_activeActions.getSize();
	for( int i = 0; i < numActiveActions; i++ )
	{
		m_activeActions[i]->onShiftWorld( offset );
	}
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
