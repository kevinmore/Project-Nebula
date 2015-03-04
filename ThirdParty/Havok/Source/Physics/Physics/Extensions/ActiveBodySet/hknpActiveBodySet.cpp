/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Extensions/ActiveBodySet/hknpActiveBodySet.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>

hknpActiveBodySet::hknpActiveBodySet()
{
	m_world = HK_NULL;
}

hknpActiveBodySet::~hknpActiveBodySet()
{
	if (m_world)
	{
		removeFromWorld(m_world);
	}
}

HK_FORCE_INLINE void hknpActiveBodySet::addBodyToInactiveList( hknpWorld* world, hknpBodyId id )
{
	const hknpBody& body = world->getSimulatedBody(id);

	int islandId = body.getDeactivatedIslandIndex();
	hknpDeactivatedIsland* island = world->m_deactivationManager->m_deactivatedIslands[islandId];

	hknpDeactivatedIsland::ActivationInfo* ai = island->m_activationListeners.expandBy(1);

	ai->m_activationListener = this;
	ai->m_userData = (void*)id.value();

	m_inactiveBodies.pushBack( id );
}

HK_FORCE_INLINE bool hknpActiveBodySet::removeBodyFromInactiveList( hknpWorld* world, hknpBodyId id )
{
	const hknpBody& body = world->getBody(id);

	int idIndex = m_inactiveBodies.lastIndexOf(id);
	if ( idIndex >= 0)
	{
		m_inactiveBodies.removeAt( idIndex );
		int islandId = body.getDeactivatedIslandIndex();
		hknpDeactivatedIsland* island = world->m_deactivationManager->m_deactivatedIslands[islandId];

		hknpDeactivatedIsland::ActivationInfo ai;
		ai.m_activationListener = this;
		ai.m_userData = (void*)id.value();
		int aiIndex = island->findActivationInfo( ai );
		island->m_activationListeners.removeAt(aiIndex);

		return true;
	}

	return false;
}

HK_FORCE_INLINE void hknpActiveBodySet::removeBodyFromLists( hknpWorld* world, hknpBodyId id )
{
	// Try the inactive list first.
	if (!removeBodyFromInactiveList(world, id ))
	{
		// Then try active list.
		int activeIndex = m_activeBodies.lastIndexOf(id);
		if ( activeIndex >= 0)
		{
			m_activeBodies.removeAt( activeIndex );
		}
	}
}

void hknpActiveBodySet::addBody( hknpBodyId id )
{
	if ( m_world )
	{
		const hknpBody& body = m_world->getSimulatedBody(id);
		if ( body.isStatic())
		{
			return;
		}
		if ( body.isActive() )
		{
			m_activeBodies.pushBack( id );
		}
		else
		{
			addBodyToInactiveList( m_world, id );
		}
	}
	else
	{
		m_activeBodies.pushBack(id);
	}
}

void hknpActiveBodySet::removeBody( hknpBodyId id )
{
	// This code is written this way because it can happen that we get a VOLUME_LEFT event
	// (from cache manifold destruction) during the frame after a body removal/destruction.
	// When that happens, the body has unusable flags.
	const hknpBody& body = m_world->getBodyUnchecked( id );

	if ( m_world && body.isValid() )
	{
		// If the body is valid, we check the flags and try to do less work.

		if ( body.isStatic())
		{
			return;
		}
		if ( body.isActive() )
		{
			int index = m_activeBodies.lastIndexOf( id );
			if ( index >=0 )
			{
				m_activeBodies.removeAt( index );
			}
			else
			{
				removeBodyFromInactiveList( m_world, id );
			}
		}
	}
	else
	{
		int index = m_activeBodies.lastIndexOf( id );
		if ( index >=0 )
		{
			m_activeBodies.removeAt( index );
			return;
		}

		index = m_inactiveBodies.lastIndexOf( id );
		if ( index >=0 )
		{
			m_inactiveBodies.removeAt( index );
			return;
		}
	}
}

void hknpActiveBodySet::addToWorld( hknpWorld* world )
{
	m_activeBodies.insertAt( m_activeBodies.getSize(), m_inactiveBodies.begin(), m_inactiveBodies.getSize() );
	m_inactiveBodies.clear();

	HK_ASSERT2( 0xf0c13cf0, m_world == HK_NULL, "You cannot add an hknpActiveBodySet twice to the world" );
	m_world = world;
	sortBodiesIntoActiveAndInactiveSets();
}

void hknpActiveBodySet::removeFromWorld( hknpWorld* world )
{
	HK_ASSERT2( 0xf0c13cf0, m_world == world, "This set has not been added to this world" );

	m_activeBodies.reserve( m_activeBodies.getSize() + m_inactiveBodies.getSize() );
	for (int i = m_inactiveBodies.getSize()-1; i>=0; i-- )
	{
		hknpBodyId id = m_inactiveBodies[i];
		removeBodyFromInactiveList( m_world, id );
		m_activeBodies.pushBackUnchecked( id );
	}
	m_world = HK_NULL;
}

void hknpActiveBodySet::sortBodiesIntoActiveAndInactiveSets()
{
	if ( m_world )
	{
		int d = 0;
		for (int i = 0; i < m_activeBodies.getSize(); i++ )
		{
#if defined(HK_PLATFORM_PS3) | defined(HK_PLATFORM_XBOX360)
			if ( i+3 < m_activeBodies.getSize())
			{
				hkMath::prefetch128( &m_world->getBodyUnchecked( m_activeBodies[i+3]) );
			}
#endif
			hknpBodyId id = m_activeBodies[i];

			// We have to check if isAddedToWorld because if it was removed during
			// this frame, it will linger in our data structures until the next frame
			// (int which cache manifold destruction will trigger a volume-left event).
			const hknpBody& body = m_world->getBodyUnchecked( id );

			if (body.isAddedToWorld())
			{
				if ( body.isActive() )
				{
					m_activeBodies[d++] = id;
				}
				else
				{
					if ( body.isInactive())
					{
						addBodyToInactiveList( m_world, id );
					}
					// fixed bodies are simply ignored and therefor disappear
				}
			}
		}
		m_activeBodies.setSize(d);
	}
}

void hknpActiveBodySet::activateCallback( hknpDeactivatedIsland* island, void* userData )
{
	hknpBodyId id = hknpBodyId( (int) hkUlong(userData) );
	int index = m_inactiveBodies.lastIndexOf(id);
	m_inactiveBodies.removeAt(index);
	m_activeBodies.pushBack(id);
}

hknpTriggerVolumeFilteredBodySet::hknpTriggerVolumeFilteredBodySet( hknpBodyId triggerVolumeBodyId )
{
	m_triggerVolumeBodyId = triggerVolumeBodyId;
	m_activateEnteringBodies = true;
	m_activateLeavingBodies  = true;
}

void hknpTriggerVolumeFilteredBodySet::setTriggerVolumeBodyId( hknpBodyId triggerVolumeBodyId )
{
	HK_ASSERT2( 0xf034df23, m_world == HK_NULL,
		"You cannot change the trigger volume body id once this body set has been added to the world." );
	m_triggerVolumeBodyId = triggerVolumeBodyId;
}

hknpBodyId hknpTriggerVolumeFilteredBodySet::getTriggerVolumeBodyId()
{
	return m_triggerVolumeBodyId;
}

void hknpTriggerVolumeFilteredBodySet::addToWorld( hknpWorld* world )
{
	hknpActiveBodySet::addToWorld( world );
	m_world->getEventSignal( hknpEventType::TRIGGER_VOLUME, m_triggerVolumeBodyId ).subscribe(
		this, &hknpTriggerVolumeFilteredBodySet::onTriggerVolumeEvent, "hknpTriggerVolumeActiveBodySet" );
}

void hknpTriggerVolumeFilteredBodySet::removeFromWorld( hknpWorld* world )
{
	m_world->getEventSignal( hknpEventType::TRIGGER_VOLUME, m_triggerVolumeBodyId ).unsubscribeAll(this);
	hknpActiveBodySet::removeFromWorld( world );
}

void hknpTriggerVolumeFilteredBodySet::onTriggerVolumeEvent( const hknpEventHandlerInput& input, const hknpEvent& event )
{
	const hknpTriggerVolumeEvent& triggerVolumeEvent = event.asTriggerVolumeEvent();
	const hknpBodyId id			= input.getThisBody(triggerVolumeEvent);
	const hknpBodyId otherId	= input.getOtherBody(triggerVolumeEvent);
	if ( triggerVolumeEvent.m_status == hknpTriggerVolumeEvent::STATUS_ENTERED )
	{
		addBody( otherId );
		if ( m_activateLeavingBodies )
		{
			m_world->activateBody( otherId );
		}
	}
	else
	{
		removeBody( otherId );
		if ( m_activateEnteringBodies && m_world->getBodyUnchecked(otherId).isValid() ) // The body might have been destroyed last frame
		{
			m_world->activateBody( otherId );
		}
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
