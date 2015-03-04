/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE void hknpDeactivatedIsland::setIslandId( hknpIslandId id )
{
	m_islandId = id;
	m_headConnectedIslandId = id;
}

HK_FORCE_INLINE void hknpDeactivationManager::setBodyDeactivation( hknpBodyId bodyId, bool enableDeactivation )
{
	const hknpBody& body = m_world->getSimulatedBody(bodyId);

	if (body.isDynamic())
	{
		hkUint8 &numChecks = m_deactivationStates[body.m_motionId.value()].m_numDeactivationChecks;
		if ( enableDeactivation )
		{
			// Only reset if previously disabled
			if (numChecks == 0xff)
			{
				numChecks = 0;
			}
		}
		else
		{
			// Setting counter to 0xff disables deactivation
			numChecks = 0xff;

			// If already deactivate, activate the body
			if (!body.isActive() )
			{
				// Record the index of the island to be activated at the beginning of the next step.
				markIslandForActivation( hknpIslandId(body.getDeactivatedIslandIndex()) );
			}
		}
	}
}

HK_FORCE_INLINE hknpDeactivationState* hknpDeactivationManager::getAllDeactivationStates()
{
	return m_deactivationStates.begin();
}

HK_FORCE_INLINE void hknpDeactivationManager::reserveNumMotions(int numMotions)
{
	if( numMotions > m_deactivationStates.getSize() )
	{
		m_deactivationStates.setSize(numMotions);
	}
}

HK_FORCE_INLINE const hkPointerMap<hkUint64, int>& hknpDeactivationManager::getBodyLinks() const
{
	return m_bodyLinks;
}

HK_FORCE_INLINE void hknpDeactivationManager::initDeactivationManager( hknpWorld* world )
{
	m_world = world;
}

HK_FORCE_INLINE void hknpDeactivationManager::markIslandForActivation( hknpIslandId islandId )
{
	HK_ASSERT2( 0xf49e152, islandId.isValid(), "markIslandForActivation() requires a valid island." );

	hknpIslandId islandIdIter = m_deactivatedIslands[islandId.value()]->m_headConnectedIslandId;
	do
	{
		hknpDeactivatedIsland* HK_RESTRICT islandLst = m_deactivatedIslands[islandIdIter.value()];
		if (!islandLst->m_isMarkedForActivation)
		{
			islandLst->m_isMarkedForActivation = true;
			m_islandsMarkedForActivation.pushBack( islandIdIter );
		}
		islandIdIter = islandLst->m_nextConnectedIslandId;
	}
	while (islandIdIter.isValid());

	HK_ASSERT2( 0xde542975, m_deactivatedIslands[islandId.value()]->m_isMarkedForActivation,
		"island is expected to be in the list of connected islands" );
}

HK_FORCE_INLINE void hknpDeactivationManager::resetDeactivationFrameCounter( hknpMotionId id )
{
	hknpDeactivationState& state = m_deactivationStates[id.value()];
	// If a motion has deactivation disabled, don't reset.
	if( state.m_numDeactivationChecks == 0xff )
		return;
	state.m_numDeactivationChecks = 0;
#ifdef DEACTIVATION_DEBUG_ACTIVATION_REASON
	state.m_activationReason = hknpDeactivationState::ACTIVATIONREASON_NOT_FULLY_TESTED;
#endif
}

HK_FORCE_INLINE hkUnionFind* hknpDeactivationStepInfo::getUnionFind()
{
	return m_unionFind;
}

HK_FORCE_INLINE void hknpDeactivationManager::sortIslandsMarkedForActivation()
{
	hkAlgorithm::quickSort( m_islandsMarkedForActivation.begin(), m_islandsMarkedForActivation.getSize() );
}

HK_FORCE_INLINE void hknpDeactivationManager::clearIslandsMarkedForDeactivation()
{
	m_islandsMarkedForActivation.clear();
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
