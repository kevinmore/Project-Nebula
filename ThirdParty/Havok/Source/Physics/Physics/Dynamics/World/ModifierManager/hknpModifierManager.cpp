/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>
#include <Physics/Physics/Collide/Filter/AlwaysHit/hknpAlwaysHitCollisionFilter.h>


HK_COMPILE_TIME_ASSERT( hknpModifierManager::NUM_BODY_FLAGS == sizeof(hknpBodyFlags) * 8 );

hknpModifierManager::ModifierEntries::ModifierEntries()
:	m_allEnablingFlags(0)
,	m_numModifiers(0)
{
	for(int i=0; i< MAX_MODIFIERS_PER_FUNCTION; i++)
	{
		m_entries[i].m_enablingFlags = 0;
	}
}

hknpModifierManager::hknpModifierManager()
{
	m_collisionFilter = &hknpAlwaysHitCollisionFilter::g_instance;
	m_collisionQueryFilter   = &hknpAlwaysHitCollisionFilter::g_instance;
	m_neighborWeldingModifier		= HK_NULL;
	m_motionWeldingModifier		= HK_NULL;
	m_triangleWeldingModifier	= HK_NULL;

	for (int i=0; i<hknpConstraintSolverType::NUM_TYPES; ++i)
	{
		m_constraintSolvers[i] = HK_NULL;
	}

	for (int i=0; i<hknpCollisionCacheType::NUM_TYPES; ++i)
	{
		m_collisionDetectors[i] = HK_NULL;
	}

	m_globalBodyFlags = 0;

	hkString::memSet( m_globalBodyFlagCounts, 0, sizeof(m_globalBodyFlagCounts) );
}


#if !defined(HK_PLATFORM_SPU)

void hknpModifierManager::addModifier( hknpModifierFlags enablingFlags, hknpModifier* modifier, Priority priority )
{
	hkUint32 enabledFunctions = modifier->getEnabledFunctions();
	addModifier( enablingFlags, modifier, enabledFunctions, priority );
}

#endif


void hknpModifierManager::addModifier( hknpModifierFlags enablingFlags, hknpModifier* modifier, hkUint32 enabledFunctions, Priority priority )
{
	for( int i=0; enabledFunctions; enabledFunctions=enabledFunctions>>1, i++ )
	{
		if ( 0 == (1 & enabledFunctions) )
		{
			continue;
		}

		ModifierEntries& iEntry = m_modifiersPerFunction[ i ];

		if ( iEntry.m_numModifiers == MAX_MODIFIERS_PER_FUNCTION )
		{
			HK_ASSERT( 0xf06edfe1, iEntry.m_numModifiers < MAX_MODIFIERS_PER_FUNCTION );
			HK_BREAKPOINT(0xaf31ea31);
		}
		int index = iEntry.m_numModifiers++;
		if ( priority ==  PRIORITY_HIGHER )
		{
			while(index>0)
			{
				iEntry.m_entries[index] = iEntry.m_entries[index-1];
				index--;
			}
		}

		iEntry.m_allEnablingFlags = iEntry.m_allEnablingFlags | enablingFlags;
		iEntry.m_entries[index].m_enablingFlags = enablingFlags;
		iEntry.m_entries[index].m_modifier = modifier;
	}
}


#if !defined(HK_PLATFORM_SPU)

void hknpModifierManager::removeModifier( hknpModifier* modifier )
{
	int enabledFunctions = modifier->getEnabledFunctions();
	for( int i=0; enabledFunctions; enabledFunctions=enabledFunctions>>1, i++ )
	{
		if( 0 == (1 & enabledFunctions) )
		{
			continue;
		}

		ModifierEntries& entries = m_modifiersPerFunction[i];

		hknpModifierFlags allEnablingFlags = 0;
		int k = 0;
		int j = 0;
		for( ; j<entries.m_numModifiers; ++j )
		{
			if( entries.m_entries[j].m_modifier != modifier )
			{
				// Keep this entry
				allEnablingFlags |= entries.m_entries[j].m_enablingFlags;
				entries.m_entries[k] = entries.m_entries[j];
				k++;
			}
			else
			{
				// Skip it
				entries.m_numModifiers--;
			}
		}
		HK_ASSERT2( 0xf034de54, k != j, "Cannot find modifier" );

		entries.m_allEnablingFlags = allEnablingFlags;
	}
}

#endif


void hknpModifierManager::setSolver( hknpConstraintSolverType::Enum solverType, hknpConstraintSolver* solver )
{
	if ( solverType >= hknpConstraintSolverType::NUM_TYPES)
	{
		HK_BREAKPOINT(0xee5f13a5);
	}
	HK_ASSERT2(0xee5f13a6, m_constraintSolvers[solverType] == HK_NULL, "Another solver has already been registered using the same ID.");

	m_constraintSolvers[solverType] = solver;
}

void hknpModifierManager::setCollisionDetector( hknpCollisionCacheType::Enum collisionType, hknpCollisionDetector* detector )
{
	if ( collisionType >= hknpCollisionCacheType::NUM_TYPES )
	{
		HK_BREAKPOINT(0xee5f13a5);
	}
	HK_ASSERT2(0xee5f13a6, m_collisionDetectors[collisionType] == HK_NULL, "Another collision detector has already been registered using the same ID.");

	m_collisionDetectors[collisionType] = detector;
}


void hknpModifierManager::setCollisionFilter(hknpCollisionFilter* collisionFilter)
{
#if !defined(HK_PLATFORM_SPU)
	if ( collisionFilter == HK_NULL )
	{
		HK_WARN_ALWAYS( 0xaf13e132, "The default collision filter must not be NULL. Using 'AlwaysHit' collision filter instead." );
		m_collisionFilter = &hknpAlwaysHitCollisionFilter::g_instance;
	}
#endif
	m_collisionFilter = collisionFilter;
}


void hknpModifierManager::setCollisionQueryFilter(hknpCollisionFilter* collisionQueryFilter)
{
#if !defined(HK_PLATFORM_SPU)
	if ( collisionQueryFilter == HK_NULL )
	{
		HK_WARN_ALWAYS( 0xaf13e133, "The default collision query filter must not be NULL. Using 'AlwaysHit' filter instead." );
		collisionQueryFilter = &hknpAlwaysHitCollisionFilter::g_instance;
	}
#endif
	m_collisionQueryFilter = collisionQueryFilter;
}


void hknpModifierManager::incrementGlobalBodyFlags( hknpBodyFlags flags )
{
	for( int i=0; flags; flags=flags>>1, i++ )
	{
		if ( 1 & flags )
		{
			m_globalBodyFlagCounts[i]++;
			m_globalBodyFlags |= ( 1 << i );
		}
	}

	HK_ON_DEBUG( checkConsistency(); )
}

void hknpModifierManager::decrementGlobalBodyFlags( hknpBodyFlags flags )
{
	for( int i=0; flags; flags=flags>>1, i++ )
	{
		if ( 1 & flags )
		{
			if( --m_globalBodyFlagCounts[i] <= 0 )
			{
				m_globalBodyFlags &= ~( 1 << i );
			}
		}
	}

	HK_ON_DEBUG( checkConsistency(); )
}

void hknpModifierManager::checkConsistency()
{
	hknpBodyFlags computedGlobalFlags = 0;
	for( int i=0; i<NUM_BODY_FLAGS; i++ )
	{
		if( m_globalBodyFlagCounts[i] > 0 )
		{
			computedGlobalFlags |= ( 1 << i );
		}
	}
	HK_ASSERT(0x37a21770, computedGlobalFlags == m_globalBodyFlags );
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
