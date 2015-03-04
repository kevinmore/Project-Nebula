/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.h>

HK_COMPILE_TIME_ASSERT(sizeof(hknpMotionPropertiesLibrary::MotionPropertiesAddedSignal) == sizeof(void *));
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotionPropertiesLibrary::MotionPropertiesModifiedSignal) == sizeof(void *));
HK_COMPILE_TIME_ASSERT(sizeof(hknpMotionPropertiesLibrary::MotionPropertiesRemovedSignal) == sizeof(void *));


hknpMotionPropertiesLibrary::hknpMotionPropertiesLibrary( int initialCapacity )
{
	HK_ASSERT2( 0x1212c9f6, initialCapacity >= hknpMotionPropertiesId::NUM_PRESETS,
		"Motion properties library capacity must be >= hknpMotionPropertiesId::NUM_PRESETS" );

#if defined HK_PLATFORM_HAS_SPU
	// On PlayStation(R)3 we have a hard limit on the number of entries supported on SPU
	HK_ASSERT( 0x1212c9f7, initialCapacity <= HKNP_MAX_NUM_MOTION_PROPERTIES_ON_SPU );
#endif

	
}

hknpMotionPropertiesLibrary::hknpMotionPropertiesLibrary( hkFinishLoadedObjectFlag flag )
:	hkReferencedObject(flag),
	m_entries(flag)
{

}

void hknpMotionPropertiesLibrary::initialize( const hknpWorld* world )
{
	m_entries.clear();

	// Add the presets
	const hkReal gravity = world->getGravity().length<3>().getReal();
	const hkReal unitScale = world->getSolverInfo().m_unitScale.getReal();
	HK_ASSERT( 0x3b38355b, hkMath::isFinite(gravity) && hkMath::isFinite(unitScale) );
	for( int i = 0; i < hknpMotionPropertiesId::NUM_PRESETS; ++i )
	{
		hknpMotionProperties props;
		props.setPreset( hknpMotionPropertiesId::Preset(i), gravity, unitScale );
		props.m_isExclusive = true;

		HK_ON_DEBUG( hknpMotionPropertiesId id = ) addEntry(props);
		HK_ASSERT( 0xf0dfede5, id.value() == i );
	}
}

hknpMotionPropertiesId hknpMotionPropertiesLibrary::addEntry( const hknpMotionProperties& motionProperties )
{
	motionProperties.checkConsistency();

	if( !motionProperties.m_isExclusive )
	{
		// Check if there is already an entry like this
		// NOTE: this can only match allocated free list entries, no need to check if allocated
		int index = m_entries.getStorage().indexOf( motionProperties );
		if( index != -1 )
		{
			return hknpMotionPropertiesId(index);
		}
	}

	// Allocate a new entry
	{
		hknpMotionPropertiesId id = m_entries.allocate( motionProperties );
		HK_ON_PLATFORM_HAS_SPU( HK_ASSERT( 0x1ae60028, id.value() < HKNP_MAX_NUM_MOTION_PROPERTIES_ON_SPU ) );
		m_entryAddedSignal._fire(id);
		return id;
	}
}

void hknpMotionPropertiesLibrary::updateEntry(
	hknpMotionPropertiesId id, const hknpMotionProperties& motionProperties )
{
	HK_ASSERT2( 0x49d11caa, m_entries.isAllocated(id), "Invalid motion properties ID" );
	HK_ASSERT2( 0x49d11cab, id.value() >= hknpMotionPropertiesId::NUM_PRESETS, "Cannot modify preset motion properties" );
	motionProperties.checkConsistency();

	if( !(m_entries[id] == motionProperties) )
	{
		m_entries[id] = motionProperties;
		m_entryModifiedSignal._fire(id);
	}
}

void hknpMotionPropertiesLibrary::removeEntry( hknpMotionPropertiesId id )
{
	HK_ASSERT2(0x49d11cba, m_entries.isAllocated(id), "Invalid motion properties ID");
	HK_ASSERT2(0x49d11cab, id.value() >= hknpMotionPropertiesId::NUM_PRESETS, "Cannot remove preset motion properties");
	m_entryRemovedSignal._fire(id);
	m_entries.release(id);
}

void hknpMotionPropertiesLibrary::removeUnusedEntries( const hknpWorld* worlds, int numWorlds )
{
	// Mark entries in use by any valid motions
	hkLocalBitField isInUse(m_entries.getCapacity(), hkBitFieldValue::ZERO);
	{
		for( int wi = 0; wi < numWorlds; ++wi )
		{
			for( hknpMotionIterator it(worlds[wi].m_motionManager); it.isValid(); it.next() )
			{
				isInUse.set( it.getMotion().m_motionPropertiesId.value() );
			}
		}
	}

	// Free unused entries, except presets
	for( int i = hknpMotionPropertiesId::NUM_PRESETS; i < m_entries.getCapacity(); ++i )
	{
		hknpMotionPropertiesId id(i);
		if( !isInUse.get(i) && m_entries.isAllocated(id) )
		{
			m_entries.release(id);
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
