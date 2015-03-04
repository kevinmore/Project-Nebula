/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

HK_COMPILE_TIME_ASSERT(sizeof(hknpMaterialLibrary::MaterialAddedSignal) == sizeof(void *));
HK_COMPILE_TIME_ASSERT(sizeof(hknpMaterialLibrary::MaterialModifiedSignal) == sizeof(void *));
HK_COMPILE_TIME_ASSERT(sizeof(hknpMaterialLibrary::MaterialRemovedSignal) == sizeof(void *));


hknpMaterialLibrary::hknpMaterialLibrary( int initialCapacity )
{
	HK_ASSERT2( 0x1211c9f6, initialCapacity >= hknpMaterialId::NUM_PRESETS,
		"Material library capacity must be >= hknpMaterialId::NUM_PRESETS" );

	// On PlayStation(R)3 we have a hard limit on the number of materials supported on SPU
	HK_ON_PLATFORM_HAS_SPU( HK_ASSERT(0x1211c9f7, initialCapacity <= HKNP_MAX_NUM_MATERIALS_ON_SPU) );

	m_entries.grow( initialCapacity );

	//
	// Add the preset materials
	//

	for( int i = 0; i < hknpMaterialId::NUM_PRESETS; ++i )
	{
		hknpMaterial material;
		material.m_isExclusive = true;
		HK_ON_DEBUG( hknpMaterialId id = ) addEntry( material );
		HK_ASSERT( 0xf0dfede4, id.value() == i );
	}
}

hknpMaterialId hknpMaterialLibrary::addEntry( const hknpMaterial& inMaterial )
{
	// Ensure flags are synced
	hknpMaterial material = inMaterial;
	material.synchronizeFlags();

	if( !material.m_isExclusive )
	{
		// Check if there is already an identical entry.
		// NOTE: this can only match allocated free list entries, no need to check if allocated.
		int index = m_entries.getStorage().indexOf(material);
		if( index != -1 )
		{
			hknpMaterialId id(index);
			m_entries[id].m_isShared = true;
			return id;
		}
	}

	// Check that there is no other material with the same name
	if( (inMaterial.m_name.getLength() > 0) && findEntryByName( inMaterial.m_name.cString() ).isValid() )
	{
		HK_WARN( 0x684bfcb7, "A different material with the same name already exists. Cannot add entry." );
		return hknpMaterialId::invalid();
	}

	// Allocate a new entry
	{
		hknpMaterialId id = m_entries.allocate(material);
		HK_ON_PLATFORM_HAS_SPU( HK_ASSERT( 0x1ae60028, id.value() < HKNP_MAX_NUM_MATERIALS_ON_SPU ) );
		m_materialAddedSignal.fire(id);
		return id;
	}
}

hknpMaterialId hknpMaterialLibrary::addEntry( const hknpMaterialDescriptor& descriptor )
{
	hknpMaterialId id = descriptor.m_materialId;

	// If the descriptor contains a valid material ID make sure that it is present in the library
	if( id.isValid() && m_entries.isAllocated(id) )
	{
		return id;
	}
	id = hknpMaterialId::invalid();

	// Try to find a matching material by name
	if( descriptor.m_name.getLength() > 0 )
	{
		id = findEntryByName( descriptor.m_name.cString() );
	}

	// Otherwise try to add a new material
	if( !id.isValid() && descriptor.m_material )
	{
		id = addEntry( descriptor.m_material->m_material );
	}

	HK_ASSERT( 0x623f8562, id.isValid() );
	return id;
}

void hknpMaterialLibrary::updateEntry( hknpMaterialId id, const hknpMaterial& inMaterial )
{
	HK_ASSERT2( 0x49d11caa, m_entries.isAllocated(id), "Invalid material ID" );
	HK_ASSERT2( 0x49d11cab, id.value() >= hknpMaterialId::NUM_PRESETS, "Cannot modify preset materials" );

	// Ensure flags are synced
	hknpMaterial material = inMaterial;
	material.synchronizeFlags();

	if( !(m_entries[id] == material) )
	{
		m_entries[id] = material;
		m_materialModifiedSignal.fire(id);
	}
}

void hknpMaterialLibrary::removeEntry( hknpMaterialId id )
{
	HK_ASSERT2( 0x49d11caa, m_entries.isAllocated(id), "Invalid material ID" );
	HK_ASSERT2( 0x49d11cab, id.value() >= hknpMaterialId::NUM_PRESETS, "Cannot remove preset materials" );
	m_materialRemovedSignal.fire(id);
	m_entries.release(id);
}

hknpMaterialId hknpMaterialLibrary::findEntryByName( const char* name ) const
{
	for( int i = 0; i < m_entries.getCapacity(); ++i )
	{
		hknpMaterialId id(i);
		if( m_entries.isAllocated(id) && ( m_entries[id].m_name.compareTo(name) == 0 ) )
		{
			return id;
		}
	}

	return hknpMaterialId::invalid();
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
