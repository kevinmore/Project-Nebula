/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterialPalette.h>

#if !defined(HK_PLATFORM_SPU)

hknpMaterialPaletteEntryId hknpMaterialPalette::addEntry( const char* name )
{
	HK_ASSERT2( 0xaf41ee11, name, "Null name provided." );

	// Don't add the same entry twice
	for( int i=0; i<m_entries.getSize(); i++ )
	{
		if( m_entries[i].m_name.compareTo( name ) == 0 )
		{
			return hknpMaterialPaletteEntryId(i);
		}
	}

	// Create a new entry
	hknpMaterialPaletteEntryId entryId = hknpMaterialPaletteEntryId( m_entries.getSize() );

	hknpMaterialDescriptor& entry = m_entries.expandOne();
	entry.m_name.set( name );

	return entryId;
}

hknpMaterialPaletteEntryId hknpMaterialPalette::addEntry( hknpRefMaterial* material )
{
	HK_ASSERT2( 0xaf41ee14, material, "Null material provided." );

	// Don't add the same entry twice
	for( int i=0; i<m_entries.getSize(); i++ )
	{
		if( m_entries[i].m_material == material )
		{
			return hknpMaterialPaletteEntryId(i);
		}
	}

	// Create a new entry
	hknpMaterialPaletteEntryId entryId = hknpMaterialPaletteEntryId( m_entries.getSize() );

	hknpMaterialDescriptor& entry = m_entries.expandOne();
	entry.m_material = material;

	return entryId;
}

hknpMaterialPaletteEntryId hknpMaterialPalette::addEntry( hknpMaterialId globalMaterialId )
{
	HK_ASSERT2( 0xaf41ee12, globalMaterialId != hknpMaterialId::invalid(), "Invalid material ID provided." );

	// Don't add the same entry twice
	for( int i=0; i<m_entries.getSize(); i++ )
	{
		if( m_entries[i].m_materialId == globalMaterialId )
		{
			return hknpMaterialPaletteEntryId(i);
		}
	}

	// Create a new entry
	hknpMaterialPaletteEntryId entryId = hknpMaterialPaletteEntryId( m_entries.getSize() );

	hknpMaterialDescriptor& entry = m_entries.expandOne();
	entry.m_materialId = globalMaterialId;

	return entryId;
}

#endif	// !defined(HK_PLATFORM_SPU)

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
