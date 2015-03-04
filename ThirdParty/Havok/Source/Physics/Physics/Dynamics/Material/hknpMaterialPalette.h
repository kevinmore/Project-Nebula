/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MATERIAL_PALETTE_H
#define HKNP_MATERIAL_PALETTE_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>


/// Identifier for entries in a material palette.
HK_DECLARE_HANDLE( hknpMaterialPaletteEntryId, hkUint16, 0xffff );

/// A material palette is a collection of material descriptors. It is used with the hknpUFMShapeTagCodec to shorten
/// material IDs in shape tags and delay binding to global materials until after shape construction.
/// See hknpMaterialDescriptor for more details.
class hknpMaterialPalette : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Empty constructor.
		HK_FORCE_INLINE hknpMaterialPalette() {}

		/// Serialization constructor.
		HK_FORCE_INLINE hknpMaterialPalette( hkFinishLoadedObjectFlag flag );

		/// Add a named entry, for late material binding.
		hknpMaterialPaletteEntryId addEntry( const char* name );

		/// Add a material entry, for late material creation.
		hknpMaterialPaletteEntryId addEntry( hknpRefMaterial* material );

		/// Add an entry using a known global material ID.
		hknpMaterialPaletteEntryId addEntry( hknpMaterialId globalMaterialId );

		/// Gives read-only access to the palette entries.
		HK_FORCE_INLINE const hkArray<hknpMaterialDescriptor>& getEntries() const;

	protected:

		hkArray<hknpMaterialDescriptor> m_entries;
};

#include <Physics/Physics/Dynamics/Material/hknpMaterialPalette.inl>


#endif // HKNP_MATERIAL_PALETTE_H

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
