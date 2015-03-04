/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MATERIAL_LIBRARY_H
#define HKNP_MATERIAL_LIBRARY_H

#include <Common/Base/Types/hkSignalSlots.h>
#include <Common/Base/Container/FreeListArray/hkFreeListArray.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>


/// A library of materials.
/// WARNING: This class is not thread safe. If you are calling non-const methods from different threads make sure to
/// synchronize access properly.
class hknpMaterialLibrary : public hkReferencedObject
{
	public:

		typedef hkFreeListArray<hknpMaterial, hknpMaterialId, 8, hknpMaterial::FreeListArrayOperations> FreeListArray;

		/// Signals fired by the library. Subscribe to signal data members to receive them.
		HK_DECLARE_SIGNAL( MaterialAddedSignal, hkSignal1<hknpMaterialId> );
		HK_DECLARE_SIGNAL( MaterialModifiedSignal, hkSignal1<hknpMaterialId> );
		HK_DECLARE_SIGNAL( MaterialRemovedSignal, hkSignal1<hknpMaterialId> );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Construct a material library with an initial capacity.
		/// This also adds the preset materials.
		hknpMaterialLibrary( int initialCapacity = 16 );

		/// Serialization constructor.
		HK_FORCE_INLINE hknpMaterialLibrary( hkFinishLoadedObjectFlag flag );

		/// Add a material.
		/// If another identical material already exists and material.m_isExclusive is set to FALSE,
		/// this returns the existing material's ID rather than allocating a new material.
		hknpMaterialId addEntry( const hknpMaterial& material );

		/// Add a material via a descriptor.
		/// If the material descriptor contains a valid material ID, this will simply return that ID if it is already
		/// present in the library. Otherwise, this will first try to find a matching material by name, if provided.
		/// Otherwise this will allocate a new material, if provided. The resulting material ID is returned.
		hknpMaterialId addEntry( const hknpMaterialDescriptor& descriptor );

		/// Get read-only access to a material.
		HK_FORCE_INLINE const hknpMaterial& getEntry( hknpMaterialId id ) const;

		/// Update an existing material.
		void updateEntry( hknpMaterialId id, const hknpMaterial& material );

		/// Remove a material.
		void removeEntry( hknpMaterialId id );

		/// Find a material by name.
		/// Returns hknpMaterial::invalid() if not found.
		hknpMaterialId findEntryByName( const char* name ) const;

		/// Get read-only access to the material storage buffer.
		/// Note: Use getIterator() instead to iterate over all allocated entries.
		HK_FORCE_INLINE const hknpMaterial* getBuffer() const;

		/// Get the capacity of the storage buffer.
		HK_FORCE_INLINE int getCapacity() const;

		/// Get an iterator over all allocated materials.
		HK_FORCE_INLINE FreeListArray::Iterator getIterator() const;

	public:

		/// Signal fired after allocating a new material in addEntry()
		MaterialAddedSignal m_materialAddedSignal;			//+nosave +overridetype(void*)

		/// Signal fired when a material is modified in updateEntry()
		MaterialModifiedSignal m_materialModifiedSignal;	//+nosave +overridetype(void*)

		/// Signal fired just before releasing a material in removeEntry()
		MaterialRemovedSignal m_materialRemovedSignal;		//+nosave +overridetype(void*)

	protected:

		/// The free list array of materials.
		FreeListArray m_entries;
};

#include <Physics/Physics/Dynamics/Material/hknpMaterialLibrary.inl>


#endif // HKNP_MATERIAL_LIBRARY_H

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
