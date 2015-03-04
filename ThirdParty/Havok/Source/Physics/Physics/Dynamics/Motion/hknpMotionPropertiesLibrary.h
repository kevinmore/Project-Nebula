/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_PROPERTIES_LIBRARY_H
#define HKNP_MOTION_PROPERTIES_LIBRARY_H

#include <Common/Base/Types/hkSignalSlots.h>
#include <Common/Base/Container/FreeListArray/hkFreeListArray.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionProperties.h>


/// A library of motion properties.
/// \sa hknpMotionProperties
class hknpMotionPropertiesLibrary : public hkReferencedObject
{
	public:

		typedef hkFreeListArray<hknpMotionProperties, hknpMotionPropertiesId, 8, hknpMotionProperties::FreeListArrayOperations> FreeListArray;

		/// Signals fired by the library. Subscribe to signal data members to receive them.
		HK_DECLARE_SIGNAL( MotionPropertiesAddedSignal, hkSignal1<hknpMotionPropertiesId> );
		HK_DECLARE_SIGNAL( MotionPropertiesModifiedSignal, hkSignal1<hknpMotionPropertiesId> );
		HK_DECLARE_SIGNAL( MotionPropertiesRemovedSignal, hkSignal1<hknpMotionPropertiesId> );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Construct a motion properties library with an initial capacity.
		hknpMotionPropertiesLibrary( int initialCapacity = 16 );

		/// Serialization constructor.
		hknpMotionPropertiesLibrary( hkFinishLoadedObjectFlag flag );

		/// Initialize the library with preset motion properties appropriate for the given world setup.
		void initialize( const hknpWorld* world );

		/// Add a motion properties.
		/// If another identical entry already exists and motionProperties.m_isExclusive is set to FALSE,
		/// this returns the existing entry's ID rather than allocating a new entry.
		hknpMotionPropertiesId addEntry( const hknpMotionProperties& motionProperties );

		/// Get read-only access to a motion properties.
		HK_FORCE_INLINE const hknpMotionProperties& getEntry( hknpMotionPropertiesId id ) const;

		/// Update an existing motion properties.
		void updateEntry( hknpMotionPropertiesId id, const hknpMotionProperties& motionProperties );

		/// Remove a motion properties.
		void removeEntry( hknpMotionPropertiesId id );

		/// Remove any motion properties that are not in use by any motions in the given worlds.
		void removeUnusedEntries( const hknpWorld* worlds, int numWorlds );

		/// Get read-only access to the motion properties storage buffer.
		/// Note: Use getIterator() instead to iterate over all allocated entries.
		HK_FORCE_INLINE const hknpMotionProperties* getBuffer() const;

		/// Get the capacity of the storage buffer.
		HK_FORCE_INLINE int getCapacity() const;

		/// Get an iterator over all allocated motion properties.
		HK_FORCE_INLINE FreeListArray::Iterator getIterator() const;

	public:

		/// Signal fired after allocating a new motion properties in addEntry()
		MotionPropertiesAddedSignal m_entryAddedSignal;			//+nosave +overridetype(void*)

		/// Signal fired after updating a motion properties in updateEntry()
		MotionPropertiesModifiedSignal m_entryModifiedSignal;	//+nosave +overridetype(void*)

		/// Signal fired just before releasing a motion properties in removeEntry()
		MotionPropertiesRemovedSignal m_entryRemovedSignal;		//+nosave +overridetype(void*)

	protected:

		/// The free list array of motion properties.
		FreeListArray m_entries;
};

#include <Physics/Physics/Dynamics/Motion/hknpMotionPropertiesLibrary.inl>


#endif // HKNP_MOTION_PROPERTIES_LIBRARY_H

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
