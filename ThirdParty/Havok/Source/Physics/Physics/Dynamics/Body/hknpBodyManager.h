/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_BODY_MANAGER_H
#define HKNP_BODY_MANAGER_H

#include <Common/Base/Container/BitField/hkBitField.h>

#include <Physics/Physics/Dynamics/Body/hknpBody.h>
#include <Physics/Physics/Collide/BroadPhase/hknpBroadPhase.h>

class hknpMotionManager;


/// Helper class to manage bodies and associated data.
/// Do not use this directly. Use the world functions instead.
class hknpBodyManager
{
	public:

		/// An iterator to enumerate all allocated bodies, excluding those marked for deletion.
		class BodyIterator
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyManager::BodyIterator );

				HK_FORCE_INLINE void next();			///< Advance to the next allocated body
				HK_FORCE_INLINE bool isValid() const;	///< Is the iterator still valid?

				HK_FORCE_INLINE hknpBodyId getBodyId() const;		///< Get the current body ID
				HK_FORCE_INLINE const hknpBody& getBody() const;	///< Get read only access to the current body

			private:

				friend class hknpBodyManager;

				HK_FORCE_INLINE BodyIterator( const hknpBodyManager& manager );

				const hknpBodyManager& m_bodyManager;
				int m_bodyIndex;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyManager );

		/// Constructor.
		/// The capacity sets the maximum number of bodies that may be allocated.
		/// If userBodyBuffer is NULL, a buffer will be allocated automatically with the requested capacity.
		hknpBodyManager( hknpWorld* world, hknpBody* userBodyBuffer, hkUint32 capacity );

		/// Destructor.
		~hknpBodyManager();

		//
		// Buffer management
		//

		/// Get the size of the body buffer. This is the maximum number of bodies that can be allocated.
		HK_FORCE_INLINE hkUint32 getCapacity() const;

		/// Get read only access to the start of the body buffer.
		HK_FORCE_INLINE const hknpBody* getBodyBuffer() const;

		/// Get read write access to the start of the body buffer. Internal use only.
		HK_FORCE_INLINE hknpBody* accessBodyBuffer();

		/// Relocate and/or resize the body buffer, and resize all associated arrays.
		/// The new buffer may overlap the existing buffer, or even be at the same address with a different capacity.
		/// If buffer is NULL, a buffer will be allocated automatically with the requested capacity.
		/// Returns TRUE if the relocation succeeded.
		hkBool relocateBodyBuffer( hknpBody* buffer, hkUint32 capacity );

		//
		// Body management
		//

		/// Allocate an uninitialized body.
		hknpBodyId allocateBody();

		/// Allocate an uninitialized body with the given body ID, if possible.
		hkResult allocateBody( hknpBodyId bodyId );

		/// Initialize a static body.
		void initializeStaticBody( hknpBody* HK_RESTRICT body, hknpBodyId bodyId, const hknpBodyCinfo& cInfo );

		/// Initialize a dynamic body.
		void initializeDynamicBody( hknpBody* HK_RESTRICT body, hknpBodyId bodyId, const hknpBodyCinfo& cInfo );

		/// Mark a body for deletion.
		/// This invalidates the body but does not yet free the ID.
		HK_FORCE_INLINE void markBodyForDeletion( hknpBodyId id );

		/// Free the IDs of all marked bodies.
		void deleteMarkedBodies();

		/// Get the number of allocated bodies.
		HK_FORCE_INLINE hkUint32 getNumAllocatedBodies() const;

		/// Get a body ID which is equal to or greater than all allocated IDs.
		/// Note: The returned ID may not refer to a currently allocated body.
		HK_FORCE_INLINE hknpBodyId getPeakBodyId() const;

		/// Get an iterator over all valid bodies (allocated and not marked for deletion).
		HK_FORCE_INLINE BodyIterator getBodyIterator() const;

		/// Sort the linked list of free bodies. Also recalculate the peak body ID.
		void rebuildFreeList();

		/// Fill a construction info based on an existing body.
		void getBodyCinfo( hknpBodyId bodyId, hknpBodyCinfo& cinfoOut ) const;

		/// Get read only access to a body. Does not check if the body is valid.
		HK_FORCE_INLINE const hknpBody& getBody( hknpBodyId id ) const;

		/// Get read write access to a body. Does not check if the body is valid.
		HK_FORCE_INLINE hknpBody& accessBody( hknpBodyId id );

		//
		// Body properties
		//

		/// Set a body name.
		HK_FORCE_INLINE void setBodyName( hknpBodyId bodyId, const char* name );

		/// Get a body name.
		HK_FORCE_INLINE const char* getBodyName( hknpBodyId bodyId ) const;

		/// Find a body for a given name.
		HK_FORCE_INLINE hknpBodyId findBodyByName( const char* name );

		/// Set a property on a body.
		template< typename T >
		HK_FORCE_INLINE void setProperty( hknpBodyId bodyId, hknpPropertyKey key, const T& value );

		/// Get a property from a body. Returns HK_NULL if the property is not set on the body.
		template< typename T >
		HK_FORCE_INLINE T* getProperty( hknpBodyId bodyId, hknpPropertyKey key ) const;

		/// Clear a property from a body, if present.
		void clearProperty( hknpBodyId bodyId, hknpPropertyKey key );

		/// Clear a property from all bodies, if present.
		void clearPropertyFromAllBodies( hknpPropertyKey key );

		/// Clear all properties from a body.
		void clearAllPropertiesFromBody( hknpBodyId bodyId );

		/// Clear all properties from all bodies.
		void clearAllPropertiesFromAllBodies();

		//
		// Active body cache
		//

		/// Get the number of active bodies.
		HK_FORCE_INLINE hkUint32 getNumActiveBodies() const;

		/// Get all active bodies (with 4 extra padded zeros after back()).
		const hkArray<hknpBodyId>& getActiveBodies();

		/// Prefetch all active bodies.
		void prefetchActiveBodies();

		/// Rebuild the cached active body array deterministically.
		void rebuildActiveBodyArray();

		//
		// Other
		//

		HK_FORCE_INLINE const hkArray<hkAabb16>& getPreviousAabbs() const { return m_previousAabbs; }
		HK_FORCE_INLINE hkArray<hkAabb16>& accessPreviousAabbs() { return m_previousAabbs; }

		HK_FORCE_INLINE hknpCellIndex getCellIndex( hknpBodyId id ) const { return m_bodyIdToCellIndexMap[ id.value()]; }

		/// Rebuild the cached m_bodyIdToCellIdx map deterministically.
		void rebuildBodyIdToCellIndexMap();

		/// Update the previous AABBs, needs to executed after you run the broad phase.
		void updatePreviousAabbsOfActiveBodies();

		/// Reset the previous AABB and the temp flags for static bodies.
		void resetPreviousAabbsAndFlagsOfStaticBodies();

		/// Check internal consistency.
		void checkConsistency() const;

	protected:

		/// activate a group of bodies, called by the hknpDeactivationManager or world for new bodies
		void addActiveBodyGroup( hknpBodyId bodyId );

		/// deactivate a group of bodies, called by the hknpDeactivationManager
		void removeActiveBodyGroup( hknpBodyId bodyId );

		/// add a single body to an already existing active group
		void addSingleBodyToActiveGroup( hknpBodyId bodyId );

		/// remove a single body from a group of active bodies
		void removeSingleBodyFromActiveList( hknpBodyId bodyId );

		HK_FORCE_INLINE void updateBodyToCellIndexTable( hknpBodyId firstBodyOfGroup, hknpCellIndex cellIndex );

		/// Remove a single body from the lists of static and dynamic bodies pending to add. Returns true if the body
		/// was present in one of the lists.
		HK_FORCE_INLINE hkBool32 removeSingleBodyFromPendingAddLists( hknpBodyId bodyId );

		HK_FORCE_INLINE bool isBodyWaitingToBeAdded( hknpBodyId id ) const;


		enum ScheduledBodyFlagsEnum {
			ADD_ACTIVE				= 1<<0,
			ADD_INACTIVE			= 1<<2,
			REBUILD_MASS_PROPERTIES	= 1<<3,
			TO_BE_DELETED			= 1<<4,
			MOVED_STATIC			= 1<<5,
		};
		typedef hkFlags<ScheduledBodyFlagsEnum, hkUint16> ScheduledBodyFlags;

		typedef hknpBodyId::Type BodyIndexType;

		enum
		{
			INVALID_BODY_INDEX = hknpBodyId::INVALID,
		};

		struct ScheduledBodyChange
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyManager::ScheduledBodyChange );
			HK_FORCE_INLINE void clear();

			hknpBodyId			m_bodyId;
			ScheduledBodyFlags	m_scheduledBodyFlags;
			BodyIndexType		m_pendingAddIndex;
		};


		void appendToPendingAddList( const hknpBodyId* ids, int numIds, ScheduledBodyFlags addFlag );

		void clearPendingAddLists();

		HK_FORCE_INLINE	void setScheduledBodyFlags( hknpBodyId id, ScheduledBodyFlags flags );

		HK_FORCE_INLINE void clearScheduledBodyFlags( hknpBodyId id, ScheduledBodyFlags flags );

		HK_FORCE_INLINE ScheduledBodyFlags getScheduledBodyFlags( hknpBodyId bodyId ) const;

		void clearAllScheduledBodyChanges();

	public:

		// A buffer to store a set of properties.
		struct PropertyBuffer
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpBodyManager::PropertyBuffer );

			int m_bufferSize;		///< Size of the buffer, in bytes
			int m_propertySize;		///< Size of a single property, in bytes (for debugging)
			hkBitField m_occupancy;	///< One bit per body to indicate if the property is set for that body
			void* m_properties;		///< One property per body

			static PropertyBuffer* construct( int propertySize, int capacity, const PropertyBuffer* source = HK_NULL );
			static void destruct( PropertyBuffer* buffer );
		};

	protected:

		/// Pointer to the Physics world. Only needed to fire callbacks.
		hknpWorld* m_world;

		/// The world's motion manager.
		hknpMotionManager* m_motionManager;

		//
		// Storage
		//

		/// Storage for all bodies (allocated and free).
		hkArray<hknpBody> m_bodies;

		/// Whether the body buffer was allocated by the user or by this manager.
		hkBool m_bodyBufferIsUserOwned;

		/// The first element of a linked list of free bodies.
		hknpBodyId m_firstFreeBodyId;

		/// Previous body AABBs.
		/// This must be the same size as m_bodies.
		
		hkArray<hkAabb16> m_previousAabbs;

		/// Optional body names.
		/// This is lazily allocated with the same size as m_bodies by setBodyName(), when a non-null name is set.
		hkArray<hkStringPtr> m_bodyNames;

		/// Optional generic body properties.
		/// Each key's property buffer is lazily allocated with the same size as m_bodies by setBodyProperty().
		hkMap< hknpPropertyKey, PropertyBuffer* > m_propertyMap;

		//
		// Caches
		//

		/// Number of allocated bodies, including special body 0.
		hkUint32 m_numAllocatedBodies;

		/// Number of bodies scheduled for deletion.
		hkUint32 m_numMarkedBodies;

		/// Peak index of all bodies that were allocated.
		hkUint32 m_peakBodyIndex;

		/// A cached list of active bodies (hknpBody::isActive()).
		hkArray<hknpBodyId> m_activeBodyIds;

		/// A cached map of bodyId to body->motion.m_cellIndex, to allow for faster dispatching.
		hkArray<hknpCellIndex> m_bodyIdToCellIndexMap;

		//
		// Pending changes
		//

		/// List of scheduled body changes.
		hkArray<ScheduledBodyChange> m_scheduledBodyChanges;

		/// List of indices into the list of scheduled body changes (m_scheduledBodyChanges).
		/// Is INVALID_BODY_INDEX for bodies with no scheduled changes.
		hkArray<BodyIndexType> m_scheduledBodyChangeIndices;

		/// Bodies which need to be added to the world.
		hkArray<hknpBodyId> m_bodiesToAddAsActive;

		/// Bodies which need to be added to the world (that start inactive).
		hkArray<hknpBodyId> m_bodiesToAddAsInactive;


		friend class hknpWorld;
		friend class hknpWorldEx;
		friend class hknpMotionManager;
		friend class hknpDeactivationManager;
		friend class hknpWorldShiftUtil;
		friend class hknpShapeManager;
};

typedef hknpBodyManager::BodyIterator hknpBodyIterator;

#include <Physics/Physics/Dynamics/Body/hknpBodyManager.inl>


#endif // HKNP_BODY_MANAGER_H

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
