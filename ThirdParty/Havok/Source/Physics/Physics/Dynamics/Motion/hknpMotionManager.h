/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MOTION_MANAGER_H
#define HKNP_MOTION_MANAGER_H

#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>

class hknpSpaceSplitter;
class hknpMotion;
class hknpBodyManager;
struct hknpMotionCinfo;

//HK_ON_SPU( "this should not be included on SPU" );


/// Helper class to manage motions and associated data.
/// Do not use this directly. Use the world functions instead.
class hknpMotionManager
{
	public:

		/// An iterator to enumerate all allocated motions, excluding those marked for deletion.
		class MotionIterator
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionManager::MotionIterator );

				MotionIterator( const hknpMotionManager& manager );

				HK_FORCE_INLINE void next();			///< Advance to the next allocated motion
				HK_FORCE_INLINE bool isValid() const;	///< Is the iterator still valid?

				HK_FORCE_INLINE hknpMotionId getMotionId() const;		///< Get the current motion ID
				HK_FORCE_INLINE const hknpMotion& getMotion() const;	///< Get read only access to the current motion

			private:

				const hknpMotionManager& m_motionManager;
				hknpMotionId::Type m_index;
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionManager );

		/// Constructor.
		hknpMotionManager();

		/// Destructor.
		~hknpMotionManager();

		/// Initialize the manager.
		/// The capacity sets the maximum number of motions that may be allocated.
		/// If userMotionBuffer is NULL, a buffer will be allocated automatically with the requested capacity.
		void initialize( hknpMotion* userMotionBuffer, int capacity, hknpBodyManager* bodyManager, hknpSpaceSplitter& splitter );

		//
		// Buffer management
		//

		/// Get the size of the motion buffer. This is the maximum number of motions that can be allocated.
		HK_FORCE_INLINE hkUint32 getCapacity() const;

		/// Read-only access to the motion buffer.
		HK_FORCE_INLINE const hknpMotion* getMotionBuffer() const;

		/// Read-write access to the motion buffer.
		HK_FORCE_INLINE hknpMotion* accessMotionBuffer();

		/// Relocate and/or resize the motion buffer.
		/// The new buffer may overlap the existing buffer, or even be at the same address with a different capacity.
		/// If buffer is NULL, a buffer will be allocated automatically with the requested capacity.
		/// Returns TRUE if the relocation succeeded.
		hkBool relocateMotionBuffer( hknpMotion* buffer, hkUint32 capacity );

		//
		// Motion management
		//

		/// Allocate an uninitialized motion.
		hknpMotionId allocateMotion();

		/// Initialize a motion from a construction info, set all external references to invalid.
		void initializeMotion( hknpMotion* HK_RESTRICT motion,
			const hknpMotionCinfo& motionCinfo, const hknpSpaceSplitter& spaceSplitter ) const;

		/// Mark a motion for deletion.
		/// This invalidates the motion but does not yet free the ID.
		HK_FORCE_INLINE void markMotionForDeletion( hknpMotionId id );

		/// Free the IDs of all marked motions.
		void deleteMarkedMotions();

		/// Get the number of allocated motions.
		HK_FORCE_INLINE hkUint32 getNumAllocatedMotions() const;

		/// Get the highest motion ID ever used.
		HK_FORCE_INLINE hknpMotionId getPeakMotionId() const;

		/// Get an iterator over all valid motions (allocated and not marked for deletion).
		HK_FORCE_INLINE MotionIterator getMotionIterator() const;

		/// Check if a given ID is valid (allocated and not marked for deletion).
		HK_FORCE_INLINE hkBool32 isMotionValid( hknpMotionId id ) const;

		/// Sort the linked list of free motions.
		void rebuildFreeList();

		/// Get construction info for an existing motion.
		void getMotionCinfo( hknpMotionId motionId, hknpMotionCinfo& cinfoOut ) const;

		//
		// Other
		//

		HK_FORCE_INLINE int getNumCells() const;

		HK_FORCE_INLINE const hkArray<hknpMotionId>& getSolverIdToMotionIdForCell( int cellIdx ) const;

		/// Update the cell index for a motion.
		/// Accesses the bodies to update the bodyId to cellIdx table.
		HK_FORCE_INLINE void updateCellIdx( hknpMotion& motion, hknpMotionId motionId, hknpCellIndex newCellIndex );

		/// Calculate the number of active motions (no fixed motions counted)
		int calcNumActiveMotions() const;

		/// Build a map of global solver ID to motion ID, and a global solver ID startIndex for each cell.
		void buildSolverIdToMotionIdMap( hknpIdxRangeGrid& cellIdxToGlobalSolverIdOut, hkArray<hknpMotionId>& solverIdToMotionIdOut );

		/// Complete deterministic rebuild of the motion histogram.
		void rebuildMotionHistogram( const hknpSpaceSplitter& spaceSplitter );

		/// called when there is a new active motion.
		/// Normally called by the body manager
		void addActiveMotion( hknpMotion& motion, hknpMotionId motionId );
		void removeActiveMotion( hknpMotion& motion, hknpMotionId motionId );

		//
		// For debugging
		//

		// lock motionToSolverId tables for active motions, this should find multi-threaded read write access to the histogram.
		void lockMotionTables() { m_isLocked = true; }

		void unlockMotionTables() { m_isLocked = false; }

		// check the consistency of the histogram
		void checkConsistency();

	protected:

		/// Set the cell index, don't update any secondary data structures
		HK_FORCE_INLINE static void overrideCellIndexInternal( hknpMotion& motion, hknpCellIndex newCellIndex );

	protected:

		struct CellData
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpMotionManager::CellData );

			// Map of local solver ID to motion ID.
			// Includes solver ID 0 mapped to motion ID 0.
			hkArray<hknpMotionId> m_solverIdToMotionId;
		};

	protected:

		/// The world's body manager.
		hknpBodyManager* m_bodyManager;

		//
		// Storage
		//

		/// Storage for all motions (allocated and free).
		hkArray<hknpMotion> m_motions;

		/// Whether the motion buffer was allocated by the user or by this manager.
		hkBool m_motionBufferIsUserOwned;

		/// The first element of a linked list of free motions.
		hknpMotionId m_firstFreeMotionId;

		/// The first element of a linked list of allocated motions marked for deletion.
		hknpMotionId m_firstMarkedMotionId;

		//
		// Caches
		//

		/// Current number of allocated motions, including special motion 0.
		hkUint32 m_numAllocatedMotions;

		/// Current number of motions marked for deletion.
		hkUint32 m_numMarkedMotions;

		/// Peak index of all motions that were allocated.
		hkUint32 m_peakMotionIndex;

		//
		// Other
		//

		/// Data to map solverId to motion.
		hkArray<CellData> m_activeMotionGrid;

		/// Flag to check for safe multi-threaded access to histogram.
		hkBool m_isLocked;


		friend class hknpWorld;
		friend class hknpWorldEx;
		friend class hknpBodyManager;
		friend class hknpMotionConstraintManager;
};

typedef hknpMotionManager::MotionIterator hknpMotionIterator;

#include <Physics/Physics/Dynamics/Motion/hknpMotionManager.inl>


#endif // HKNP_MOTION_MANAGER_H

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
