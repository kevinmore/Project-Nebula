/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SPACE_SPLITTER_H
#define HKNP_SPACE_SPLITTER_H

#include <Common/Base/Math/Vector/hkIntVector.h>

struct hknpSpaceSplitterData;
class hkIntSpaceUtil;
class hkAabb;

/// A link is represents two connected hknpCellIds.
/// It is used to schedule solver algorithms which work on two set of objects from 2 cells.
typedef hkHandle<hkUint8,0xff,struct hknpLinkUniqueId> hknpLinkId;


/// The space splitter groups bodies into cells. This is the basis for multithreading the engine.
class hknpSpaceSplitter : public hkBaseObject
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSpaceSplitter );

		/// Type of the space splitter.
		enum SpaceSplitterType
		{
			TYPE_SINGLE,	///< Single cell for single threaded simulation.
			TYPE_DYNAMIC,	///< Automatic, gives best multi threading results.
			TYPE_GRID,		///< Fixed grid, suboptimal multi threading.
		};

		enum
		{
			MAX_NUM_GRID_CELLS = 16,
			MAX_SPACE_SPLITTER_SIZE = 0x100 + sizeof(hkVector4) * MAX_NUM_GRID_CELLS,
		};

		/// A link connecting two cells.
		struct Link
		{
			hknpCellIndex m_cellA;	///< Cell index A.
			hknpCellIndex m_cellB;	///< Cell index B.

			Link() {}
			Link(hknpCellIndex cellA, hknpCellIndex cellB) : m_cellA(cellA), m_cellB(cellB) {}
		};

	public:

		/// Ctor
		hknpSpaceSplitter(SpaceSplitterType type);

		/// Destructor.
		virtual ~hknpSpaceSplitter() {}

		/// Calculates the cell index for a given world space position.
		HK_FORCE_INLINE int getCellIdx(hkVector4Parameter pos) const;

		/// Return the number of cells.
		HK_FORCE_INLINE	int getNumCells() const;

		// the connection of 2 cells is a link
		static HK_FORCE_INLINE	int getLinkIdx(int cellIdx0, int cellIdx1);

		/// returns 1 if cellIdx0 > cellIdx1
		static HK_FORCE_INLINE int HK_CALL isLinkFlipped(int cellIdx0, int cellIdx1);

		// the connection of 2 cells is a link if cellIdxx != INVALID_CELL_INDEX and cellIdx0<=cellIdx1
		static HK_FORCE_INLINE	int getLinkIdxUnchecked(int cellIdx0, int cellIdx1);

		/// Returns the maximum number of links we have to solve.
		/// This is the number of items in the upper triangle
		/// where we have numCells rows and columns.
		HK_FORCE_INLINE	int getNumLinks() const;


		/////////////////////
		// Internal section

		HK_FORCE_INLINE void islandActivated(int bodyCount, const hkAabb& aabb);

		/// Set the number of cells.
		void setNumCells(int numCells);

#if !defined (HK_PLATFORM_SPU)

		HK_FORCE_INLINE void applyThreadData( const hknpSpaceSplitterData* threadData, int numThreads, const hkIntSpaceUtil* intSpaceUtil );

		/// Called when the world coordinate system has been shifted by adding 'offset' to all positions.
		virtual void onShiftWorld(hkVector4Parameter offset) = 0;

		/// Returns the size of the object, used for transferring it to SPU
		virtual int getSize() const = 0;

		HK_FORCE_INLINE void calcInitialCellPositions(hknpWorld* world); // Not yet used.

		/// Build the m_linksSortedForMultithreadedSolving array.
		void initSortedLinks(int numCells);

		/// Schedule a set of links sorted by task length (longer tasks come before shorter ones).
		/// numCells must be even.
		/// numLinks should be less than or equal to numCells*(numCells+1) / 2.
		/// sortedTasks and outLinks have to be of size numLinks.
		/// For good parallelism, numCells must be at least equal to double the number of threads.
		static void scheduleLinks(int numCells, int numLinks, Link* sortedLinks, Link* outLinks);

#endif

	public:

		/// Helper array which has all possible links between cells sorted in a multithreaded friendly way.
		hkArray<Link> m_linksSortedForMultithreadedSolving;

		/// The type of the space splitter
		hkPadSpu<hkUint32 > m_type;

	protected:

		/// The number of cells.
		hkPadSpu<int> m_numGridCells;

		/// The number of links
		hkPadSpu<int> m_numLinks;
};


/// Thread local space splitter update information
struct hknpSpaceSplitterData
{
	// Helper class for space splitter data.
	class Int64Vector4
	{
	public:

		HK_FORCE_INLINE void setZero();

		HK_FORCE_INLINE void add(const Int64Vector4& a);

		HK_FORCE_INLINE void add(hkIntVectorParameter a);

		HK_FORCE_INLINE void addMul(hkIntVectorParameter a, int b);

		HK_FORCE_INLINE void convertToF32( hkVector4& vOut ) const;

		template <int I>
		HK_FORCE_INLINE hkInt64 getComponent() const;

		hkInt64 m_values[4];
	};

	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSpaceSplitterData);

	HK_FORCE_INLINE void reset();

	Int64Vector4 m_cellCenter[hknpSpaceSplitter::MAX_NUM_GRID_CELLS];
};


/// A dummy space splitter used when running single threaded simulation. getCellIdx always returns 0.
class hknpSingleCellSpaceSplitter : public hknpSpaceSplitter
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSingleCellSpaceSplitter );

		hknpSingleCellSpaceSplitter();
		HK_FORCE_INLINE void getCellIdxImpl(hkVector4Parameter pos, int *cellOut) const {*cellOut=0;}
		HK_FORCE_INLINE void islandActivatedImpl(int bodyCount, const hkAabb& aabb) {}

#if !defined (HK_PLATFORM_SPU)

		HK_FORCE_INLINE void applyThreadDataImpl( const hknpSpaceSplitterData* threadData, int numThreads, const hkIntSpaceUtil* intSpaceUtil ) {}
		virtual void onShiftWorld(hkVector4Parameter offset) {}
		virtual int getSize() const;
		HK_FORCE_INLINE void calcInitialCellPositionsImpl(hknpWorld* world) {}

#endif
};


/// A fixed space splitter for multi threading. It assigns cells based on a static grid.
class hknpGridSpaceSplitter : public hknpSpaceSplitter
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpGridSpaceSplitter );

		hknpGridSpaceSplitter(hkReal cellSize, int numCells);
		void getCellIdxImpl(hkVector4Parameter pos, int *cellOut) const;
		HK_FORCE_INLINE void islandActivatedImpl(int bodyCount, const hkAabb& aabb) {}

#if !defined (HK_PLATFORM_SPU)

		HK_FORCE_INLINE void applyThreadDataImpl( const hknpSpaceSplitterData* threadData, int numThreads, const hkIntSpaceUtil* intSpaceUtil ) {}
		virtual void onShiftWorld(hkVector4Parameter offset);
		virtual int getSize() const;
		HK_FORCE_INLINE void calcInitialCellPositionsImpl(hknpWorld* world) {}

#endif

	public:

		hkVector4 m_origin;
		hkSimdReal m_shiftAxis0;
		hkSimdReal m_shiftAxis1;

		hkVector4 m_axis0;	// the axis times inverted cell size
		hkVector4 m_axis1;

		hkPadSpu<int> m_numSplits;		///< sqrt(numCells)
		hkPadSpu<int> m_splitMask;
};


/// An adaptive space splitter used for multi threading.
class hknpDynamicSpaceSplitter : public hknpSpaceSplitter
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpDynamicSpaceSplitter );

		hknpDynamicSpaceSplitter(int numCells);
		void getCellIdxImpl(hkVector4Parameter pos, int *cellOut) const;

		void islandActivatedImpl(int bodyCount, const hkAabb& aabb);

#if !defined (HK_PLATFORM_SPU)

		void applyThreadDataImpl( const hknpSpaceSplitterData* threadData, int numThreads, const hkIntSpaceUtil* intSpaceUtil );
		virtual void onShiftWorld(hkVector4Parameter offset);
		virtual int getSize() const;
		void calcInitialCellPositionsImpl(hknpWorld* world);

#endif

	public:

		hkInt32		m_totalPopulation;
		hkInt32		m_populationCount[MAX_NUM_GRID_CELLS];
		hkVector4	m_clusterCenters[MAX_NUM_GRID_CELLS];
};

#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.inl>


#endif // HKNP_SPACE_SPLITTER_H

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
