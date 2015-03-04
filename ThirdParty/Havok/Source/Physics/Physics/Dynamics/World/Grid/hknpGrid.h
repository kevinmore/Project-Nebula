/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_GRID_H
#define HKNP_GRID_H

#include <Common/Base/Container/BlockStream/hkBlockStream.h>


///
template<typename Entry>
class hknpGrid
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpGrid<Entry> );

		HK_FORCE_INLINE hknpGrid() {}

		HK_FORCE_INLINE void setSize( int size );
		HK_FORCE_INLINE int getSize();

		HK_FORCE_INLINE void clearGrid();
		HK_FORCE_INLINE void setInvalid();
		HK_FORCE_INLINE bool isEmpty() const;	// checks for a non zero size and all elements empty

		HK_FORCE_INLINE int getLinkedNumElements( int entryIndex );	// Gets the number of elements for all linked ranges of this entry.

		HK_FORCE_INLINE const Entry&	operator[](int i) const	{ return m_entries[i]; }
		HK_FORCE_INLINE Entry&			operator[](int i)		{ return m_entries[i]; }

		/// Add a range to a grid, this works even if the grid already has a non-empty range for the specified entry.
		/// Any writer can be used, a common choice is the same writer to which the range's elements are written.
		/// range.m_next is used (see hkBlockStreamBase::LinkedRange) to link the ranges.
		template<typename WRITER, typename RANGE>
		HK_FORCE_INLINE void HK_CALL addRange( WRITER& writer, int entryIndex, const RANGE& range );

	public:

		HK_ALIGN16( hkArray<Entry> ) m_entries;
};


///
template<typename Entry>
class hknpGridEntries
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpGridEntries<Entry> );

		HK_FORCE_INLINE hknpGridEntries()
		{
			m_entries = HK_NULL;
#if defined(HK_PLATFORM_SIM)
			m_numEntries = 0;
#endif
		}

		HK_FORCE_INLINE void set( hknpGrid<Entry>* grid )
		{
			if( grid )
			{
				m_entries = grid->m_entries.begin();
#if defined(HK_PLATFORM_SIM)
				m_numEntries = grid->m_entries.getSize();
#endif
			}
		}

	public:

		hkPadSpu<Entry*> m_entries;
#if defined(HK_PLATFORM_SIM)
		int	m_numEntries;
#endif
};



///
struct hknpIdxRange
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpIdxRange );

	HK_FORCE_INLINE hknpIdxRange() {}
	HK_FORCE_INLINE hknpIdxRange( int start, int numElements );

	HK_FORCE_INLINE void clearRange();
	HK_FORCE_INLINE bool isEmpty() const;

	int m_start;
	int m_numElements;
};

typedef hknpGrid<hknpIdxRange>						hknpIdxRangeGrid;

typedef hknpGrid<hkBlockStreamBase::Range>			hknpCdCacheGrid;
typedef hknpGridEntries<hkBlockStreamBase::Range>	hknpCdCacheGridEntries;

#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.inl>

#endif // HKNP_GRID_H

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
