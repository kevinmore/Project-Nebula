/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#define HKNP_DISPATCH_SPACE_SPLITTER( SPLITTER, FUNC, PARAMS )			\
	if ( SPLITTER->m_type == hknpSpaceSplitter::TYPE_SINGLE)		{  static_cast<hknpSingleCellSpaceSplitter*>(SPLITTER)->FUNC PARAMS;	}				\
	else if ( SPLITTER->m_type == hknpSpaceSplitter::TYPE_DYNAMIC)	{  static_cast<hknpDynamicSpaceSplitter*>(SPLITTER)->FUNC PARAMS;	}		\
	else if ( SPLITTER->m_type == hknpSpaceSplitter::TYPE_GRID)		{  static_cast<hknpGridSpaceSplitter*>(SPLITTER)->FUNC PARAMS;	}

/// Helper method to dispatch different types. 'Const' variant.
#define HKNP_DISPATCH_CONST_SPACE_SPLITTER( SPLITTER, FUNC, PARAMS )	\
	if ( SPLITTER->m_type == hknpSpaceSplitter::TYPE_SINGLE)		{  static_cast<const hknpSingleCellSpaceSplitter*>(SPLITTER)->FUNC PARAMS;	}				\
	else if ( SPLITTER->m_type == hknpSpaceSplitter::TYPE_DYNAMIC)	{  static_cast<const hknpDynamicSpaceSplitter*>(SPLITTER)->FUNC PARAMS;	}		\
	else if ( SPLITTER->m_type == hknpSpaceSplitter::TYPE_GRID)		{ static_cast<const hknpGridSpaceSplitter*>(SPLITTER)->FUNC PARAMS;	}


HK_FORCE_INLINE int hknpSpaceSplitter::getCellIdx(hkVector4Parameter pos) const
{
	int res=0;
	HKNP_DISPATCH_CONST_SPACE_SPLITTER( this, getCellIdxImpl, ( pos, &res ) );
	return res;
}


#if !defined (HK_PLATFORM_SPU)
HK_FORCE_INLINE void hknpSpaceSplitter::calcInitialCellPositions(hknpWorld* world)
{
	HKNP_DISPATCH_SPACE_SPLITTER( this, calcInitialCellPositionsImpl, ( world ) );
}

HK_FORCE_INLINE void hknpSpaceSplitter::applyThreadData( const hknpSpaceSplitterData* threadData, int numThreads, const hkIntSpaceUtil* intSpaceUtil )
{
	HKNP_DISPATCH_SPACE_SPLITTER( this, applyThreadDataImpl, ( threadData, numThreads, intSpaceUtil ) );
}
#endif


HK_FORCE_INLINE void hknpSpaceSplitter::islandActivated(int bodyCount, const hkAabb& aabb)
{
	HKNP_DISPATCH_SPACE_SPLITTER( this, islandActivatedImpl, ( bodyCount, aabb ) );
}


HK_FORCE_INLINE int hknpSpaceSplitter::getLinkIdxUnchecked(int cellIdx0, int cellIdx1)
{
	HK_ASSERT( 0xf0fdcd34, cellIdx0<=cellIdx1 && cellIdx0 != HKNP_INVALID_CELL_IDX && cellIdx1 != HKNP_INVALID_CELL_IDX);
	return cellIdx0 + ((HK_HINT_SIZE16(cellIdx1) * HK_HINT_SIZE16(cellIdx1+1))>>1);	// upper triangle only
}

HK_FORCE_INLINE int hknpSpaceSplitter::getLinkIdx(int cellIdx0, int cellIdx1)
{
	HK_ASSERT2(0xad843222, cellIdx0 != HKNP_INVALID_CELL_IDX || cellIdx1 != HKNP_INVALID_CELL_IDX, "This function doesn't work for two invalid cell indices.");

	if( cellIdx1 < cellIdx0 )
	{
		if ( cellIdx0 == HKNP_INVALID_CELL_IDX )
		{
			cellIdx0 = cellIdx1;
		}
		return getLinkIdxUnchecked( cellIdx1, cellIdx0 );
	}
	if ( cellIdx1 == HKNP_INVALID_CELL_IDX )
	{
		cellIdx1 = cellIdx0;
	}
	return getLinkIdxUnchecked( cellIdx0, cellIdx1 );
}

HK_FORCE_INLINE int HK_CALL hknpSpaceSplitter::isLinkFlipped(int cellIdx0, int cellIdx1)
{
	return hkUint32(cellIdx1-cellIdx0) >> 31;
}



HK_FORCE_INLINE int hknpSpaceSplitter::getNumCells() const
{
	return m_numGridCells;
}

HK_FORCE_INLINE int hknpSpaceSplitter::getNumLinks() const
{
	return m_numLinks;
}


HK_FORCE_INLINE void hknpSpaceSplitterData::reset()
{
	for (int i=0; i<hknpSpaceSplitter::MAX_NUM_GRID_CELLS; i++)
	{
		m_cellCenter[i].setZero();
	}
}

HK_FORCE_INLINE void hknpSpaceSplitterData::Int64Vector4::setZero()
{
	m_values[0] = 0;
	m_values[1] = 0;
	m_values[2] = 0;
	m_values[3] = 0;
}

HK_FORCE_INLINE void hknpSpaceSplitterData::Int64Vector4::add(const Int64Vector4& a)
{
	hkInt64 v0 = a.m_values[0];
	hkInt64 v1 = a.m_values[1];
	hkInt64 v2 = a.m_values[2];
	hkInt64 v3 = a.m_values[3];
	m_values[0] += v0;
	m_values[1] += v1;
	m_values[2] += v2;
	m_values[3] += v3;
}

HK_FORCE_INLINE void hknpSpaceSplitterData::Int64Vector4::add(hkIntVectorParameter a)
{
	int v0 = a.getComponent<0>();
	int v1 = a.getComponent<1>();
	int v2 = a.getComponent<2>();
	int v3 = a.getComponent<3>();
	m_values[0] += v0;
	m_values[1] += v1;
	m_values[2] += v2;
	m_values[3] += v3;
}

HK_FORCE_INLINE void hknpSpaceSplitterData::Int64Vector4::addMul(hkIntVectorParameter a, int b)
{
	int v0 = a.getComponent<0>()*b;
	int v1 = a.getComponent<1>()*b;
	int v2 = a.getComponent<2>()*b;
	int v3 = a.getComponent<3>()*b;
	m_values[0] += v0;
	m_values[1] += v1;
	m_values[2] += v2;
	m_values[3] += v3;
}

HK_FORCE_INLINE void hknpSpaceSplitterData::Int64Vector4::convertToF32( hkVector4& vOut ) const
{
	hkReal v0 = (hkReal)m_values[0];
	hkReal v1 = (hkReal)m_values[1];
	hkReal v2 = (hkReal)m_values[2];
	hkReal v3 = (hkReal)m_values[3];
	vOut.set(v0, v1, v2, v3);
}

template <int I>
HK_FORCE_INLINE hkInt64 hknpSpaceSplitterData::Int64Vector4::getComponent() const
{
	return m_values[I];
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
