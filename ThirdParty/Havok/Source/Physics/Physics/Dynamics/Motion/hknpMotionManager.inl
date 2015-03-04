/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE void hknpMotionManager::MotionIterator::next()
{
	while( ++m_index <= m_motionManager.m_peakMotionIndex )
	{
		if ( m_motionManager.m_motions[m_index].isValid() )
		{
			return;
		}
	}
	m_index = hknpMotionId::InvalidValue;
}

HK_FORCE_INLINE bool hknpMotionManager::MotionIterator::isValid() const
{
	return m_index != hknpMotionId::InvalidValue;
}

HK_FORCE_INLINE hknpMotionId hknpMotionManager::MotionIterator::getMotionId() const
{
	return hknpMotionId(m_index);
}

HK_FORCE_INLINE const hknpMotion& hknpMotionManager::MotionIterator::getMotion() const
{
	return m_motionManager.getMotionBuffer()[m_index];
}

HK_FORCE_INLINE void hknpMotionManager::markMotionForDeletion( hknpMotionId id )
{
	HK_ASSERT2( 0x579c08da, (id.value() > 0) && isMotionValid(id), "Marking an invalid motion for deletion" );

	// Invalidate the motion
	
	hknpMotion& motion = m_motions[id.value()];
	motion.reset();
	motion.m_spaceSplitterWeight = 0;
	motion.m_solverId = hknpSolverId::invalid();

	// Add to linked list
	
	(*(reinterpret_cast<hknpMotionId*>(&motion))) = m_firstMarkedMotionId;
	m_firstMarkedMotionId = id;

	m_numMarkedMotions++;
}

HK_FORCE_INLINE const hkArray<hknpMotionId>& hknpMotionManager::getSolverIdToMotionIdForCell( int cellIdx ) const
{
	return m_activeMotionGrid[cellIdx].m_solverIdToMotionId;
}

HK_FORCE_INLINE int hknpMotionManager::getNumCells() const
{
	return m_activeMotionGrid.getSize();
}

HK_FORCE_INLINE hkUint32 hknpMotionManager::getCapacity() const
{
	return m_motions.getCapacity();
}

HK_FORCE_INLINE hkUint32 hknpMotionManager::getNumAllocatedMotions() const
{
	return m_numAllocatedMotions;
}

HK_FORCE_INLINE hknpMotionId hknpMotionManager::getPeakMotionId() const
{
	return hknpMotionId(m_peakMotionIndex);
}

HK_FORCE_INLINE hknpMotionIterator hknpMotionManager::getMotionIterator() const
{
	return MotionIterator(*this);
}

HK_FORCE_INLINE hkBool32 hknpMotionManager::isMotionValid( hknpMotionId id ) const
{
	return (id.value() <= m_peakMotionIndex) && m_motions[id.value()].isValid();
}

HK_FORCE_INLINE hknpMotion* hknpMotionManager::accessMotionBuffer()
{
	return m_motions.begin();
}

HK_FORCE_INLINE const hknpMotion* hknpMotionManager::getMotionBuffer() const
{
	return m_motions.begin();
}

HK_FORCE_INLINE void hknpMotionManager::updateCellIdx( hknpMotion& motion, hknpMotionId motionId, hknpCellIndex newCellIndex )
{
	if ( newCellIndex == motion.m_cellIndex )
	{
		// this can happen if other commands have exchanged the motion (like detach, attach).
		return;
	}
	//HK_ASSERT( 0xf054f590, newCellIndex != motion.m_cellIndex );
	if ( motion.isActive() )
	{
		removeActiveMotion( motion, motionId );
		motion.m_cellIndex = newCellIndex;
		addActiveMotion( motion, motionId );
	}
	else
	{
		motion.m_cellIndex = newCellIndex;
	}
	m_bodyManager->updateBodyToCellIndexTable( motion.m_firstAttachedBodyId, newCellIndex );
}

HK_FORCE_INLINE void hknpMotionManager::overrideCellIndexInternal( hknpMotion& motion, hknpCellIndex newCellIndex )
{
	motion.m_cellIndex = newCellIndex;
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
