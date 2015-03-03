/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE hknpConstraintSolverJacobianRange2::hknpConstraintSolverJacobianRange2()
HK_ON_DEBUG(: m_solverId(hknpConstraintSolverType::NUM_TYPES))
{
}

HK_FORCE_INLINE void hknpConstraintSolverJacobianRange2::initRange(hknpConstraintSolverType::Enum solverId, Flags flags)
{
	m_solverId = hknpConstraintSolverId(solverId);
	m_flags = flags;
}


HK_FORCE_INLINE hknpConstraintSolverSchedulerGridInfo::hknpConstraintSolverSchedulerGridInfo() {}
HK_FORCE_INLINE void hknpConstraintSolverSchedulerGridInfo::setIsLinkGrid()				{ m_flags.andWith(~CELL_ARRAY); m_flags.orWith(LINK_GRID); }
HK_FORCE_INLINE bool hknpConstraintSolverSchedulerGridInfo::isLinkGrid() const			{ return m_flags.anyIsSet(LINK_GRID); }
HK_FORCE_INLINE void hknpConstraintSolverSchedulerGridInfo::setIsCellArray()			{ m_flags.andWith(~LINK_GRID); m_flags.orWith(CELL_ARRAY); }
HK_FORCE_INLINE bool hknpConstraintSolverSchedulerGridInfo::isCellArray() const			{ return m_flags.anyIsSet(CELL_ARRAY); }
HK_FORCE_INLINE void hknpConstraintSolverSchedulerGridInfo::setPriority(hknpConstraintSolverPriority value)	{ m_priority = value; }
HK_FORCE_INLINE hknpConstraintSolverPriority hknpConstraintSolverSchedulerGridInfo::getPriority() const		{ return m_priority; }

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
