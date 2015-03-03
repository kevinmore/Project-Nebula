/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE void hknpSimulationThreadContext::finalizeCommandWriters()
{
	m_commandWriter->m_writer.finalize();
}

HK_FORCE_INLINE void hknpSimulationThreadContext::beginCommands( int gridEntryIndex ) const
{
	HK_ON_DEBUG( HK_ASSERT(0xf0345467, m_currentGridEntryDebug == -1); )
	HK_ON_DEBUG( m_currentGridEntryDebug = gridEntryIndex );
	m_currentGridEntryRange.setStartPoint( &m_commandWriter->m_writer );
}

HK_FORCE_INLINE void hknpSimulationThreadContext::execCommand( const hkCommand& command ) const
{
	HK_ON_DEBUG( HK_ASSERT2(0xf0345467, m_currentGridEntryDebug != -1, "A beginCommands call should precede adding any commands"); )
	m_commandWriter->exec( command );
	hknpSimulationDeterminismUtil::check(&command);
}

HK_FORCE_INLINE hkCommand* hknpSimulationThreadContext::beginCommand( int size ) const
{
	HK_ON_DEBUG( HK_ASSERT2(0xf0345467, m_currentGridEntryDebug != -1, "A beginCommands call should precede adding any commands"); )
	return (hkCommand*) (m_commandWriter->allocBuffer( size ));
}

HK_FORCE_INLINE void hknpSimulationThreadContext::endCommand( const hkCommand* command ) const
{
	HK_ON_DEBUG( HK_ASSERT2(0xf0345467, m_currentGridEntryDebug != -1, "A beginCommands call should precede adding any commands"); )
	hknpSimulationDeterminismUtil::check(command);
}

HK_FORCE_INLINE void hknpSimulationThreadContext::endCommands( int gridEntryIndex ) const
{
	HK_ON_DEBUG( HK_ASSERT2(0xf0345467, m_currentGridEntryDebug == gridEntryIndex, "There is a mismatch between the beginCommands and endCommands calls"); )
	m_currentGridEntryRange.setEndPoint( &m_commandWriter->m_writer );
	m_commandGrid->addRange( m_commandWriter->m_writer, gridEntryIndex, m_currentGridEntryRange );
	m_currentGridEntryRange.setStartPoint( &m_commandWriter->m_writer );
	HK_ON_DEBUG( m_currentGridEntryDebug = -1; )
}

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE void hknpSimulationThreadContext::appendCommand( const hkCommand& command ) const
{
	HK_ON_DEBUG( HK_ASSERT2(0xf0345467, m_currentGridEntryDebug != -1, "A beginCommands call should precede adding any commands"); )
		m_commandWriter->append( command );
	hknpSimulationDeterminismUtil::check(&command);
}

#endif	// !HK_PLATFORM_SPU

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
