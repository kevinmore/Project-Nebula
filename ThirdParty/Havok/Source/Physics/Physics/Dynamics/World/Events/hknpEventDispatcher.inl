/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE void hknpEventDispatcher::exec( const hkCommand& command )
{
	const hknpEvent& event = (const hknpEvent&)command;
	hknpEventHandlerInput handleInput;
	handleInput.m_world = m_world;
	handleInput.m_solverData = m_solverData;
	handleInput.m_commandWriter = m_commandWriter;
	handleInput.m_simulationThreadContext = m_simulationThreadContext;
	handleInput.m_bodiesAreReversed = 0;
	HK_ON_DEBUG(handleInput.m_bodiesAreReversed = hkUint32(-1));

	//
	//	fire global events
	//
	{
		for( EntryIdx eId = m_bodyToEntryMap[0]; eId != INVALID_ENTRY; eId = m_entryPool[eId].m_nextEntry )
		{
			m_entryPool[eId].execEntry( handleInput, event );
		}
	}

	//
	//	fire local binary body events
	//
	if( event.isBinaryEvent() )
	{
		const hknpBinaryBodyEvent& binEvent = event.asBinaryEvent();
		for( int k=0; k <2; k++ )
		{
			hknpBodyId bodyId = binEvent.m_bodyIds[k];
			if( bodyId.value() < (hkUint32)m_bodyToEntryMap.getSize() )
			{
				handleInput.m_bodiesAreReversed = k;
				for( EntryIdx eId = m_bodyToEntryMap[bodyId.value()]; eId != INVALID_ENTRY; eId = m_entryPool[eId].m_nextEntry )
				{
					m_entryPool[eId].execEntry( handleInput, event );
				}
			}
		}
	}

	//
	//	fire local unary body events
	//
	else if (event.isUnaryEvent())
	{
		const hknpUnaryBodyEvent& unaryEvent = event.asUnaryEvent();
		hknpBodyId bodyId = unaryEvent.m_bodyId;
		if( bodyId.value() < (hkUint32)m_bodyToEntryMap.getSize() )
		{
			handleInput.m_bodiesAreReversed = 0;
			HK_ON_DEBUG(handleInput.m_bodiesAreReversed = hkUint32(-1));
			for( EntryIdx eId = m_bodyToEntryMap[bodyId.value()]; eId != INVALID_ENTRY; eId = m_entryPool[eId].m_nextEntry )
			{
				m_entryPool[eId].execEntry( handleInput, event );
			}
		}
	}
}

HK_FORCE_INLINE void hknpEventDispatcher::flushRemainingEvents()
{

}

HK_FORCE_INLINE void hknpEventDispatcher::beginDispatch( hknpSolverData* solverData, hknpSimulationThreadContext* tl )
{
	m_solverData = solverData;
	m_simulationThreadContext = tl;
}

HK_FORCE_INLINE void hknpEventDispatcher::endDispatch()
{
	m_solverData = HK_NULL;
	m_simulationThreadContext = HK_NULL;
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
