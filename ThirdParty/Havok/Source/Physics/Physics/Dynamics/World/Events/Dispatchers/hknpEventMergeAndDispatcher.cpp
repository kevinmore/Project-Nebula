/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/Events/Dispatchers/hknpEventMergeAndDispatcher.h>

#include <Physics/Physics/Dynamics/World/hknpWorld.h>


hknpEventMergeAndDispatcher::hknpEventMergeAndDispatcher(hknpWorld* world)
	: hknpEventDispatcher(world)
{
}

HK_FORCE_INLINE void hknpEventMergeAndDispatcher::merge( const hknpTriggerVolumeEvent& newEvent )
{
	bool entryFound = false;
	int entryIndex = 0;
	const int triggerCount = ( newEvent.m_status == hknpTriggerVolumeEvent::STATUS_ENTERED ) ? +1 : -1;

	for( ; !entryFound && entryIndex < m_triggerVolumeEvents.getSize(); ++entryIndex )
	{
		const hknpTriggerVolumeEvent& currentEvent = m_triggerVolumeEvents[entryIndex].m_triggerVolumeEvent;
		entryFound = currentEvent.m_bodyIds[0] == newEvent.m_bodyIds[0] &&
			currentEvent.m_bodyIds[1] == newEvent.m_bodyIds[1] &&
			currentEvent.m_shapeKeys[0] == newEvent.m_shapeKeys[0] &&
			currentEvent.m_shapeKeys[1] == newEvent.m_shapeKeys[1];
	}

	if( entryFound )
	{
		HK_ASSERT( 0x00022322, entryIndex > 0 && entryIndex <= m_triggerVolumeEvents.getSize() );
		TriggerVolumeEventWithCount& matchingEvent = m_triggerVolumeEvents[entryIndex-1];
		//if( matchingEvent.m_type != event.m_type )
		//{
		//	matchingEvent.m_type = hknpTriggerVolumeEvent::DATA_UPDATED;
		//}
		matchingEvent.m_count += triggerCount;
	}
	else
	{
		m_triggerVolumeEvents.pushBack( TriggerVolumeEventWithCount( newEvent, triggerCount ) );
	}
}

void hknpEventMergeAndDispatcher::exec( const hkCommand& command )
{
	if( command.m_secondaryType == hknpEventType::TRIGGER_VOLUME )
	{
		const hknpTriggerVolumeEvent& event = static_cast<const hknpTriggerVolumeEvent&>(command);
		merge( event );
	}
	else
	{
		hknpEventDispatcher::exec(command);
	}
}

void hknpEventMergeAndDispatcher::flushRemainingEvents()
{
	for(int i = 0; i < m_triggerVolumeEvents.getSize(); ++i)
	{
		int statusCount = m_triggerVolumeEvents[i].m_count;
		m_triggerVolumeEvents[i].m_triggerVolumeEvent.m_status = (statusCount == -1) ?
			hknpTriggerVolumeEvent::STATUS_EXITED :
			( statusCount == 0 ? hknpTriggerVolumeEvent::STATUS_UPDATED : hknpTriggerVolumeEvent::STATUS_ENTERED );
		hknpEventDispatcher::exec(m_triggerVolumeEvents[i].m_triggerVolumeEvent);
	}
	m_triggerVolumeEvents.clear();
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
