/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_EVENT_MERGER_H
#define HKNP_EVENT_MERGER_H

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>

class hknpWorld;


class hknpEventMergeAndDispatcher : public hknpEventDispatcher
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);

		hknpEventMergeAndDispatcher(hknpWorld* world);

		/// dispatch a command (implements hkSecondaryCommandDispatcher)
		virtual void exec( const hkCommand& command );

		/// execute and remove all events remaining from previously applied merges
		virtual void flushRemainingEvents();

	protected:

		struct TriggerVolumeEventWithCount
		{
			TriggerVolumeEventWithCount(const hknpTriggerVolumeEvent& initEvent, int initCount)
				: m_triggerVolumeEvent(initEvent), m_count(initCount)
			{}

			hknpTriggerVolumeEvent m_triggerVolumeEvent;
			int m_count;
		};

		HK_FORCE_INLINE void merge( const hknpTriggerVolumeEvent& newTriggerVolumeEvent );

		hkArray<TriggerVolumeEventWithCount> m_triggerVolumeEvents;
};

#endif

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
