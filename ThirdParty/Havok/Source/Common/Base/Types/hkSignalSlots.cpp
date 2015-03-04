/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/hkSignalSlots.h>
#include <Common/Base/System/hkBaseSystem.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#ifdef HK_ENABLE_SIGNAL_TRACKING

namespace hkSignalInternals
{
	struct	SpinLock
	{
		struct Scope { Scope(SpinLock& l) : m_lock(&l) { l.enter(); } ~Scope() { m_lock->leave(); } SpinLock* m_lock; };

		HK_FORCE_INLINE					SpinLock() : m_value(0) {}
		HK_FORCE_INLINE					~SpinLock()				{ HK_ASSERT2(0xADAB24F0,m_value==0,"Unreleased lock"); }
		HK_FORCE_INLINE void			enter()					{ for(;;) { if(!atomicAdd(&m_value,1)) break; else atomicAdd(&m_value,-1),spin(); }}
		HK_FORCE_INLINE void			leave()					{ atomicAdd(&m_value,-1); }
		HK_FORCE_INLINE void			spin() const			{ do {} while(*(volatile hkUint32*)&m_value); }
		HK_FORCE_INLINE static hkUint32	atomicAdd( volatile hkUint32* value,int increment)
		{
			return hkCriticalSection::atomicExchangeAdd((hkUint32*)value,increment);
		}
	private:
		hkUint32	m_value;
	};
}

using namespace hkSignalInternals;

//
hkSignal*		hkSignal::s_root;
static SpinLock	hkSignal_lock;

//
void hkSignal::beginTrack(hkSignal* signal)
{
	SpinLock::Scope	lock(hkSignal_lock);

	// Add this signal to the beginning of the double-linked list rooted at s_root.
	if ( s_root != HK_NULL ) s_root->m_links[0] = signal;
	signal->m_links[0]	=	HK_NULL;
	signal->m_links[1]	=	hkSignal::s_root;
	s_root				=	signal;
}

//
void hkSignal::endTrack(hkSignal* signal)
{
	SpinLock::Scope	lock(hkSignal_lock);

	// Remove this signal from the double-linked list.
	if(signal->m_links[1]) signal->m_links[1]->m_links[0] = signal->m_links[0];
	if(signal->m_links[0]) signal->m_links[0]->m_links[1] = signal->m_links[1]; else hkSignal::s_root = signal->m_links[0];
}

#endif

#ifndef HK_PLATFORM_SPU

//
int	hkSignal::getNumSubscriptions() const
{
	int n=0;
	for(hkSlot* s = getSlots(); s; s=s->getNext())
	{
		n += s->hasNoSubscription() ? 0 : 1;
	}
	return n;
}

//
void hkSignal::unsubscribeAll(void* object)
{
	HK_ON_DEBUG(int numFound = 0);
	SlotList* prev = &m_slots;
	for(hkSlot* slot = getSlots(), *next = HK_NULL; slot; slot = next)
	{
		next = slot->getNext();
		if(slot->m_object == object) 
		{
			HK_ON_DEBUG(numFound++);
			if (m_slots.getInt())
			{
				// the slot list is in use, delay the deletion
				slot->unsubscribe();
				prev = &slot->m_next;
			}
			else
			{
				// delete the slot directly
				prev->setPtr(next);
				delete slot;
			}
		}
		else
		{
			prev = &slot->m_next;
		}

	}
	#ifdef HK_DEBUG
	if(!numFound)
	{
		HK_WARN_ALWAYS(0xC21C0FB9, "No subscription(s) found.");
	}
	#endif
}

//
hkBool hkSignal::unsubscribeInternal(void* object, const void* method, int length)
{
	SlotList* prev = &m_slots;
	for(hkSlot* slot = getSlots(); slot; slot = slot->getNext())
	{
		if(slot->m_object == object && slot->matchMethod(method, length))
		{
			if (m_slots.getInt())
			{
				// the slot list is in use, delay the deletion
				slot->unsubscribe();
			}
			else
			{
				// delete the slot directly
				prev->setPtr(slot->getNext());
				delete slot;
			}
			return true;
		}
		prev = &slot->m_next;
	}
	return false;
}

//
hkSlot* hkSignal::find(void* object, const void* method, int length)
{
	for(hkSlot* slot = getSlots(); slot; slot = slot->getNext())
	{
		if(slot->m_object == object && !slot->hasNoSubscription() && slot->matchMethod(method, length))
		{
			return slot;
		}
	}
	return HK_NULL;
}

//
void hkSignal::destroy()
{
	if(hkBaseSystem::isInitialized())
	{
		#ifdef HK_SIGNAL_DEBUG
		if(getNumSubscriptions())
		{
			HK_WARN_ALWAYS(0x2AC66B60, "" << getNumSubscriptions() << " slot(s) still connected on '" << getName() << "' destruction");
			printReport();
		}
		#endif
		reset();
	}
}

//
void hkSignal::reset()
{
	HK_ASSERT(0x1a3e92cf, m_slots.getInt() == 0);
	if(m_slots)
	{
		for(hkSlot *slot = m_slots, *next = slot->getNext(); slot;)
		{
			delete slot;
			slot = next;
			next = next ? next->getNext() : HK_NULL;
		}
		m_slots = HK_NULL;
	}
}

//
void hkSignal::printReport() const
{
	HK_TRACE("<" << getName() << ">(" << this << ")");
	for(const hkSlot* slot = getSlots(); slot; slot = slot->getNext())
	{
		if(!slot->hasNoSubscription())
		{
			HK_TRACE("\t["<< slot->getName() << "](" << slot << ")");
		}
	}
}

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
