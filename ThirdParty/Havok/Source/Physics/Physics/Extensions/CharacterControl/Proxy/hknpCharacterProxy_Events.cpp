/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxy.h>
#include <Physics/Physics/Extensions/CharacterControl/Proxy/hknpCharacterProxyListener.h>


void hknpCharacterProxy::addListener(hknpCharacterProxyListener* listener)
{
	HK_ASSERT2(0x5efeeea3, m_listeners.indexOf(listener) < 0, "You tried to add a character proxy listener listener twice");
	m_listeners.pushBack(listener);
	listener->addReference();
	// Special case to let the user know about the initialized shape.
	listener->shapeChangedCallback(this);
}


void hknpCharacterProxy::removeListener(hknpCharacterProxyListener* listener)
{
	const int i = m_listeners.indexOf(listener);
	HK_ASSERT2(0x2c6b3925, i >= 0, "You tried to remove a character proxy listener, which was never added");
	m_listeners.removeAt(i);
	listener->removeReference();
}

void hknpCharacterProxy::onPostSolveSignal( hknpWorld* /*world*/ )
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->characterCallback(this);
	}
}

void hknpCharacterProxy::fireContactAdded(const hknpCollisionResult& point) const
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->contactPointAddedCallback(this, point);
	}
}


void hknpCharacterProxy::fireContactRemoved(const hknpCollisionResult& point) const
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->contactPointRemovedCallback(this, point);
	}
}


void hknpCharacterProxy::fireCharacterInteraction(hknpCharacterProxy* otherProxy, const hkContactPoint& contact)
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->characterInteractionCallback(this, otherProxy, contact);
	}
}


void hknpCharacterProxy::fireObjectInteraction(const hknpCharacterObjectInteractionEvent& input, hknpCharacterObjectInteractionResult& output)
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->objectInteractionCallback(this, input, output);
	}
}


void hknpCharacterProxy::fireConstraintsProcessed(const hkArray<hknpCollisionResult>& manifold, hkSimplexSolverInput& input) const
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->processConstraintsCallback(this, manifold, input);
	}
}


void hknpCharacterProxy::fireShapeChanged(const hknpShape* /*shape*/) const
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->shapeChangedCallback(this);
	}
}


void hknpCharacterProxy::fireTriggerVolumeInteraction(hknpBodyId triggerBodyId, hknpShapeKey triggerShapeKey,
	hknpTriggerVolumeEvent::Status status) const
{
	if(m_listeners.isEmpty()) return;

	for (int i = m_listeners.getSize() - 1; i >= 0; --i)
	{
		m_listeners[i]->triggerVolumeInteractionCallback(this, triggerBodyId, triggerShapeKey, status);
	}
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
