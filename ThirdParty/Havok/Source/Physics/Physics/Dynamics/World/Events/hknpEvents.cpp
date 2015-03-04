/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEvents.h>

void hknpBodyActivationEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "IslandActivationEvent ";
	if ( m_activated )
	{
		out << " Activated";
	}
	else
	{
		out << " Deactivated";
	}
	{
		out << " Id=" << m_bodyId.value();
	}
}

void hknpBodyExitedBroadPhaseEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "BodyExitedBroadPhaseEvent id=" << m_bodyId.value();
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
