/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Math/Vector/Mx/hkMxVectorUtil.h>

#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Dynamics/Modifier/hknpModifier.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>


enum Status
{
	MANIFOLD_CREATED,
	MANIFOLD_DESTROYED,
};

HK_COMPILE_TIME_ASSERT( hknpManifoldStatusEvent::MANIFOLD_CREATED == 0 );
HK_COMPILE_TIME_ASSERT( hknpManifoldStatusEvent::MANIFOLD_DESTROYED == 1 );

const char* hknpManifoldStatusEvent::getStatusAsString() const
{
	static const char* text[2] = {	"CREATED", "DESTROYED" };
	return text[m_status];
}


void hknpManifoldProcessedEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "hknpManifoldProcessedEvent bodyIds=" << m_bodyIds[0].value() << "," << m_bodyIds[1].value() << " normal=" << m_manifold.m_normal;
	for (int i =0; i < m_numContactPoints; i++)
	{
		out << "\n\t\tpos=" << m_manifold.m_positions[i] << " dist=" << m_manifold.m_distances(i);
	}
}

void hknpManifoldStatusEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "hknpManifoldStatusEvent " << getStatusAsString() << " bodyIds=" << m_bodyIds[0].value() << "," << m_bodyIds[1].value();
}

void hknpTriggerVolumeEvent::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "TriggerVolumeEvent bodyIds=" << m_bodyIds[0].value() << "," << m_bodyIds[1].value();
	if ( m_status == STATUS_ENTERED )
	{
		out << " Status=Entered";
	}
	else
	{
		out << " Status=Exited";
	}
}

void hknpReserved0Event::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved";
}

void hknpReserved1Event::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved";
}

void hknpReserved2Event::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved";
}

void hknpReserved3Event::printCommand( hknpWorld* world, hkOstream& out ) const
{
	out << "Reserved";
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
