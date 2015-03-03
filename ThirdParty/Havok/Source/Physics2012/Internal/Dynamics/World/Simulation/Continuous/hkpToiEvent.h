/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_TOI_EVENT_H
#define HK_TOI_EVENT_H

class hkpEntity;
class hkpDynamicsContactMgr;
#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>
#include <Physics2012/Dynamics/Constraint/Contact/hkpContactPointProperties.h>

	// Holds information necessary for processing a Time-Of-Impact (TOI) event.
struct hkpToiEvent
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNCOLLIDE, hkpToiEvent );

		HK_DECLARE_POD_TYPE( );

			// Time-Of-Impact, value may be postponed (increased) depending on the qualities of involved bodies.
		HK_ALIGN16( hkTime m_time );

			// Separating velocity of bodies at TOI. This is a negative value, as the bodies are approaching
			// each other.
		hkReal m_seperatingVelocity;

			// Tells if the TOI should use simple handling. Simple handling is only used when debris-quality objects
			// are involved in a collision. It is handled by simply backstepping the debris objects till the initial time
			// of impact, and not reintegrating them at all.
		hkBool m_useSimpleHandling;

			// The two colliding bodies.
		hkpEntity*      m_entities[2];

			// Contact manager.
		hkpDynamicsContactMgr* m_contactMgr;

#if defined (HK_PLATFORM_HAS_SPU)
			// Contact point recorded at the TOI.
		hkContactPoint m_contactPoint;
#endif

			// Friction, restitution and user data.
			// Must be aligned when using Spu.
		hkpContactPointProperties m_properties;

			// Extended user data to store hkpShapeKey hierarchies.
		hkpContactPointProperties::UserData m_extendedUserDatas[HK_NUM_EXTENDED_USER_DATAS_IN_TOI_EVENT];

#if !defined (HK_PLATFORM_HAS_SPU)
			// Contact point recorded at the TOI.
		hkContactPoint m_contactPoint;
#endif
};

#endif // HK_TOI_EVENT_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
