/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/hknpCollideSharedData.h>

#if !defined(HK_PLATFORM_SPU)

hknpInternalCollideSharedData::hknpInternalCollideSharedData( hknpWorld* world )	// constructor used for single threaded simulation
{
	hknpInternalCollideSharedData* HK_RESTRICT self = this;
	self->m_solverInfo			= &world->m_solverInfo;
	self->m_collisionTolerance	= world->m_collisionTolerance;
	self->m_spaceSplitter		= world->m_spaceSplitter;
	self->m_spaceSplitterSize	= world->m_spaceSplitter->getSize();
	self->m_intSpaceUtil		= &world->m_intSpaceUtil;
	self->m_bodies				= world->m_bodyManager.accessBodyBuffer();
	self->m_motions				= world->m_motionManager.accessMotionBuffer();
	self->m_enableRebuildCdCaches1 = hkUint32(~0);
	self->m_enableRebuildCdCaches2 = 0;
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
