/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_CACHE_DEACTIVATION_POLICY_H
#define HKNP_COLLISION_CACHE_DEACTIVATION_POLICY_H

#include <Physics/Physics/hknpTypes.h>

class hknpCdCacheReader;
class hknpCdCacheWriter;

/// A policy controlling what should happen to deactivated caches.
/// You can either use this class or derive from this class and implement your own policy.
/// This policy runs in a single thread on the PPU.
class hknpDeactiveCdCacheFilter : public hkReferencedObject
{
	//+hk.MemoryTracker(ignore=True)
	public:

		hknpDeactiveCdCacheFilter() : m_deleteInactiveCvxCaches(false), m_deleteInactiveMeshCaches(false) {}

		virtual ~hknpDeactiveCdCacheFilter() {}

		/// deactivate all caches
		virtual void deactivateCaches(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const hkArray<hknpBodyId>& deactivatedBodies,
			hknpCdCacheReader& cdCacheReader, hknpCdCacheStream& inactiveChildCdCacheStreamIn,
			hknpCdCacheWriter& inactiveCdCacheWriter, hknpCdCacheWriter& inactiveChildCdCacheWriter,
			hkArray<hknpBodyIdPair>& deactivatedDeletedCachesOut );

	public:

		hkBool m_deleteInactiveCvxCaches;
		hkBool m_deleteInactiveMeshCaches;
};


#endif	//HKNP_COLLISION_CACHE_DEACTIVATION_POLICY_H

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
