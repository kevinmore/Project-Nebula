/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_ANY_HIT_COLLISION_QUERY_COLLECTOR_H
#define HKNP_ANY_HIT_COLLISION_QUERY_COLLECTOR_H

#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>


/// A collision query collector that exits on the first hit it finds. The stored hit is not necessarily the closest hit possible.
/// This should be used for visibility checks where you only care about whether any hit occurs, not where it occurs.
class hknpAnyHitCollector : public hknpClosestHitCollector
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpAnyHitCollector );

		/// Constructor.
		HK_FORCE_INLINE hknpAnyHitCollector()
		{
			m_hints |= HINT_STOP_AT_FIRST_HIT;
		}

		// hknpCollisionQueryCollector implementation.
		virtual void addHit( const hknpCollisionResult& hit ) HK_OVERRIDE;
};


#endif // HKNP_ANY_HIT_COLLISION_QUERY_COLLECTOR_H

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
