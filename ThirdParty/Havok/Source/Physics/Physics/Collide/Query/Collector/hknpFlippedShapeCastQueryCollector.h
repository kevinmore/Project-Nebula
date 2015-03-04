/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_FLIPPED_SHAPE_CAST_QUERY_COLLECTOR_H
#define HKNP_FLIPPED_SHAPE_CAST_QUERY_COLLECTOR_H

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Physics/Physics/Collide/Query/Collector/hknpCollisionQueryCollector.h>


// This virtual collector handles the flipping back of the query result before forwarding the hit to the original
// collector.
class hknpFlippedShapeCastQueryCollector : public hknpCollisionQueryCollector
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpFlippedShapeCastQueryCollector );

		/// Constructor.
		HK_FORCE_INLINE hknpFlippedShapeCastQueryCollector(
			hkVector4Parameter castDirectionWS,
			hknpCollisionQueryCollector* childCollector );

		//
		// hknpCollisionQueryCollector implementation
		//

		HK_FORCE_INLINE virtual void reset();
		HK_FORCE_INLINE virtual bool hasHit() const;
		HK_FORCE_INLINE virtual int getNumHits() const;
		HK_FORCE_INLINE virtual const hknpCollisionResult* getHits() const;
		HK_FORCE_INLINE virtual void addHit( const hknpCollisionResult& flippedHit );

	protected:

		hkVector4						m_castDirectionWS;
		hknpCollisionQueryCollector*	m_childCollector;
};

#include <Physics/Physics/Collide/Query/Collector/hknpFlippedShapeCastQueryCollector.inl>


#endif // HKNP_FLIPPED_SHAPE_CAST_QUERY_COLLECTOR_H

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
