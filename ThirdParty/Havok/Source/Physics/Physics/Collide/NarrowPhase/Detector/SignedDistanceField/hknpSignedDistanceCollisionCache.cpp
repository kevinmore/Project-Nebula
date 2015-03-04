/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceCollisionCache.h>

HK_COMPILE_TIME_ASSERT( ( sizeof(hknpConvexCompositeCollisionCache) & (HK_REAL_ALIGNMENT-1) ) ==  0 );


void HK_CALL hknpSignedDistanceFieldCollisionCache::construct(
	const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
	hknpCdCacheWriter* cacheWriterInOut )
{
	hknpCollisionCache* HK_RESTRICT newCache = cacheWriterInOut->reserve( sizeof(hknpSignedDistanceFieldCollisionCache) );
	hknpSignedDistanceFieldCollisionCache* HK_RESTRICT newCvxSdfCache = static_cast<hknpSignedDistanceFieldCollisionCache*>( newCache );
	hkString::memClear16(newCvxSdfCache, sizeof(hknpSignedDistanceFieldCollisionCache)>>4);

	newCvxSdfCache->init(hknpCollisionCacheType::DISTANCE_FIELD, sizeof(hknpSignedDistanceFieldCollisionCache));
	newCvxSdfCache->m_bodyA = bodyA.m_id;
	newCvxSdfCache->m_bodyB = bodyB.m_id;

#if defined(HK_PLATFORM_HAS_SPU)

	// At this stage we can be sure that the shape with hknpCollisionDispatchType::DISTANCE_FIELD is in slot B, so we
	// only need to check if the shape in slot A supports getAllShapeKeys() on SPU or not.

#if defined( HK_DEBUG ) && !defined( HK_PLATFORM_SPU )

	// Warn the user about incorrectly set SPU flags on compounds.
	if ( ( bodyA.m_shape->getType() == hknpShapeType::STATIC_COMPOUND || bodyA.m_shape->getType() == hknpShapeType::DYNAMIC_COMPOUND ) )
	{
		const hknpCompoundShape* compound = (const hknpCompoundShape*)bodyA.m_shape;
		if ( !compound->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU ) ) { HK_ASSERT2( 0xaf1e3242, (compound->propagateSpuFlags() & hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU) == 0, "This compound shape contains a child shape that does not support getAllShapeKeys() on SPU. Make sure to set the root compound shape's hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU flag accordingly. See hknpCompoundShape::computeSpuFlags() for a utility method to do this." ); }
		if ( !compound->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU   ) ) { HK_ASSERT2( 0xaf1e3542, (compound->propagateSpuFlags() & hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU)   == 0, "This compound shape contains a child shape that is not supported on SPU. Make sure to set the root compound shape's hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU flag accordingly. See hknpCompoundShape::computeSpuFlags() for a utility method to do this." ); }
	}

#endif

	hkUint8 additionalSpuFlag = 0;

#if defined( HKNP_DISABLE_SIGNED_DISTANCE_FIELD_COLLISIONS_ON_SPU )

	additionalSpuFlag = hknpBody::FORCE_NARROW_PHASE_PPU;

#else

	bool shapeAFitForSlotA = !bodyA.m_shape->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU );
	bool unsupportedShape = bodyA.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) | bodyB.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU );
	if ( !shapeAFitForSlotA || unsupportedShape )
	{
		additionalSpuFlag = hknpBody::FORCE_NARROW_PHASE_PPU;
	}

#endif

	newCvxSdfCache->m_spuFlags = bodyA.m_spuFlags.get() | bodyB.m_spuFlags.get() | additionalSpuFlag;

#endif

	newCvxSdfCache->m_childCdCacheRange.clearRange();

	cacheWriterInOut->advance( sizeof(hknpSignedDistanceFieldCollisionCache) );
}

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
