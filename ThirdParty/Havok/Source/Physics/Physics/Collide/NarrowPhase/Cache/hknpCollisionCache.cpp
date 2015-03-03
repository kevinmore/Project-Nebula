/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSignedDistanceCollisionCache.h>


// for memCpy16 etc
HK_COMPILE_TIME_ASSERT( (sizeof(hknpConvexConvexCollisionCache)         & (HK_REAL_ALIGNMENT-1)) ==  0 );
HK_COMPILE_TIME_ASSERT( (sizeof(hknpManifoldCollisionCache)             & (HK_REAL_ALIGNMENT-1)) ==  0 );
HK_COMPILE_TIME_ASSERT( (sizeof(hknpConvexConvexManifoldCollisionCache) & (HK_REAL_ALIGNMENT-1)) ==  0 );
HK_COMPILE_TIME_ASSERT( (sizeof(hknpCompositeCollisionCache)            & (HK_REAL_ALIGNMENT-1)) ==  0 );
HK_COMPILE_TIME_ASSERT( (sizeof(hknpConvexCompositeCollisionCache)      & (HK_REAL_ALIGNMENT-1)) ==  0 );
HK_COMPILE_TIME_ASSERT( (sizeof(hknpCompositeCompositeCollisionCache)   & (HK_REAL_ALIGNMENT-1)) ==  0 );
HK_COMPILE_TIME_ASSERT( (sizeof(hknpSetShapeKeyACollisionCache)         & (HK_REAL_ALIGNMENT-1)) ==  0 );

HK_COMPILE_TIME_ASSERT( (sizeof(hknpConvexConvexManifoldCollisionCache) ) >= 64 );
#if !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT( (sizeof(hknpConvexConvexManifoldCollisionCache) ) <= 160);
#endif

HK_COMPILE_TIME_ASSERT( 2*sizeof(hknpBodyId) >= sizeof(hknpShapeKey) );	// make sure a shape key fits in two body IDs


HK_FORCE_INLINE void hknpConvexConvexCollisionCache::resetCdCacheImpl()
{
	m_sizeInQuads = (sizeof(hknpConvexConvexCollisionCache)>>4);
	m_gskCache.init();	// reset GSK cache
}


HK_FORCE_INLINE void hknpConvexCompositeCollisionCache::resetCdCacheImpl()
{
	m_childCdCacheRange.clearRange();
	resetNmpState();
}


#if !defined(HK_PLATFORM_SPU)

void HK_CALL hknpConvexConvexCollisionCache::construct(
	const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
	hknpCdCacheWriter* cacheWriterInOut )
{
#if !defined( HKNP_DISABLE_SIGNED_DISTANCE_FIELD_COLLISIONS_ON_SPU )
	// Special case capsule-vs-capsule as capsule-vs-SDF
	if( bodyA.m_shape->getType() == hknpShapeType::CAPSULE && bodyB.m_shape->getType() == hknpShapeType::CAPSULE )
	{
		if( bodyA.m_shape->m_convexRadius < bodyB.m_shape->m_convexRadius )
		{
			hknpSignedDistanceFieldCollisionCache::construct( world, bodyA, bodyB, cacheWriterInOut );
		}
		else
		{
			hknpSignedDistanceFieldCollisionCache::construct( world, bodyB, bodyA, cacheWriterInOut );
		}
		return;
	}
#endif

	const hknpBodyQuality* qualityA = &world.getBodyQualityLibrary()->getEntry( bodyA.m_qualityId );
	const hknpBodyQuality* qualityB = &world.getBodyQualityLibrary()->getEntry( bodyB.m_qualityId );
	const hknpMaterial* materialA = &world.getMaterialLibrary()->getEntry( bodyA.m_materialId );
	const hknpMaterial* materialB = &world.getMaterialLibrary()->getEntry( bodyB.m_materialId );

	// Allocate the cache
	hknpCollisionCache* HK_RESTRICT newCache = cacheWriterInOut->reserve( sizeof(hknpConvexConvexCollisionCache) );
	hknpConvexConvexCollisionCache* HK_RESTRICT newCvxCvxCache = static_cast<hknpConvexConvexCollisionCache*>( newCache );

	// Initialize it
	newCvxCvxCache->init();
	newCvxCvxCache->m_bodyA = bodyA.m_id;
	newCvxCvxCache->m_bodyB = bodyB.m_id;

#if defined(HK_PLATFORM_HAS_SPU)
	bool unsupportedShape = bodyA.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) | bodyB.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU );
	hkUint8 additionalSpuFlag = 0;
	if ( unsupportedShape )
	{
		additionalSpuFlag = hknpBody::FORCE_NARROW_PHASE_PPU;
	}
	newCvxCvxCache->m_spuFlags = bodyA.m_spuFlags.get() | bodyB.m_spuFlags.get() | additionalSpuFlag;
#endif

	newCvxCvxCache->setQuality( world.m_solverInfo,
		bodyA, qualityA, *materialA, *bodyA.m_shape,
		bodyB, qualityB, *materialB, *bodyB.m_shape
	);

	cacheWriterInOut->advance( sizeof(hknpConvexConvexCollisionCache) );
}


void HK_CALL hknpConvexCompositeCollisionCache::construct(
	const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
	hknpCdCacheWriter* cacheWriterInOut )
{
	// Allocate the cache
	hknpCollisionCache* newCache = cacheWriterInOut->reserve( sizeof(hknpConvexCompositeCollisionCache) );

	hknpConvexCompositeCollisionCache* HK_RESTRICT newCcCache = static_cast<hknpConvexCompositeCollisionCache*>( newCache );
	hkString::memClear16(newCcCache, sizeof(hknpConvexCompositeCollisionCache)>>4);

	// Initialize it
	newCcCache->init();
	newCcCache->m_bodyA = bodyA.m_id;
	newCcCache->m_bodyB = bodyB.m_id;

#if defined(HK_PLATFORM_HAS_SPU)

#if defined( HK_DEBUG )
	// Warn the user about incorrectly set SPU flags on compounds.
	if( bodyB.m_shape->getType() == hknpShapeType::STATIC_COMPOUND ||
		bodyB.m_shape->getType() == hknpShapeType::DYNAMIC_COMPOUND )
	{
		const hknpCompoundShape* compound = static_cast<const hknpCompoundShape*>( bodyB.m_shape );
		if( !compound->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU ) )
		{
			HK_ASSERT2( 0xaf1e3242, (compound->propagateSpuFlags() & hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU) == 0,
				"This compound shape contains a child shape that does not support getAllShapeKeys() on SPU. "
				"Make sure to set the root compound shape's hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU flag accordingly. "
				"See hknpCompoundShape::computeSpuFlags() for a utility method to do this." );
		}
		if( !compound->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) )
		{
			HK_ASSERT2( 0xaf1e3542, (compound->propagateSpuFlags() & hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU) == 0,
				"This compound shape contains a child shape that is not supported on SPU. "
				"Make sure to set the root compound shape's hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU flag accordingly. "
				"See hknpCompoundShape::computeSpuFlags() for a utility method to do this." );
		}
	}
#endif

	hkUint8 additionalSpuFlags = 0;
	if( bodyA.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) |
		bodyB.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) )
	{
		additionalSpuFlags = hknpBody::FORCE_NARROW_PHASE_PPU;
	}
	newCcCache->m_spuFlags = bodyA.m_spuFlags.get() | bodyB.m_spuFlags.get() | additionalSpuFlags;

#endif

	cacheWriterInOut->advance( sizeof(hknpConvexCompositeCollisionCache) );
}


HK_FORCE_INLINE bool hknpCompositeCompositeCollisionCache_checkForBodySwap( const hknpBody& bodyA, const hknpBody& bodyB )
{
	// If bodyA is static, interchange bodyA and bodyB to avoid enumerating all shape keys in the static body
	// (assumed to be a landscape or similar).
	

	hkBool swap = false;

	if( bodyA.isStatic() && bodyB.isDynamic() )
	{
		swap = true;
	}
	else if( bodyA.isDynamic() && bodyB.isDynamic() )
	{
		const hknpShape* compShapeA = bodyA.m_shape;
		const hknpShape* compShapeB = bodyB.m_shape;
		const hknpShapeType::Enum compShapeAType = compShapeA->getType();
		const hknpShapeType::Enum compShapeBType = compShapeB->getType();
		const bool isCompoundA = (compShapeAType == hknpShapeType::STATIC_COMPOUND) || (compShapeAType == hknpShapeType::DYNAMIC_COMPOUND);
		const bool isCompoundB = (compShapeBType == hknpShapeType::STATIC_COMPOUND) || (compShapeBType == hknpShapeType::DYNAMIC_COMPOUND);
		if( isCompoundA && isCompoundB )
		{
			swap = compShapeA->getNumShapeKeyBits() <= compShapeB->getNumShapeKeyBits();
		}
		else
		{
			/*
			if( !isCompoundA && !isCompoundB )
			{
				HK_WARN_ALWAYS( 0xf0cf5676, "Cannot handle mesh mesh collisions: ignoring" );
			}
			*/

			swap = !isCompoundA;
		}
	}

	return swap;
}

void HK_CALL hknpCompositeCompositeCollisionCache::construct(
	const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
	hknpCdCacheWriter* cacheWriterInOut )
{
	// Allocate the cache
	hknpCollisionCache* newCache = cacheWriterInOut->reserve( sizeof(hknpCompositeCompositeCollisionCache) );

	hknpCompositeCompositeCollisionCache* HK_RESTRICT newCcCache = static_cast<hknpCompositeCompositeCollisionCache*>( newCache );
	hkString::memClear16(newCcCache, sizeof(hknpCompositeCompositeCollisionCache)>>4);

	// Initialize it
	newCcCache->init();

#if !defined(HK_PLATFORM_HAS_SPU)

	hkBool swap = hknpCompositeCompositeCollisionCache_checkForBodySwap( bodyA, bodyB );

	if( swap )
	{
		newCcCache->m_bodyA = bodyB.m_id;
		newCcCache->m_bodyB = bodyA.m_id;
	}
	else
	{
		newCcCache->m_bodyA = bodyA.m_id;
		newCcCache->m_bodyB = bodyB.m_id;
	}
	newCcCache->m_spuFlags = bodyA.m_spuFlags.get() | bodyB.m_spuFlags.get();

#else

#if defined( HK_DEBUG )

	// Warn the user about incorrectly set SPU flags on compounds.
	if ( ( bodyA.m_shape->getType() == hknpShapeType::STATIC_COMPOUND || bodyA.m_shape->getType() == hknpShapeType::DYNAMIC_COMPOUND ) )
	{
		const hknpCompoundShape* compound = (const hknpCompoundShape*)bodyA.m_shape;
		if ( !compound->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU ) ) { HK_ASSERT2( 0xaf1e3242, (compound->propagateSpuFlags() & hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU) == 0, "This compound shape contains a child shape that does not support getAllShapeKeys() on SPU. Make sure to set the root compound shape's hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU flag accordingly. See hknpCompoundShape::computeSpuFlags() for a utility method to do this." ); }
		if ( !compound->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU   ) ) { HK_ASSERT2( 0xaf1e3542, (compound->propagateSpuFlags() & hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU)   == 0, "This compound shape contains a child shape that is not supported on SPU. Make sure to set the root compound shape's hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU flag accordingly. See hknpCompoundShape::computeSpuFlags() for a utility method to do this." ); }
	}

	// Warn the user about incorrectly set SPU flags on compounds.
	if ( ( bodyB.m_shape->getType() == hknpShapeType::STATIC_COMPOUND || bodyB.m_shape->getType() == hknpShapeType::DYNAMIC_COMPOUND ) )
	{
		const hknpCompoundShape* compound = (const hknpCompoundShape*)bodyB.m_shape;
		if ( !compound->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU ) ) { HK_ASSERT2( 0xaf1e3242, (compound->propagateSpuFlags() & hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU) == 0, "This compound shape contains a child shape that does not support getAllShapeKeys() on SPU. Make sure to set the root compound shape's hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU flag accordingly. See hknpCompoundShape::computeSpuFlags() for a utility method to do this." ); }
		if ( !compound->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU   ) ) { HK_ASSERT2( 0xaf1e3542, (compound->propagateSpuFlags() & hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU)   == 0, "This compound shape contains a child shape that is not supported on SPU. Make sure to set the root compound shape's hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU flag accordingly. See hknpCompoundShape::computeSpuFlags() for a utility method to do this." ); }
	}

#endif

	// The 'composite vs composite' collision detector calls getAllShapeKeys() on the shape in slot A. As some shapes
	// (e.g. hknpCompressedMeshShape) do not implement getAllShapeKeys() on SPU we need to make sure that these either
	// end up in slot B or on PPU (if both shapes are of the unsupported kind.)

	bool shapeAFitForSlotA = !bodyA.m_shape->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU );
	bool shapeBFitForSlotA = !bodyB.m_shape->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU );

	// Should any of the 2 shapes not be available on SPU at all then we need to force the pair onto PPU.

	bool unsupportedShape = bodyA.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) | bodyB.m_shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU );

	hkBool swap = false;
	hkUint8 additionalSpuFlag = 0;

	if ( unsupportedShape )
	{
		additionalSpuFlag = hknpBody::FORCE_NARROW_PHASE_PPU;
	}
	else
	{
		if ( !shapeAFitForSlotA )
		{
			if ( !shapeBFitForSlotA )
			{
				// Colliding 2 shapes that both do not support getAllShapeKeys() on SPU needs to be forced to PPU.
				additionalSpuFlag = hknpBody::FORCE_NARROW_PHASE_PPU;
			}
			else
			{
				// Shape in slot A does not support getAllShapeKeys() on SPU, so we need to move this to slot B.
				swap = true;
			}
		}
	}

	if ( ( shapeAFitForSlotA && shapeBFitForSlotA ) || additionalSpuFlag == hknpBody::FORCE_NARROW_PHASE_PPU )
	{
		swap = hknpCompositeCompositeCollisionCache_checkForBodySwap( bodyA, bodyB );
	}

	if( swap )
	{
		newCcCache->m_bodyA = bodyB.m_id;
		newCcCache->m_bodyB = bodyA.m_id;
	}
	else
	{
		newCcCache->m_bodyA = bodyA.m_id;
		newCcCache->m_bodyB = bodyB.m_id;
	}

	newCcCache->m_spuFlags = bodyA.m_spuFlags.get() | bodyB.m_spuFlags.get() | additionalSpuFlag;

#endif

	cacheWriterInOut->advance( sizeof(hknpCompositeCompositeCollisionCache) );
}

#endif	// !HK_PLATFORM_SPU



void hknpManifoldCollisionCache::setFrictionAndRestitution(const hknpMaterial& x, const hknpMaterial& y)
{
	hknpManifoldCollisionCache* HK_RESTRICT output = this;

	// Friction

	hkSimdReal staticFriction;
	hkSimdReal dynamicFriction;

	{
		{
			hkSimdReal sA; sA.load<1>( &x.m_staticFriction );
			hkSimdReal sB; sB.load<1>( &y.m_staticFriction );
			staticFriction.setMax( sA, sB );
		}

		hknpMaterial::CombinePolicy policy = hkMath::max2( x.m_frictionCombinePolicy, y.m_frictionCombinePolicy );

		{
			hkSimdReal a; a.load<1>( &x.m_dynamicFriction );
			hkSimdReal b; b.load<1>( &y.m_dynamicFriction );
			if ( policy == hknpMaterial::COMBINE_AVG)
			{
				dynamicFriction = (a*b).sqrt<HK_ACC_12_BIT, HK_SQRT_SET_ZERO>();
			}
			else if ( policy == hknpMaterial::COMBINE_MIN)
			{
				dynamicFriction.setMin( a, b );
			}
			else
			{
				dynamicFriction.setMax( a, b );
			}
		}
		staticFriction = staticFriction - dynamicFriction;
		staticFriction.setMax( staticFriction, hkSimdReal_0 );
	}

	// Restitution
	hkSimdReal restitution;
	{
		hknpMaterial::CombinePolicy policy = hkMath::max2( x.m_restitutionCombinePolicy, y.m_restitutionCombinePolicy );
		hkSimdReal a; a.load<1>( &x.m_restitution );
		hkSimdReal b; b.load<1>( &y.m_restitution );
		if ( policy == hknpMaterial::COMBINE_AVG)
		{
			restitution = (a*b).sqrt<HK_ACC_12_BIT, HK_SQRT_SET_ZERO>();
		}
		else if ( policy == hknpMaterial::COMBINE_MIN)
		{
			restitution.setMin( a, b );
		}
		else
		{
			restitution.setMax( a, b );
		}
	}
	// store all values at the end
	restitution.store<1>( & output->m_restitution );
	dynamicFriction.store<1>( &output->m_friction );
	staticFriction.store<1>(  &output->m_staticFrictionExtra );
}


void hknpManifoldCollisionCache::fireManifoldDestroyed(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpCdCacheDestructReason::Enum reason
	)
{
	_fireManifoldDestroyed( tl, sharedData, cdBodyA, cdBodyB, reason);
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
