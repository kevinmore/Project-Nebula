/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>
#include <Physics/Physics/Collide/NarrowPhase/Detector/hknpCollisionDetector.h>


HK_FORCE_INLINE void hknpCollisionCache::init( hknpCollisionCacheType::Enum type, int size )
{
	m_type = type;
	HK_ASSERT( 0xf2bbcc1a, (size & 0xf) == 0 );
	m_sizeInQuads = hkUint8(size>>4);
	m_linearTim = 0;
}

HK_FORCE_INLINE void hknpConvexConvexCollisionCache::getLeafShapeKeysImpl(
	HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyA, HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyB ) const
{
	*shapeKeyB = getShapeKey();
}


HK_FORCE_INLINE void hknpCollisionCache::destruct(
   const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
   hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
   hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
   hknpCdCacheDestructReason::Enum reason )
{
	hknpCollisionDetector* detector = tl.m_modifierManager->m_collisionDetectors[ m_type ];
	detector->destructCollisionCache( tl, sharedData, this, childCdCacheStream, childCdCacheStreamPpu, cdBodyA, cdBodyB, reason );
}

HK_FORCE_INLINE void hknpCollisionCache::moveAndConsumeChildCaches(
	const hknpSimulationThreadContext& tl,
	hknpCdCacheStream* childCdCacheStream,
	hknpCdCacheStream* childCdCacheStreamPpu,
	hknpCdCacheWriter* childCdCacheWriter )
{
	hknpCollisionDetector* detector = tl.m_modifierManager->m_collisionDetectors[ m_type ];
	if ( detector->m_useChildCaches )
	{
		hknpCompositeCollisionDetector* compDetector = static_cast<hknpCompositeCollisionDetector*>(detector);
		hknpCompositeCollisionCache* compCache = static_cast<hknpCompositeCollisionCache*>(this);
		compDetector->moveAndConsumeChildCaches( tl, compCache, childCdCacheStream, childCdCacheStreamPpu, childCdCacheWriter );
	}
}

HK_FORCE_INLINE void hknpConvexConvexCollisionCache::setQuality( const hknpSolverInfo& solverInfo,
	const hknpBody& bodyA, const hknpBodyQuality* qualityA, const hknpMaterial& materialA, const hknpShape& shapeA,
	const hknpBody& bodyB, const hknpBodyQuality* qualityB, const hknpMaterial& materialB, const hknpShape& shapeB )
{
	// Set quality flags
	{
		hknpBodyQuality::Flags qualityFlags;
		hknpBodyQuality::combineBodyQualities( qualityA, qualityB, &qualityFlags );

		const int materialFlags = materialA.m_flags.get() | materialB.m_flags.get();
		const int shapeFlags = shapeA.getFlags().get() | shapeB.getFlags().get();

		if( ( materialFlags & hknpMaterial::ENABLE_TRIGGER_VOLUME ) |
			( shapeFlags & hknpShape::USE_SINGLE_POINT_MANIFOLD ) )
		{
			qualityFlags.orWith( hknpBodyQuality::FORCE_GSK_SINGLE_POINT_MANIFOLD );
		}

		if( qualityFlags.anyIsSet( hknpBodyQuality::ENABLE_MOTION_WELDING | hknpBodyQuality::ENABLE_NEIGHBOR_WELDING ) )
		{
			qualityFlags.orWith( hknpBodyQuality::FORCE_GSK_EXECUTION );
		}

		m_qualityFlags.setAll( hkUint16(qualityFlags.get()) );
	}

	// Override TIM if we want to ignore initial collisions
	if( HK_VERY_UNLIKELY( bodyA.m_flags.get() & bodyB.m_flags.get() & hknpBody::TEMP_DROP_NEW_CVX_CVX_COLLISIONS ) )
	{
		hkSimdReal baseA; baseA.setFromUint16( bodyA.m_maxContactDistance );
		hkSimdReal baseB; baseB.setFromUint16( bodyB.m_maxContactDistance );
		hkSimdReal distA; distA.setFromHalf( materialA.m_disablingCollisionsBetweenCvxCvxDynamicObjectsDistance );
		hkSimdReal distB; distB.setFromHalf( materialB.m_disablingCollisionsBetweenCvxCvxDynamicObjectsDistance );

		hkSimdReal dist; dist.setMax( distA, distB );
		dist = dist * solverInfo.m_distanceToLinearTim + ( baseA + baseB );

		dist.storeSaturateUint16( &m_linearTim );
	}
}


#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE void hknpCollisionCache::moveChildCachesWithoutConsuming(
	const hknpSimulationThreadContext& tl, hknpCdCacheWriter* childCdCacheWriter )
{
	hknpCollisionDetector* detector = tl.m_modifierManager->m_collisionDetectors[ m_type ];
	if ( detector->m_useChildCaches )
	{
		hknpCompositeCollisionDetector* compDetector = static_cast<hknpCompositeCollisionDetector*>(detector);
		hknpCompositeCollisionCache* compCache = static_cast<hknpCompositeCollisionCache*>(this);
		compDetector->moveChildCachesWithoutConsuming( tl, compCache, childCdCacheWriter );
	}
}

#endif	// !HK_PLATFORM_SPU


/// Helper method to dispatch different types. 'Const' variant.
#define HKNP_DISPATCH_GET_CHILD_SHAPE_KEYS( CACHE, FUNC, PARAMS )									\
	if ( CACHE->m_type == hknpCollisionCacheType::CONVEX_CONVEX) { static_cast<const hknpConvexConvexCollisionCache*>(CACHE)->FUNC PARAMS;	}				\
	else if ( CACHE->m_type == hknpCollisionCacheType::SET_SHAPE_KEY_A)	{ static_cast<const hknpSetShapeKeyACollisionCache*>(CACHE)->FUNC PARAMS;	}


HK_FORCE_INLINE void hknpCollisionCache::getLeafShapeKeys(
	HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyA, HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyB ) const
{
	HKNP_DISPATCH_GET_CHILD_SHAPE_KEYS( this, getLeafShapeKeysImpl, ( shapeKeyA, shapeKeyB ) );
}


HK_FORCE_INLINE void hknpConvexCompositeCollisionCache::init()
{
	hknpCollisionCache::init( hknpCollisionCacheType::CONVEX_COMPOSITE, sizeof(hknpConvexCompositeCollisionCache) );
	m_childCdCacheRange.clearRange();
	resetNmpState();
}


HK_FORCE_INLINE void hknpConvexConvexCollisionCache::init()
{
	hkString::memSet16<sizeof( hknpConvexConvexCollisionCache)>( this, &hkVector4::getZero() );
	hknpCollisionCache::init( hknpCollisionCacheType::CONVEX_CONVEX, sizeof(*this) );
	m_gskCache.init();
	m_propertyOffsets = 0;
}

HK_FORCE_INLINE bool hknpConvexConvexCollisionCache::hasManifoldData() const
{
	return m_sizeInQuads > ( sizeof(hknpConvexConvexCollisionCache) >> 4 );
}

HK_FORCE_INLINE void hknpConvexConvexCollisionCache::_destructCdCacheImpl(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpCdCacheDestructReason::Enum reason
	) const
{
	if ( hasManifoldData() )
	{
		hknpManifoldCollisionCache* manifoldCache = (hknpManifoldCollisionCache*)const_cast<hknpConvexConvexCollisionCache*>(this);
		manifoldCache->fireManifoldDestroyed(tl, sharedData, cdBodyA, cdBodyB, reason);
	}
}


template<class T>
HK_FORCE_INLINE T* hknpConvexConvexCollisionCache::promoteTo( int sizeOfT )
{
	T* HK_RESTRICT self = (T*)this;
	hkString::memClear16(
		hkAddByteOffset( self, sizeof(hknpConvexConvexCollisionCache)),
		(sizeOfT + hknpConvexConvexCollisionCache::MAX_PROPERTY_BUFFER_SIZE - sizeof(hknpConvexConvexCollisionCache)) >> 4
		);
	self->m_maxImpulse.setMax();
	self->m_fractionOfClippedImpulseToApply = 255;
	self->m_propertyBufferOffset = hkUchar(sizeOfT);
	HK_COMPILE_TIME_ASSERT( sizeof(T) < 256 );
	return self;
}


HK_FORCE_INLINE void hknpManifoldCollisionCache::demoteToCvxCvxCache()
{
	m_sizeInQuads = sizeof(hknpConvexConvexCollisionCache) >> 4;
	m_propertyOffsets = 0;	// clear all properties
}

HK_FORCE_INLINE void hknpManifoldCollisionCache::_fireManifoldDestroyed(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpCdCacheDestructReason::Enum reason )
{
	HK_ASSERT( 0xf0456587, this->hasManifoldData() );
	hknpBodyFlags enabledModifiers = m_bodyAndMaterialFlags;
	if( tl.m_modifierManager->isFunctionRegistered( hknpModifier::FUNCTION_MANIFOLD_CREATED_OR_DESTROYED, enabledModifiers ) )
	{
		hknpManifoldCollisionCache* HK_RESTRICT manifoldCache = (hknpManifoldCollisionCache*)this;
		HK_ON_DEBUG( int propertyBufferSize = manifoldCache->getPropertyBufferSize() );
		HKNP_FIRE_MODIFIER( tl.m_modifierManager, hknpModifier::FUNCTION_MANIFOLD_CREATED_OR_DESTROYED, enabledModifiers,
			modifier->manifoldDestroyedCallback( tl, sharedData, cdBodyA, cdBodyB, manifoldCache, reason )
			);

		// check that the size hasn't changed
		HK_ASSERT2( 0xf0345456, propertyBufferSize == manifoldCache->getPropertyBufferSize(),
			"You can't add more properties after the manifold created callback." );
	}
}

HK_FORCE_INLINE void hknpCompositeCompositeCollisionCache::resetCdCacheImpl()
{
	resetNmpState();
	m_childCdCacheRange.clearRange();
}

HK_FORCE_INLINE void hknpCompositeCompositeCollisionCache::init()
{
	hknpCollisionCache::init( hknpCollisionCacheType::COMPOSITE_COMPOSITE, sizeof(hknpCompositeCompositeCollisionCache) );
	resetCdCacheImpl();
}

//
// Cache properties functions
//

HK_FORCE_INLINE hkUint8 hknpConvexConvexCollisionCache::getPropertyOffset( hkUint8 key ) const
{
	return ( m_propertyOffsets >> (key*4) ) & 0x0f;
}

HK_FORCE_INLINE void hknpConvexConvexCollisionCache::setPropertyOffset( hkUint8 key, hkUint8 offset )
{
	// Maximum addressable value is 64 bytes (MAX_PROPERTIES_BUFFER_SIZE)
	HK_COMPILE_TIME_ASSERT( 0x0f >= (MAX_PROPERTY_BUFFER_SIZE >>2) - 1);
	HK_ASSERT(0xb0db76c3, offset  <= (MAX_PROPERTY_BUFFER_SIZE>>2));
	hkUint32 data = m_propertyOffsets;
	data &= ~((0x0f) << (key*4));
	data |= ((offset) << (key*4));
	m_propertyOffsets = data;
}

// Helper functions for properties
namespace hknpCollisionCachePropertyUtils
{
	// Returns whether a given key is used or not.
	HK_FORCE_INLINE bool _isPropertyUsed( hkUint8 key, hkUint8 flags )
	{
		return flags & (1<<key);
	}
}

HK_FORCE_INLINE hkUint8 hknpManifoldCollisionCache::getPropertyBufferSize() const
{
	hkUint8 result = MAX_PROPERTY_BUFFER_SIZE;
	if( !(m_propertyKeysUsed & (1 << BUFFER_SIZE_PROPERTY)) )
	{
		// Buffer not full : get the next offset value
		result = getPropertyOffset(BUFFER_SIZE_PROPERTY) << 2;
	}

#ifdef HK_DEBUG
	{
		// Consistency check
		HK_ASSERT( 0xf68d6d98, result <= MAX_PROPERTY_BUFFER_SIZE );
		HK_ASSERT( 0xf68d6d99, result + m_propertyBufferOffset  <= HKNP_MAX_COLLISION_CACHE_SIZE );

		// Check for duplicates
		for( hkUint8 i = hkUint8(FIRST_PROPERTY_KEY) ; i < hkUint8(NUM_PROPERTY_KEYS) ; ++i )
		{
			for( hkUint8 j = i+1 ; j < hkUint8(NUM_PROPERTY_KEYS) ; ++j )
			{
				if( hknpCollisionCachePropertyUtils::_isPropertyUsed( i, m_propertyKeysUsed ) &&
					hknpCollisionCachePropertyUtils::_isPropertyUsed( j, m_propertyKeysUsed ) )
				{
					HK_ASSERT( 0x068d6d98, getPropertyOffset(i) != getPropertyOffset(j) );
				}
			}
		}
	}
#endif

	return result;
}

template < typename T >
HK_FORCE_INLINE T* hknpManifoldCollisionCache::accessProperty( PropertyKey key ) const
{
	HK_ASSERT2( 0x068d6d90, key >= FIRST_PROPERTY_KEY && key < NUM_PROPERTY_KEYS, "Invalid key" );

	T* HK_RESTRICT ptr = HK_NULL;
	if( hknpCollisionCachePropertyUtils::_isPropertyUsed( hkUint8(key), m_propertyKeysUsed ) )
	{
		ptr = reinterpret_cast<T* HK_RESTRICT>(
			hkAddByteOffset( (void*)this, m_propertyBufferOffset + (getPropertyOffset(hkUint8(key)) << 2) ) );

#ifdef HK_DEBUG
		// Sanity check to ensure that we're not overwriting data
		{
			hkUlong offset = getPropertyOffset(hkUint8(key)) << 2;
			HK_ASSERT( 0x068d6d92, offset < MAX_PROPERTY_BUFFER_SIZE );

			// Find next offset if it exists and check that we're not beyond our borders.
			hkUint8  nextOffset = MAX_PROPERTY_BUFFER_SIZE;
			for( hkUint8 i = hkUint8(FIRST_PROPERTY_KEY); i < hkUint8(NUM_PROPERTY_KEYS); ++i )
			{
				if( hknpCollisionCachePropertyUtils::_isPropertyUsed(i, m_propertyKeysUsed) )
				{
					hkUint8 currentOffset = getPropertyOffset(i) << 2;
					if( currentOffset > offset && currentOffset < nextOffset )
					{
						nextOffset = currentOffset;
					}
				}
			}

			// If we didn't find anything at this point, that means our slot is the last allocated slot.
			// We can find the next offset value in the first slot if the buffer isn't full.
			if( nextOffset == MAX_PROPERTY_BUFFER_SIZE &&
				!hknpCollisionCachePropertyUtils::_isPropertyUsed(BUFFER_SIZE_PROPERTY, m_propertyKeysUsed) )
			{
				nextOffset = getPropertyOffset(BUFFER_SIZE_PROPERTY) << 2;
			}

			// Check if we're not overwriting.
			HK_ASSERT2(0x068d6d93, offset + sizeof(T) <= nextOffset, "Potentially overwriting the next property data." \
				" This could be a wrong type passed as template parameter.");
		}
#endif
	}

	return ptr;
}

template < typename T >
HK_FORCE_INLINE T* hknpManifoldCollisionCache::allocateProperty( PropertyKey key, hkUint32 alignment )
{
	HK_ASSERT2( 0x068d6d90, key >= FIRST_PROPERTY_KEY && key < NUM_PROPERTY_KEYS, "Invalid key." );

	hkUint32 size = sizeof(T);
	HK_ASSERT2( 0xb0db76c5, size < MAX_PROPERTY_BUFFER_SIZE, "Size too big to fit." );
	HK_ASSERT2( 0xb0db76c2, alignment >= 4, "Alignment must be at least 4 bytes." );
	HK_ASSERT2( 0xb0db76c3, hkMath::isPower2(alignment), "Alignment should be a power of 2." );
	HK_ASSERT2( 0xb0db76c3, alignment < MAX_PROPERTY_BUFFER_SIZE / 2, "Alignment too big to fit." );

	if( hknpCollisionCachePropertyUtils::_isPropertyUsed( hkUint8(key), m_propertyKeysUsed ) )
	{
		HK_ASSERT2( 0xb0db76c7, false, "Attempting to allocate a property key twice." );
		return HK_NULL;
	}

	// Size is rounded up to the next dword (4-bytes).
	const hkUint8 sizeInBytes = HK_NEXT_MULTIPLE_OF( 1 << 2, hkUint8(size) );

	// Calculate offset from start of properties buffer
	hkUint8 offsetInBytes;
	{
		offsetInBytes = getPropertyOffset( BUFFER_SIZE_PROPERTY ) << 2;

		// Now we align the offset as requested.
		offsetInBytes = HK_NEXT_MULTIPLE_OF( hkUint8(alignment), offsetInBytes );

#ifdef HK_DEBUG
		HK_ASSERT( 0xf0dcde34, m_propertyBufferOffset != 0 );
		const char* propertyBuffer = (const char*)( hkAddByteOffset( (void*)this, m_propertyBufferOffset ) );
#if !defined(HK_ALIGN_RELAX_CHECKS)
		HK_ASSERT2( 0xb0db76c7, (hkUlong(propertyBuffer + (offsetInBytes)) & (alignment -1)) == 0, "Incorrect data alignment." );
#endif
#endif
	}

	if( offsetInBytes + sizeInBytes > MAX_PROPERTY_BUFFER_SIZE )
	{
		HK_ASSERT2( 0xb0db76c6, false, "Property buffer is full." );
		return HK_NULL;
	}

	// Allocate the property
	m_propertyKeysUsed |= (1 << key);
	setPropertyOffset( hkUint8(key), (offsetInBytes) >> 2 );

	// Update the buffer size
	if( offsetInBytes + sizeInBytes < MAX_PROPERTY_BUFFER_SIZE )
	{
		setPropertyOffset( BUFFER_SIZE_PROPERTY, (offsetInBytes + sizeInBytes) >> 2 );
	}
	else
	{
		// Mark the buffer as full.
		m_propertyKeysUsed |= (1 << BUFFER_SIZE_PROPERTY);
	}

	// Update the cache's total size.
	{
		hkUlong totalSize = m_propertyBufferOffset + offsetInBytes + sizeInBytes;
		HK_ASSERT( 0xf6d6d9c, hkUlong(m_propertyBufferOffset) + getPropertyBufferSize() == totalSize );
		// Round size up to 16-bytes limit
		totalSize = HK_NEXT_MULTIPLE_OF( 1<<4, totalSize );
		HK_ASSERT( 0xf68d6d9a, totalSize <= HKNP_MAX_COLLISION_CACHE_SIZE );
		// Check that total size in quads fits in a single byte
		HK_ASSERT( 0xf68d6d9b, totalSize >>4 <= 0xff );
		m_sizeInQuads = HK_NEXT_MULTIPLE_OF(1<<4, hkUint8(totalSize)) >> 4;
	}

	// Return a pointer to the property
	return reinterpret_cast<T*>( hkAddByteOffset( (void*)this, m_propertyBufferOffset + offsetInBytes ) );
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
