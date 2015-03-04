/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>


#if !defined(HK_PLATFORM_SPU)

hknpMaskedCompositeShape::hknpMaskedCompositeShape( const hknpCompositeShape* shape )
	: m_shape(shape)
{
	m_numShapeKeyBits = 0;
	m_mask = m_shape->createShapeKeyMask();
	HK_ASSERT2( 0x3f1ca0c1, m_mask, "The supplied shape does not support key masking." );
	m_maskWrapper.m_maskedShape = this;

#if !defined(HK_PLATFORM_HAS_SPU) || defined(HK_PLATFORM_SPU)
	m_childShapeSize = 0;
	m_maskSize = 0;
#else
	m_childShapeSize = shape->calcSize();
	HK_ASSERT2( 0xaf4431f4, m_childShapeSize <= HKNP_MAX_SHAPE_SIZE_ON_SPU - sizeof(hknpMaskedCompositeShape), "The size of the to-be-masked child shape exceeds the allowed size on SPU." );
	m_maskSize = m_mask->calcSize();
	HK_ASSERT2( 0xaf4431f1, m_maskSize <= HKNP_MAX_SHAPE_KEY_MASK_SIZE_ON_SPU, "The size of the shape key mask exceeds the allowed size on SPU." );
#endif
}

hknpMaskedCompositeShape::~hknpMaskedCompositeShape()
{
	m_mutationSignals.m_shapeDestroyed.fire();
	if( m_mask )
	{
		delete m_mask;
		m_mask = HK_NULL;
	}
}

#else

HK_NEVER_INLINE const hknpCompositeShape* hknpMaskedCompositeShape::getChildShape() const
{
	// Fetch child shape from PPU if it's not yet on SPU. The child shape will be uploaded to the same overall shape
	// buffer right behind this hknpMaskedCompositeShape. If m_childShapeSize is 0 then the child shape is already
	// available on SPU (either directly following this hknpMaskedCompositeShape shape in SPU memory if it has been
	// uploaded from PPU or somewhere else in SPU memory if it has been created locally on SPU.)
	if ( m_childShapeSize > 0 )
	{
		hknpCompositeShape* childShapeOnSpu = const_cast<hknpCompositeShape*>( hkAddByteOffsetConst( reinterpret_cast<const hknpCompositeShape*>(this), sizeof(*this) ) );
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( childShapeOnSpu, m_shape, m_childShapeSize, hkSpuDmaManager::READ_COPY );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS( m_shape, childShapeOnSpu, m_childShapeSize );
		hknpShapeVirtualTableUtil::patchVirtualTable( childShapeOnSpu );
		m_shape = childShapeOnSpu;
		m_childShapeSize = 0;
	}
	return m_shape;
}

#endif

void hknpMaskedCompositeShape::calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const
{
	getChildShape()->calcAabb( transform, aabbOut );
}

int hknpMaskedCompositeShape::calcSize() const
{
	return sizeof(hknpMaskedCompositeShape);
}

#if !defined(HK_PLATFORM_SPU)

hkRefNew<hknpShapeKeyIterator> hknpMaskedCompositeShape::createShapeKeyIterator( const hknpShapeKeyMask* mask ) const
{
	HK_ASSERT( 0x745411f8, mask == HK_NULL );
	return getChildShape()->createShapeKeyIterator( m_mask );
}

#else

hknpShapeKeyIterator* hknpMaskedCompositeShape::createShapeKeyIterator( hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask ) const
{
	HK_ASSERT( 0x745411f8, mask == HK_NULL );
	return getChildShape()->createShapeKeyIterator( buffer, bufferSize, m_mask );
}

#endif

void hknpMaskedCompositeShape::getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const
{
#if !defined( HK_PLATFORM_SPU )
	HK_WARN_ON_DEBUG_IF( !m_mask->isShapeKeyEnabled(key), 0x745411f8,
		"Calling hknpMaskedCompositeShape::getLeafShape() with a disabled key" );
#endif

	getChildShape()->getLeafShape( key, collector );
}

void hknpMaskedCompositeShape::queryAabbImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut) const
{
	hknpShapeQueryInfo maskedTargetShapeInfo( &targetShapeInfo );
#if defined( HK_PLATFORM_SPU )
	HK_ALIGN16( hkUint8 maskBuffer[HKNP_MAX_SHAPE_KEY_MASK_SIZE_ON_SPU] );
	hknpShapeKeyMask* maskOnSpu = (hknpShapeKeyMask*)&maskBuffer[0];
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( maskOnSpu, m_mask, m_maskSize, hkSpuDmaManager::READ_COPY );
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( m_mask, maskOnSpu, m_maskSize );
	maskedTargetShapeInfo.m_shapeKeyMask = maskOnSpu;
#else
	maskedTargetShapeInfo.m_shapeKeyMask = m_mask;
#endif

	getChildShape()->queryAabbImpl(
		queryContext, query, queryShapeInfo,
		targetShapeFilterData, maskedTargetShapeInfo,
		hits, nmpInOut );

#if defined( HK_DEBUG ) && !defined( HK_PLATFORM_SPU )
	// Check that we didn't get any disabled keys
	for( int i=0, n=hits->getSize(); i<n; ++i )
	{
		HK_ASSERT( 0x745411f9, m_mask->isShapeKeyEnabled((*hits)[i]) );
	}
#endif
}

void hknpMaskedCompositeShape::queryAabbImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut) const
{
	hknpShapeQueryInfo maskedTargetShapeInfo( &targetShapeInfo );
	maskedTargetShapeInfo.m_shapeKeyMask = m_mask;
	getChildShape()->queryAabbImpl(
		queryContext, query, queryShapeInfo,
		targetShapeFilterData, maskedTargetShapeInfo,
		collector, nmpInOut );
}

void hknpMaskedCompositeShape::castRayImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpRayCastQuery& query,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector ) const
{
	hknpShapeQueryInfo maskedTargetShapeInfo( &targetShapeInfo );
	maskedTargetShapeInfo.m_shapeKeyMask = m_mask;
	getChildShape()->castRayImpl( queryContext, query, targetShapeFilterData, maskedTargetShapeInfo, collector );
}

#if !defined(HK_PLATFORM_SPU)
void hknpMaskedCompositeShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,	hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#else
void hknpMaskedCompositeShape::getAllShapeKeys(	const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,	hkUint8* shapeBuffer, int shapeBufferSize, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#endif
{
	HK_ASSERT( 0x44032137, mask == HK_NULL );

#if !defined(HK_PLATFORM_SPU)
	getChildShape()->getAllShapeKeys( shapeKeyPath, m_mask, keyPathsOut );
#else
	HK_ALIGN16( hkUint8 maskBuffer[HKNP_MAX_SHAPE_KEY_MASK_SIZE_ON_SPU] );
	hknpShapeKeyMask* maskOnSpu = (hknpShapeKeyMask*)&maskBuffer[0];
	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( maskOnSpu, m_mask, m_maskSize, hkSpuDmaManager::READ_COPY );
	HK_SPU_DMA_PERFORM_FINAL_CHECKS( m_mask, maskOnSpu, m_maskSize );

	const hknpCompositeShape* childShape = getChildShape();
	childShape->patchShapeKeyMaskVTable( maskOnSpu );
	childShape->getAllShapeKeys( shapeKeyPath, maskOnSpu, shapeBuffer, shapeBufferSize, keyPathsOut );
#endif

#if defined(HK_DEBUG) && !defined(HK_PLATFORM_SPU)
	// Check that we didn't get any disabled keys
	for( int i=0, n=keyPathsOut->getSize(); i<n; ++i )
	{
		HK_ASSERT( 0x745411fa, m_mask->isShapeKeyEnabled( (*keyPathsOut)[i].getKey() ) );
	}
#endif
}


#if !defined(HK_PLATFORM_SPU)

hknpShape::MutationSignals* hknpMaskedCompositeShape::getMutationSignals()
{
	return &m_mutationSignals;
}

hkResult hknpMaskedCompositeShape::buildSurfaceGeometry(
	const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut) const
{
	
	return getChildShape()->buildSurfaceGeometry( config, geometryOut );
}

#if !defined( HK_PLATFORM_SPU )

void hknpMaskedCompositeShape::MaskWrapper::setShapeKeyEnabled( hknpShapeKey key, bool isEnabled )
{
	m_maskedShape->m_mask->setShapeKeyEnabled( key, isEnabled );
}

bool hknpMaskedCompositeShape::MaskWrapper::commitChanges()
{
	bool res = m_maskedShape->m_mask->commitChanges();
	m_maskedShape->m_mutationSignals.m_shapeMutated.fire( MUTATION_DISCARD_CACHED_DISTANCES );
	return res;
}

#endif

bool hknpMaskedCompositeShape::MaskWrapper::isShapeKeyEnabled( hknpShapeKey key ) const
{
	return m_maskedShape->m_mask->isShapeKeyEnabled( key );
}

int hknpMaskedCompositeShape::MaskWrapper::calcSize() const
{
	return sizeof( *this );
}

#endif	// !HK_PLATFORM_SPU

#if defined( HK_PLATFORM_PPU )

//
//	Automatically set the SPU flags on this shape.

void hknpMaskedCompositeShape::computeSpuFlags()
{
	const_cast<hknpCompositeShape*>(m_shape.val())->computeSpuFlags();
	m_flags.orWith(m_shape->getFlags().get(hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU) | m_shape->getFlags().get(hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU));
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
