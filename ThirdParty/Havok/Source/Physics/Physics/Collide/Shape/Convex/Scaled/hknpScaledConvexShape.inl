/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>


HK_FORCE_INLINE void hknpScaledConvexShapeBase::init(
	const hknpConvexShape* childShape, hkVector4Parameter scale, ScaleMode mode )
{
	HK_ASSERT( 0xaf1e32aa, childShape && childShape->m_dispatchType == hknpCollisionDispatchType::CONVEX );
	m_childShape = childShape;
#if !defined(HK_PLATFORM_HAS_SPU) || defined(HK_PLATFORM_SPU)
	m_childShapeSize = 0;
#else
	m_childShapeSize = childShape->calcSize();
	HK_ASSERT2( 0xaf4431f4, (hkUint32)m_childShapeSize <= HKNP_MAX_SHAPE_SIZE_ON_SPU - sizeof(hknpScaledConvexShape), "The size of the to-be-scaled convex child shape exceeds the allowed size on SPU." );
#endif
	const hkUint16 inheritedFlags = USE_SINGLE_POINT_MANIFOLD | USE_SMALL_FACE_INDICES;
	m_flags.orWith( childShape->getFlags().get(inheritedFlags) );
	setScale( scale, mode );
}

HK_FORCE_INLINE void hknpScaledConvexShapeBase::setScale( hkVector4Parameter scale, hknpShape::ScaleMode mode )
{
	m_scale = scale;
	m_convexRadius = getChildShape()->m_convexRadius;
	hknpShapeUtil::calcScalingParameters( *getChildShape(), mode, &m_scale, &m_convexRadius, &m_translation );
}

HK_FORCE_INLINE const hkVector4& hknpScaledConvexShapeBase::getScale() const
{
	return m_scale;
}

HK_FORCE_INLINE const hkVector4& hknpScaledConvexShapeBase::getTranslation() const
{
	return m_translation;
}

HK_FORCE_INLINE hknpScaledConvexShapeBase* hknpScaledConvexShapeBase::createInPlace(
	const hknpConvexShape* childShape, hkVector4Parameter scale, ScaleMode scaleMode, hkUint8* buffer, int bufferSize )
{
#if !defined(HK_ALIGN_RELAX_CHECKS)
	HK_ASSERT2( 0x3795317e, !(hkUlong(buffer) & 0xF), "Buffer must be 16-byte aligned" );
	HK_ASSERT2( 0x1e943735, (hkUint32)bufferSize >= sizeof(hknpScaledConvexShapeBase), "Shape too large to fit in buffer" );
#endif

#if !defined(HK_PLATFORM_SPU)
	// In-place construct
	hknpScaledConvexShapeBase* scaledConvex = new (buffer) hknpScaledConvexShapeBase( childShape, scale, scaleMode );
#else
	// Initialize manually to avoid creating the v-table
	hknpScaledConvexShapeBase* scaledConvex = reinterpret_cast<hknpScaledConvexShapeBase*>(buffer);
	scaledConvex->hknpShape::init( hknpCollisionDispatchType::CONVEX );
	scaledConvex->init( childShape, scale, scaleMode );

	hknpShapeVirtualTableUtil::patchVirtualTableWithType<hknpShapeType::SCALED_CONVEX>(scaledConvex);
#endif

	scaledConvex->m_memSizeAndFlags = 0;
	return scaledConvex;
}

HK_FORCE_INLINE void hknpScaledConvexShapeBase::scaleVertex( hkVector4Parameter vertex, hkVector4& scaledVertexOut )
{
	scaledVertexOut.setAddMul( m_translation, m_scale, vertex );
}

HK_FORCE_INLINE hkReal hknpScaledConvexShapeBase::boundMinAngle( hkReal minAngle ) const
{
	// Crudely bound min angle
	hkVector4 nonUniformScale; nonUniformScale.setNormalizedEnsureUnitLength<3>( m_scale );
	return minAngle * nonUniformScale( nonUniformScale.getIndexOfMinAbsComponent<3>() );
}

HK_FORCE_INLINE const hknpConvexShape* hknpScaledConvexShapeBase::getChildShape() const
{
#if defined(HK_PLATFORM_SPU)
	// Fetch child shape from PPU if it's not yet on SPU. The child shape will be uploaded to the same overall shape
	// buffer right behind this hknpScaledConvexShapeBase. If m_childShapeSize is 0 then the child shape is already
	// available on SPU (either directly following this hknpScaledConvexShapeBase shape in SPU memory if it has been
	// uploaded from PPU or somewhere else in SPU memory if it has been created locally on SPU.)
	if ( m_childShapeSize > 0 )
	{
		hknpConvexShape* childShapeOnSpu = const_cast<hknpConvexShape*>( hkAddByteOffsetConst( reinterpret_cast<const hknpConvexShape*>(this), sizeof(*this) ) );
		hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( childShapeOnSpu, m_childShape, m_childShapeSize, hkSpuDmaManager::READ_COPY );
		HK_SPU_DMA_PERFORM_FINAL_CHECKS( m_childShape, childShapeOnSpu, m_childShapeSize );
		hknpShapeVirtualTableUtil::patchVirtualTable( childShapeOnSpu );
		m_childShape = childShapeOnSpu;
		m_childShapeSize = 0;
	}
	return m_childShape;
#else
	return m_childShape;
#endif
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
