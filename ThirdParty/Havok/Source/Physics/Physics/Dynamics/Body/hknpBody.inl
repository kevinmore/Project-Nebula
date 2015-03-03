/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Collide/Shape/hknpShape.h>


HK_FORCE_INLINE hknpBody::hknpBody()
{
	HK_ON_DEBUG( hkString::memSet( this, 0xcd, hkSizeOf(hknpBody) ); )
	m_userData = 0;
}

HK_FORCE_INLINE void hknpBody::operator=( const hknpBody& other )
{
	hkString::memCpy16NonEmpty( this, &other, sizeof(hknpBody)>>4 );
}


HK_FORCE_INLINE hkBool32 hknpBody::isValid() const
{
	return m_flags.anyIsSet( IS_STATIC | IS_DYNAMIC );
}

HK_FORCE_INLINE hkBool32 hknpBody::isAddedToWorld() const
{
	return (m_broadPhaseId - HKNP_INVALID_BROAD_PHASE_ID);
}

HK_FORCE_INLINE hkBool32 hknpBody::isStatic() const
{
	return m_flags.anyIsSet( IS_STATIC );
}

HK_FORCE_INLINE hkBool32 hknpBody::isDynamic() const
{
	return m_flags.anyIsSet( IS_DYNAMIC );
}

HK_FORCE_INLINE hkBool32 hknpBody::isKeyframed() const
{
	return m_flags.anyIsSet( IS_KEYFRAMED );
}

HK_FORCE_INLINE hkBool32 hknpBody::isStaticOrKeyframed() const
{
	return m_flags.anyIsSet( IS_STATIC | IS_KEYFRAMED );
}

HK_FORCE_INLINE hkBool32 hknpBody::isActive() const
{
	return m_flags.anyIsSet( IS_ACTIVE );
}

HK_FORCE_INLINE hkBool32 hknpBody::isInactive() const
{
	return 0 == ( m_flags.anyIsSet( IS_STATIC | IS_ACTIVE ) );
}

HK_FORCE_INLINE const hkTransform& hknpBody::getTransform() const
{
	return m_transform;
}

HK_FORCE_INLINE void hknpBody::getMotionToBodyTransformEstimate( hkQTransform* HK_RESTRICT bodyFromMotionOut ) const
{
	m_motionToBodyRotation.unpack( &bodyFromMotionOut->m_rotation.m_vec );
	const hkReal* p = &m_transform.getColumn(0)(0);
	bodyFromMotionOut->m_translation.set( p[0+3], p[4+3], p[8+3] );
}

HK_FORCE_INLINE void hknpBody::setTransform(const hkTransform& transform)
{
#if HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED
	// we move the intermediate variables into local register to avoid load,store,load,store,...
	hkVector4 w0; w0.setXYZ_W(transform.getRotation().getColumn<0>(), m_transform.getRotation().getColumn<0>() );
	hkVector4 w1; w1.setXYZ_W(transform.getRotation().getColumn<1>(), m_transform.getRotation().getColumn<1>() );
	hkVector4 w2; w2.setXYZ_W(transform.getRotation().getColumn<2>(), m_transform.getRotation().getColumn<2>() );
	hkVector4 w3; w3.setXYZ_W(transform.getTranslation(),			  m_transform.getTranslation()			 );

	hkTransform* HK_RESTRICT dest = &m_transform;
	dest->getRotation().setCols(w0,w1,w2);
	dest->setTranslation(w3);
#else
	hkTransform* HK_RESTRICT dest = &m_transform;
	dest->getRotation().getColumn<0>().setXYZ(transform.getRotation().getColumn<0>());
	dest->getRotation().getColumn<1>().setXYZ(transform.getRotation().getColumn<1>());
	dest->getRotation().getColumn<2>().setXYZ(transform.getRotation().getColumn<2>());
	dest->getTranslation().setXYZ(transform.getTranslation());
#endif
}

HK_FORCE_INLINE void hknpBody::setTransformComAndLookAhead( const hkTransform& newTransform )
{
	m_transform = newTransform;
}

HK_FORCE_INLINE int hknpBody::getDeactivatedIslandIndex() const
{
	HK_ASSERT( 0xf03ddfe2, isInactive() );
	return m_indexIntoActiveListOrDeactivatedIslandId;
}

HK_FORCE_INLINE void hknpBody::getCenterOfMassLocal( hkVector4& comLocalOut ) const
{
	hkSimdReal x = m_transform.getColumn<0>().getW();
	hkSimdReal y = m_transform.getColumn<1>().getW();
	hkSimdReal z = m_transform.getColumn<2>().getW();
	comLocalOut.set( x,y,z,z );
}

HK_FORCE_INLINE const hkReal& hknpBody::getCollisionLookAheadDistance() const
{
	return m_transform.getTranslation()(3);
}

HK_FORCE_INLINE void hknpBody::setCollisionLookAheadDistance( hkReal distance )
{
	m_transform.getTranslation()(3) = distance;
}

HK_FORCE_INLINE void hknpBody::setShape( const hknpShape* shape )
{
	m_shape = shape;

	if (shape)
	{
		int shapeSizeInBytes = shape->calcSize();
		HK_ASSERT(0xaf4431f3, (shapeSizeInBytes & 0x0f) == 0);
	#if defined(HK_PLATFORM_HAS_SPU)
		HK_ASSERT(0xaf4431f2, shapeSizeInBytes > 0 );
	#endif
		m_shapeSizeDiv16 = hkUint8(shapeSizeInBytes >> 4);

		if ( (shapeSizeInBytes > HKNP_MAX_SHAPE_SIZE_ON_SPU) || (shape->getFlags().anyIsSet(hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU)) )
		{
			m_spuFlags.orWith(FORCE_NARROW_PHASE_PPU);
		}
	}
	else
	{
		m_shapeSizeDiv16 = 0;
	}
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
