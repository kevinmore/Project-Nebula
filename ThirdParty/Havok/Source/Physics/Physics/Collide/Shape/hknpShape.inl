/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpShape::hknpShape( hknpCollisionDispatchType::Enum dispatchType )
:	hkReferencedObject()
,	m_properties(HK_NULL)
{
	init( dispatchType );
}

#endif

HK_FORCE_INLINE hknpShape::MassConfig::MassConfig()
{
	m_massOrNegativeDensity = -1.0f;
	m_inertiaFactor = 1.5f;
	m_quality = QUALITY_HIGH;
}

HK_FORCE_INLINE hknpShape::MassConfig::MassConfig( hkReal massOrNegativeDensity, hkReal inertiaFactor, Quality quality )
{
	m_massOrNegativeDensity = massOrNegativeDensity;
	m_inertiaFactor = inertiaFactor;
	m_quality = quality;
}

HK_FORCE_INLINE hknpShape::MassConfig hknpShape::MassConfig::fromMass( hkReal mass, hkReal inertiaFactor, Quality quality )
{
	HK_ASSERT( 0x1d3f55a0, mass > 0.0f );
	return MassConfig( mass, inertiaFactor, quality );
}

HK_FORCE_INLINE hknpShape::MassConfig hknpShape::MassConfig::fromDensity( hkReal density, hkReal inertiaFactor, Quality quality )
{
	HK_ASSERT( 0x3371d2a3, density > 0.0f );
	return MassConfig( -density, inertiaFactor, quality );
}

HK_FORCE_INLINE hkReal hknpShape::MassConfig::calcMassFromVolume( hkReal volume ) const
{
	if( m_massOrNegativeDensity >= 0.0f )
	{
		return m_massOrNegativeDensity;
	}
	else
	{
		return -m_massOrNegativeDensity * volume;
	}
}


HK_FORCE_INLINE void hknpShape::init( hknpCollisionDispatchType::Enum dispatchType )
{
	m_properties = HK_NULL;
	m_dispatchType = dispatchType;
	m_numShapeKeyBits = 0;
	m_userData = 0;
	m_convexRadius = 0.0f;
	m_flags.clear();
}

HK_FORCE_INLINE hknpShape::Flags hknpShape::getFlags() const
{
	return m_flags;
}

HK_FORCE_INLINE void hknpShape::setFlags( Flags flags )
{
	m_flags = flags;
}

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE void hknpShape::setProperty( hknpPropertyKey propertyKey, hkReferencedObject* propertyObject )
{
	if( m_properties == HK_NULL )
	{
		m_properties = new hkRefCountedProperties();
	}
	m_properties->addProperty( propertyKey, propertyObject );
}

HK_FORCE_INLINE void hknpShape::setMassProperties( const hknpShape::MassConfig& massConfig )
{
	hkDiagonalizedMassProperties dmp;
	buildMassProperties( massConfig, dmp );
	setMassProperties( dmp );
}

HK_FORCE_INLINE void hknpShape::setMassProperties( const hkDiagonalizedMassProperties& massProperties )
{
	// Try to reuse an already allocated mass properties object
	hknpShapeMassProperties* srcProps = (hknpShapeMassProperties*)getProperty( hknpShapePropertyKeys::MASS_PROPERTIES );
	hknpShapeMassProperties* props = srcProps ? srcProps : new hknpShapeMassProperties();

	props->m_compressedMassProperties.pack( massProperties );
	setProperty( hknpShapePropertyKeys::MASS_PROPERTIES, props );
	if( !srcProps )
	{
		props->removeReference();
	}
}

HK_FORCE_INLINE hkResult hknpShape::getMassProperties( hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	const hknpShapeMassProperties* props = (const hknpShapeMassProperties*)getProperty( hknpShapePropertyKeys::MASS_PROPERTIES );
	if( props )
	{
		props->m_compressedMassProperties.unpack( &massPropertiesOut );
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

HK_FORCE_INLINE hkResult hknpShape::getMassProperties( hkMassProperties& massPropertiesOut ) const
{
	const hknpShapeMassProperties* props = (const hknpShapeMassProperties*)getProperty( hknpShapePropertyKeys::MASS_PROPERTIES );
	if( props )
	{
		props->m_compressedMassProperties.unpack( massPropertiesOut );
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

#endif

HK_FORCE_INLINE hkReferencedObject* hknpShape::getProperty( hknpPropertyKey propertyKey ) const
{
	if( m_properties == HK_NULL )
	{
		return HK_NULL;
	}
	return m_properties->accessProperty( propertyKey );
}

HK_FORCE_INLINE const hknpConvexShape* hknpShape::asConvexShape() const
{
	if( m_flags.get(IS_CONVEX_SHAPE) )
	{
		return (const hknpConvexShape*)(this);
	}
	return HK_NULL;
}

HK_FORCE_INLINE const hknpConvexPolytopeShape* hknpShape::asConvexPolytopeShape() const
{
	if( m_flags.get(IS_CONVEX_POLYTOPE_SHAPE) )
	{
		return (const hknpConvexPolytopeShape*)(this);
	}
	return HK_NULL;
}

HK_FORCE_INLINE const hknpCompositeShape* hknpShape::asCompositeShape() const
{
	if( m_flags.get(IS_COMPOSITE_SHAPE) )
	{
		return (const hknpCompositeShape*)(this);
	}
	return HK_NULL;
}

HK_FORCE_INLINE const hknpHeightFieldShape* hknpShape::asHeightFieldShape() const
{
	if( m_flags.get(IS_HEIGHT_FIELD_SHAPE) )
	{
		return (const hknpHeightFieldShape*)(this);
	}
	return HK_NULL;
}

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE bool hknpShape::isMutable() const
{
	return const_cast<hknpShape*>(this)->getMutationSignals() != HK_NULL;
}

#endif

HK_FORCE_INLINE hkUint8 hknpShape::getNumShapeKeyBits() const
{
	return m_numShapeKeyBits;
}


HK_FORCE_INLINE hknpShapeKeyIterator::hknpShapeKeyIterator( const hknpShape& shape, const hknpShapeKeyMask* mask )
{
	m_shape = &shape;
	m_mask = mask;
	m_keyPath.reset();	// sets HKNP_INVALID_SHAPE_KEY
}

HK_FORCE_INLINE bool hknpShapeKeyIterator::isValid() const
{
	return m_keyPath.getKey() != HKNP_INVALID_SHAPE_KEY;
}

HK_FORCE_INLINE hknpShapeKey hknpShapeKeyIterator::getKey() const
{
	return m_keyPath.getKey();
}

HK_FORCE_INLINE const hknpShapeKeyPath& hknpShapeKeyIterator::getKeyPath() const
{
	return m_keyPath;
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
