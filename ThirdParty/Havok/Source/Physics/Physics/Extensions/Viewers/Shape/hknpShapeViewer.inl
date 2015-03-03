/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE void hknpShapeViewer::setInstancingEnabled( bool isEnabled )
{
	m_instancingEnabled = isEnabled;
}

HK_FORCE_INLINE void hknpShapeViewer::setConvexRadiusDisplayMode( hknpShape::ConvexRadiusDisplayMode radiusMode )
{
	m_radiusDisplayMode = radiusMode;
}

HK_FORCE_INLINE void hknpShapeViewer::removeWorld( const hknpWorld* world )
{
	removeWorld( getWorldIndex( world ) );
}

HK_FORCE_INLINE int hknpShapeViewer::getWorldIndex( const hknpWorld* world ) const
{
	const int numWorlds = m_worldDatas.getSize();
	for ( int i=0; i < numWorlds; ++i )
	{
		if ( m_worldDatas[i] && m_worldDatas[i]->m_world == world )
		{
			return i;
		}
	}
	return -1;
}

HK_FORCE_INLINE hkUlong hknpShapeViewer::composeDisplayObjectId( const hknpWorld* world, hknpBodyId bodyId ) const
{
	const int worldIndex = getWorldIndex( world );
	HK_ASSERT( 0x321f70fc, worldIndex != -1 );
	return composeDisplayObjectId( worldIndex, bodyId );
}

HK_FORCE_INLINE hkUlong hknpShapeViewer::composeDisplayObjectId( int worldIndex, hknpBodyId bodyId ) const
{
	// Use 4 bits for tag, 4 bits for world index, 24 for body ID
	hkUlong id = (((m_tag-s_tag) & 0xf) << 28) + ((worldIndex & 0xf) << 24) + bodyId.value();
	return id;
}

HK_FORCE_INLINE void hknpShapeViewer::decomposeDisplayObjectId( hkUlong id, hknpWorld*& worldOut, hknpBodyId& bodyIdOut ) const
{
	// Use 4 bits for tag, 4 bits for world index, 24 for body ID
	const int tag = int(id >> 28) & 0xf;
	if( tag == m_tag-s_tag )
	{
		const int worldIndex = int(id >> 24) & 0xf;
		if( worldIndex >= 0 && worldIndex < m_worldDatas.getSize() && m_worldDatas[ worldIndex ] )
		{
			worldOut = m_worldDatas[ worldIndex ]->m_world;
			bodyIdOut = hknpBodyId( id & ((1<<24)-1) );
			return;
		}
	}

	worldOut = HK_NULL;
	bodyIdOut = hknpBodyId::invalid();
}

HK_FORCE_INLINE hkBool hknpShapeViewer::pickObject( hkUint64 id, hkVector4Parameter worldPosition )
{
	hknpWorld* world;
	hknpBodyId bodyId;
	decomposeDisplayObjectId( (hkUlong)id, world, bodyId );

	if ( !world || !bodyId.isValid() || !world->isBodyAdded( bodyId ) )
	{
		return false;
	}

	const hknpBody& body = world->getBody( bodyId );

	if( body.isStatic() )
	{
		return false;
	}

	m_pickedWorld = world;
	m_pickedBodyId = bodyId;
	m_mouseSpringWorldPosition = worldPosition;
	m_mouseSpringLocalPosition.setTransformedInversePos( body.getTransform(), worldPosition );

	return true;
}

HK_FORCE_INLINE void hknpShapeViewer::dragObject( hkVector4Parameter newWorldSpacePoint )
{
	m_mouseSpringWorldPosition = newWorldSpacePoint;
}

HK_FORCE_INLINE void hknpShapeViewer::releaseObject()
{
	m_pickedWorld = HK_NULL;
	m_pickedBodyId = hknpBodyId::invalid();
}


inline unsigned hknpShapeViewer::GroupMapOperations::hash( hknpShapeViewer::GroupKey key, unsigned modulus )
{
	hkUint64 value = ( hkUint64(key.m_color) << 32 ) + ( hkUlong(key.m_shape) >> 4 ) + key.m_hash;
	return ( unsigned(value) * 2654435761U ) & modulus;
}

inline void hknpShapeViewer::GroupMapOperations::invalidate( hknpShapeViewer::GroupKey& key )
{
	key.m_shape = HK_NULL;
}

inline hkBool32 hknpShapeViewer::GroupMapOperations::isValid( hknpShapeViewer::GroupKey key )
{
	return ( key.m_shape != HK_NULL );
}

inline hkBool32 hknpShapeViewer::GroupMapOperations::equal( hknpShapeViewer::GroupKey key0, hknpShapeViewer::GroupKey key1 )
{
	return ( key0.m_shape == key1.m_shape ) && ( key0.m_hash == key1.m_hash ) && ( key0.m_color == key1.m_color );
}


inline unsigned hknpShapeViewer::BodyIdMapOperations::hash( hknpBodyId key, unsigned modulus )
{
	return ( key.value() * 2654435761U ) & modulus;
}

inline void hknpShapeViewer::BodyIdMapOperations::invalidate( hknpBodyId& key )
{
	key = hknpBodyId::invalid();
}

inline hkBool32 hknpShapeViewer::BodyIdMapOperations::isValid( hknpBodyId key )
{
	return ( key.isValid() );
}

inline hkBool32 hknpShapeViewer::BodyIdMapOperations::equal( hknpBodyId key0, hknpBodyId key1 )
{
	return key0 == key1;
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
