/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

template <bool pad>
void hknpPaddedManifold<pad>::checkConsistency() const
{
	if (m_numPoints)
	{
		HK_ASSERT( 0xfcdfd34, m_normal.isOk<4>() && m_normal.isNormalized<3>(1e-3f) && m_distances.isOk<4>() );
		for (int i = 0; i < m_numPoints; i++)
		{
            hkVector4 vec = m_positions[i];
			HK_ASSERT( 0xfcdfd35, vec.isOk<4>() );
		}
	}
}

template <bool pad>
void hknpPaddedManifold<pad>::init( hkVector4Parameter normal, const hkVector4* pointAndDistances, int numPoints )
{
	HK_ASSERT( 0xf0cd13dc, numPoints <= 4);
	hkString::memClear16( this, sizeof(*this)>>4);

	m_numPoints = numPoints;
	m_isNewSurface = 1;
	m_isNewManifold = 1;
// 	m_distances.setZero();
// 	m_positions[0].setZero();
// 	m_positions[1].setZero();
// 	m_positions[2].setZero();
// 	m_positions[3].setZero();
	for (int i=0; i < numPoints; i++)
	{
		const hkVector4 v = pointAndDistances[i];
		m_positions[i] = v;
		v.getComponent<3>().store<1>( &m_distances(i) );
	}
	m_normal = normal;
// 	m_collisionCache = HK_NULL;
// 	m_collisionCacheInMainMemory = HK_NULL;
// 	m_massChangerData = 0.0f;
// 	m_useIncreasedIterations = 0;
// 	m_materialB = HK_NULL;
	m_shapeKeyA = HKNP_INVALID_SHAPE_KEY;
	m_shapeKeyB = HKNP_INVALID_SHAPE_KEY;
}

template <bool pad>
void hknpPaddedManifold<pad>::init( hkVector4Parameter normal, hkVector4Parameter position, hkReal distance )
{
	const int numPoints = 1;
	HK_ASSERT( 0xf0cd13dc, numPoints <= 4);
	hkString::memClear16( this, sizeof(*this)>>4);

	m_numPoints = numPoints;
	m_isNewSurface = 1;
	m_isNewManifold = 1;
	m_distances.setAll( distance );
	hkVector4 p = position;
	m_positions[0] = p;
	m_positions[1] = p;
	m_positions[2] = p;
	m_positions[3] = p;
	m_normal = normal;
// 	m_collisionCache = HK_NULL;
// 	m_collisionCacheInMainMemory = HK_NULL;
// 	m_massChangerData = 0.0f;
// 	m_useIncreasedIterations = 0;
// 	m_materialB = HK_NULL;
	m_shapeKeyA = HKNP_INVALID_SHAPE_KEY;
	m_shapeKeyB = HKNP_INVALID_SHAPE_KEY;
}

//
//	Copies another manifold over this one

template <bool pad>
template <bool otherPad>
HK_FORCE_INLINE void hknpPaddedManifold<pad>::copy(const hknpPaddedManifold<otherPad>& other)
{
	// Copy base classes
	*reinterpret_cast<hknpManifoldBase*>(this) = reinterpret_cast<const hknpManifoldBase&>(other);

	// Copy padded members
	m_numPoints					= other.m_numPoints;
	m_manifoldType				= other.m_manifoldType;
	m_useIncreasedIterations	= other.m_useIncreasedIterations;
	m_isNewSurface				= other.m_isNewSurface;
	m_isNewManifold				= other.m_isNewManifold;
	m_massChangerData			= other.m_massChangerData;
	m_materialB					= other.m_materialB;
	m_shapeKeyA					= other.m_shapeKeyA;
	m_shapeKeyB					= other.m_shapeKeyB;
	m_collisionCache			= other.m_collisionCache;
	m_collisionCacheInMainMemory= other.m_collisionCacheInMainMemory;
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
