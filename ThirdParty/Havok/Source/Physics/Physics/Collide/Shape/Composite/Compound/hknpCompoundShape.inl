/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if !defined( HK_PLATFORM_SPU )

HK_FORCE_INLINE hknpCompoundShape::hknpCompoundShape( bool isMutableFlag )
:	m_isMutable( isMutableFlag )
{
	m_aabb.setEmpty();
}

HK_FORCE_INLINE hknpCompoundShape::hknpCompoundShape( hkFinishLoadedObjectFlag flag )
	:	hknpCompositeShape(flag)
	,	m_instances(flag)
{
	if( flag.m_finishing )
	{
		m_aabb.setEmpty();
	}
}

HK_FORCE_INLINE int hknpCompoundShape::getCapacity() const
{
	return m_instances.getCapacity();
}

HK_FORCE_INLINE hkUint8 hknpCompoundShape::calcNumShapeKeyBits( int maxNumInstances )
{
	hkUint8 numShapeKeyBits = 0;
	while( maxNumInstances )
	{
		numShapeKeyBits++;
		maxNumInstances >>= 1;
	}
	return hkMath::max2((hkUint8)1, numShapeKeyBits);
}

HK_FORCE_INLINE void hknpCompoundShape::getAllInstanceIds( hkArray<hknpShapeInstanceId>& instanceIdsOut ) const
{
	HK_ASSERT( 0x5e896099, instanceIdsOut.isEmpty() );
	const int capacity = m_instances.getCapacity();
	instanceIdsOut.reserve( capacity );
	for( int i=0; i<capacity; ++i )
	{
		if( m_instances.isAllocated( hknpShapeInstanceId(i) ) )
		{
			instanceIdsOut.pushBackUnchecked( hknpShapeInstanceId(i) );
		}
	}
}

#endif

HK_FORCE_INLINE hknpShapeInstanceId hknpCompoundShape::getInstanceIdFromShapeKey( hknpShapeKey key ) const
{
	return hknpShapeInstanceId( key >> HKNP_NUM_UNUSED_SHAPE_KEY_BITS(m_numShapeKeyBits) );
}

#if !defined( HK_PLATFORM_SPU )

HK_FORCE_INLINE void hknpCompoundShape::setInstanceEnabled( hknpShapeInstanceId instanceId, bool isEnabled )
{
	hknpShapeInstance& instance = m_instances[instanceId];
	if( instance.isEnabled() != isEnabled )
	{
		instance.setEnabled( isEnabled );
		bool aabbChanged = updateAabb();	

		HK_WARN_ON_DEBUG_IF( !isMutable(), 0x77a414e0,
			"Toggled an instance in an immutable compound shape. Simulation caches may not be updated correctly." );

		MutationFlags mutationFlags = MUTATION_DISCARD_CACHED_DISTANCES;
		if( aabbChanged )
		{
			mutationFlags.orWith( MUTATION_AABB_CHANGED );
		}
		m_mutationSignals.m_shapeMutated.fire( mutationFlags.get() );
	}
}

HK_FORCE_INLINE void hknpCompoundShape::setInstanceShapeTag( hknpShapeInstanceId instanceId, hknpShapeTag tag )
{
	hknpShapeInstance& instance = m_instances[instanceId];
	if( instance.getShapeTag() != tag )
	{
		instance.setShapeTag( tag );

		HK_WARN_ON_DEBUG_IF( !isMutable(), 0x77a414e1,
			"Changed an instance's shape tag in an immutable compound shape. Simulation caches may not be updated correctly." );
		m_mutationSignals.m_shapeMutated.fire( MUTATION_REBUILD_COLLISION_CACHES );
	}
}

#endif	// !HK_PLATFORM_SPU

HK_FORCE_INLINE hknpCompoundShape::InstanceFreeListArray::Iterator hknpCompoundShape::getShapeInstanceIterator() const
{
	return InstanceFreeListArray::Iterator( m_instances );
}

HK_FORCE_INLINE const hknpShapeInstance& hknpCompoundShape::getInstance( hknpShapeInstanceId instanceId ) const
{
	return m_instances.getAtWithCache( instanceId );
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
