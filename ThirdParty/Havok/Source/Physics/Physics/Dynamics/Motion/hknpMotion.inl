/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpMotion::hknpMotion()
{
	HK_ON_DEBUG( hkString::memSet( this, 0xcd, hkSizeOf(hknpMotion) ); )
}

HK_FORCE_INLINE void hknpMotion::operator=( const hknpMotion& other )
{
	hkString::memCpy16NonEmpty( this, &other, sizeof(hknpMotion) >> 4 );
}

HK_FORCE_INLINE void hknpMotion::reset()
{
#if defined(HK_PLATFORM_XBOX360)
	hkString::memClear128( this, hkSizeOf(hknpMotion) );
#else
	hkString::memClear16( this, sizeof(hknpMotion) >> 4 );
#endif

	m_orientation.setIdentity();
	m_cellIndex = hknpCellIndex( HKNP_INVALID_CELL_IDX );
	m_motionPropertiesId = hknpMotionPropertiesId::STATIC;
}

HK_FORCE_INLINE hkSimdReal hknpMotion::getProjectedPointVelocity( hkVector4Parameter position, hkVector4Parameter normal ) const
{
	hkVector4 relPos; relPos.setSub( position, getCenterOfMassInWorld() );
	hkVector4 arm; arm.setCross( normal, relPos );
	hkVector4 armLocal; armLocal._setRotatedInverseDir( m_orientation, arm );
	return armLocal.dot<3>( m_angularVelocity ) + m_linearVelocity.dot<3>( normal );
}

HK_FORCE_INLINE void hknpMotion::_getPointVelocity( hkVector4Parameter position, hkVector4& velOut ) const
{
	hkVector4 relPos; relPos.setSub( position, getCenterOfMassInWorld() );
	hkVector4 angVelWorld; angVelWorld._setRotatedDir( m_orientation, m_angularVelocity );
	hkVector4 arm; arm.setCross( angVelWorld, relPos );
	velOut.setAdd( arm, m_linearVelocity );
}

HK_FORCE_INLINE void hknpMotion::_getPointVelocityUsingVelocity( hkVector4Parameter linVel, hkVector4Parameter angVelWorld, hkVector4Parameter position, hkVector4& velOut ) const
{
	hkVector4 relPos; relPos.setSub( position, getCenterOfMassInWorld() );
	hkVector4 arm; arm.setCross( angVelWorld, relPos );
	velOut.setAdd( arm, linVel );
}

HK_FORCE_INLINE void hknpMotion::getInverseInertiaLocal( hkVector4& invInertiaOut ) const
{
	invInertiaOut.load<4,HK_IO_SIMD_ALIGNED>( m_inverseInertia );
}

HK_FORCE_INLINE void hknpMotion::getInertiaLocal( hkVector4& inertiaOut ) const
{
	hkVector4 invInertia; invInertia.load<4,HK_IO_SIMD_ALIGNED>( m_inverseInertia );
	inertiaOut.setReciprocal( invInertia );
}

HK_FORCE_INLINE hkSimdReal hknpMotion::getInverseMass() const
{
	hkVector4 inv; getInverseInertiaLocal( inv );
	return inv.getComponent<3>();
}

HK_FORCE_INLINE hkSimdReal hknpMotion::getMass() const
{
	hkSimdReal invMass = getInverseMass();
	if( invMass.isEqualZero() )
	{
		return hkSimdReal_0;
	}
	return invMass.reciprocal();
}

HK_FORCE_INLINE void hknpMotion::setInverseMass( hkSimdRealParameter invMass )
{
	hkVector4 unpackedInvInertia;
	unpackedInvInertia.load<4,HK_IO_SIMD_ALIGNED>( m_inverseInertia );
	hkSimdReal oldInvMass = unpackedInvInertia.getComponent<3>();
	HK_ASSERT2( 0x3d38152e, !oldInvMass.isEqualZero(),
		"Motion has infinite mass. Use setInverseInertiaAndMass() or setFromMotionProperties() to set the inertia and mass values" );
	unpackedInvInertia.mul( invMass / oldInvMass );
	unpackedInvInertia.setComponent<3>( invMass );
	unpackedInvInertia.store<4,HK_IO_SIMD_ALIGNED,HK_ROUND_NEAREST>( m_inverseInertia );
}

HK_FORCE_INLINE void hknpMotion::setInverseInertia(hkVector4Parameter invInertia )
{
	hkVector4 unpackedInvInertia;
	unpackedInvInertia.load<4,HK_IO_SIMD_ALIGNED>( m_inverseInertia );
	unpackedInvInertia.setXYZ_W( invInertia, unpackedInvInertia );
	unpackedInvInertia.store<4,HK_IO_SIMD_ALIGNED,HK_ROUND_NEAREST>( m_inverseInertia );
}

HK_FORCE_INLINE void hknpMotion::setInverseInertiaAndMass( hkVector4Parameter invInertiaAndMass )
{
	invInertiaAndMass.store<4,HK_IO_SIMD_ALIGNED,HK_ROUND_NEAREST>( m_inverseInertia );
}

HK_FORCE_INLINE void hknpMotion::setInfiniteInertiaAndMass()
{
	m_inverseInertia[0].setZero();
	m_inverseInertia[1].setZero();
	m_inverseInertia[2].setZero();
	m_inverseInertia[3].setZero();
}

HK_FORCE_INLINE void hknpMotion::setCenterOfMassInWorld( hkVector4Parameter com )
{
	m_centerOfMassAndMassFactor.setXYZ( com );
}

HK_FORCE_INLINE const hkVector4& hknpMotion::getCenterOfMassInWorld() const
{
	return m_centerOfMassAndMassFactor;
}

HK_FORCE_INLINE void hknpMotion::getWorldTransform( hkQTransform& transform ) const
{
	transform.set( m_orientation, getCenterOfMassInWorld() );
}

HK_FORCE_INLINE bool hknpMotion::isStatic() const
{
	return m_solverId == hknpSolverId(0);
}

HK_FORCE_INLINE hkBool32 hknpMotion::isDynamic() const
{
	return m_solverId.value();
}

HK_FORCE_INLINE bool hknpMotion::isActive() const
{
	return m_solverId.isValid() && !isStatic();
}

HK_FORCE_INLINE bool hknpMotion::isValid() const
{
	return m_spaceSplitterWeight!=0;
}

HK_FORCE_INLINE bool hknpMotion::hasInfiniteMass() const
{
	const int zero[2] = { 0, 0 };
	const int sizeOf = sizeof( m_inverseInertia );
	HK_COMPILE_TIME_ASSERT( sizeof(zero) == sizeOf );
	return ( hkString::memCmp( &m_inverseInertia, &zero, sizeOf ) == 0 );
}

HK_FORCE_INLINE const hkVector4& hknpMotion::getLinearVelocity() const
{
	return m_linearVelocity;
}

HK_FORCE_INLINE const hkVector4& hknpMotion::getAngularVelocityLocal() const
{
	return m_angularVelocity;
}

HK_FORCE_INLINE void hknpMotion::_getAngularVelocity( hkVector4& angVelocityOut ) const
{
	angVelocityOut._setRotatedDir( m_orientation, m_angularVelocity );
}

HK_FORCE_INLINE void hknpMotion::_applyPointImpulse( hkVector4Parameter impulse, hkVector4Parameter position )
{
	HK_ASSERT( 0xf0346912, impulse.isOk<4>() && position.isOk<4>() );

	hkVector4 linearVel  = m_linearVelocity;
	hkVector4 angularVel = m_angularVelocity;

	hkVector4 inertiaInvLocal; getInverseInertiaLocal( inertiaInvLocal );

	linearVel.addMul( inertiaInvLocal.getComponent<3>(), impulse );

	hkVector4 relMassCenter; relMassCenter.setSub( position, getCenterOfMassInWorld() );
	hkVector4 crossWs; crossWs.setCross( relMassCenter, impulse );

	hkVector4 crossLs; crossLs._setRotatedInverseDir( m_orientation, crossWs );
	angularVel.addMul( inertiaInvLocal, crossLs );

	m_linearVelocity = linearVel;
	m_angularVelocity = angularVel;
}

HK_FORCE_INLINE hkSimdReal hknpMotion::_calcProjectedInverseMass( hkVector4Parameter direction, hkVector4Parameter position ) const
{
	hkVector4 inertiaInvLocal; getInverseInertiaLocal( inertiaInvLocal );

	hkSimdReal invMass = inertiaInvLocal.getComponent<3>();

	hkVector4 relMassCenter; relMassCenter.setSub( position, getCenterOfMassInWorld() );
	hkVector4 crossWs; crossWs.setCross( relMassCenter, direction );
	hkVector4 crossLs; crossLs._setRotatedInverseDir( m_orientation, crossWs );
	hkVector4 directionLs; directionLs._setRotatedInverseDir( m_orientation, direction );
	hkVector4 angularEffect; angularEffect.setMul( inertiaInvLocal, crossLs );
	invMass = invMass + directionLs.dot<3>( angularEffect );
	return invMass;
}

HK_FORCE_INLINE void hknpMotion::_applyLinearImpulse( hkVector4Parameter impulse )
{
	HK_ASSERT( 0xf034df45, impulse.isOk<4>() );

	hkVector4 linearVel  = m_linearVelocity;
	hkVector4 inertiaInvLocal; getInverseInertiaLocal( inertiaInvLocal );
	linearVel.addMul( inertiaInvLocal.getComponent<3>(), impulse);
	m_linearVelocity = linearVel;
}

HK_FORCE_INLINE void hknpMotion::_applyAngularImpulse( hkVector4Parameter angImpulseWorld )
{
	HK_ASSERT( 0xf034df45, angImpulseWorld.isOk<4>() );

	hkVector4 angularVel = m_angularVelocity;

	hkVector4 inertiaInvLocal; getInverseInertiaLocal( inertiaInvLocal );
	hkVector4 angImpulseLocal; angImpulseLocal._setRotatedInverseDir( m_orientation, angImpulseWorld );
	angularVel.addMul( inertiaInvLocal, angImpulseLocal );

	m_angularVelocity = angularVel;
}


HK_FORCE_INLINE void hknpMotion::setLinearVelocity( hkVector4Parameter velocity )
{
	HK_ASSERT( 0xf034df45, velocity.isOk<4>() );
	m_linearVelocity = velocity;
}

HK_FORCE_INLINE void hknpMotion::_setAngularVelocity( hkVector4Parameter angVelocity )
{
	HK_ASSERT( 0xf034df45, angVelocity.isOk<4>() );
	hkVector4 angVelLocal; angVelLocal._setRotatedInverseDir( m_orientation, angVelocity );
	m_angularVelocity = angVelLocal;
}


HK_FORCE_INLINE void hknpMotion::setMassFactor( hkSimdRealParameter massFactor )
{
	m_centerOfMassAndMassFactor.setW( massFactor );
}

HK_FORCE_INLINE hkSimdReal hknpMotion::getMassFactor() const
{
	return m_centerOfMassAndMassFactor.getW();
}

HK_FORCE_INLINE void hknpMotion::setSpaceSplitterWeight( hkUint8 weight )
{
	HK_ASSERT2( 0xf034df45, weight>0, "The space splitter weight must be greater than 1." );
	HK_ASSERT2( 0xf034df45, isValid(), "setSpaceSplitterWeight requires a valid motion" );
	m_spaceSplitterWeight = weight;
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
