/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>


// hknpBody must be 16 byte aligned since we use memCpy16() etc
HK_COMPILE_TIME_ASSERT(sizeof(hknpBody) % 16 == 0);
#if !defined(HK_REAL_IS_DOUBLE)
	HK_COMPILE_TIME_ASSERT(sizeof(hknpBody) <= 144);	
	HK_COMPILE_TIME_ASSERT(sizeof(hknpBody) >= 144);
#endif


hknpBodyCinfo::hknpBodyCinfo()
{
	m_shape = HK_NULL;
	m_reservedBodyId = hknpBodyId::invalid();
	m_motionId = hknpMotionId::invalid();
	m_qualityId = hknpBodyQualityId::invalid();
	m_materialId = hknpMaterialId::DEFAULT;
	m_collisionFilterInfo = 0;
	m_flags = 0;
	m_spuFlags = 0;
	m_collisionLookAheadDistance = 0;

	m_position.setZero();
	m_orientation.setIdentity();
	m_localFrame = HK_NULL;
}

void hknpBody::setBodyToMotionTranslation( hkVector4Parameter offset )
{
	hkReal* HK_RESTRICT p = &m_transform.getColumn(0)(0);
	offset.getComponent<0>().store<1>( &p[0+3] );
	offset.getComponent<1>().store<1>( &p[4+3] );
	offset.getComponent<2>().store<1>( &p[8+3] );
}

void hknpBody::updateMotionToBodyTransform( const hknpMotion& motion, const hkQuaternion* cachedBodyOrientation )
{
	hkVector4 comMinusPosition; comMinusPosition.setSub( motion.getCenterOfMassInWorld(), m_transform.getTranslation() );
	hkVector4 centerOfMassLocal; centerOfMassLocal._setRotatedInverseDir( m_transform.getRotation(), comMinusPosition );
	this->setBodyToMotionTranslation(centerOfMassLocal);

	hkQuaternion bodyQmotion;
	if ( cachedBodyOrientation )
	{
		bodyQmotion.setInverseMul( *cachedBodyOrientation, motion.m_orientation );
	}
	else
	{
		hkQuaternion bodyOrientation; bodyOrientation.set( m_transform.getRotation() );
		bodyQmotion.setInverseMul( bodyOrientation, motion.m_orientation );
	}

	m_motionToBodyRotation.pack( bodyQmotion.m_vec );
}

void hknpBody::updateComCenteredBoundingRadius( const hknpMotion& motion )
{
	// Calculate body's COM bounding sphere
	// get AABB in body space
	hkAabb aabb;
	m_shape->calcAabb( hkTransform::getIdentity(), aabb );
	HK_ASSERT( 0xb82534a2, aabb.isValid() );

	// get COM in body space
	hkVector4 offset;
	offset._setTransformedInversePos( getTransform(), motion.getCenterOfMassInWorld() );

	// move AABB to COM space
	aabb.m_min.sub(offset);
	aabb.m_max.sub(offset);

	// find the corner which is furthest away from the origin
	aabb.m_min.setAbs(aabb.m_min);
	aabb.m_max.setAbs(aabb.m_max);
	hkVector4 m; m.setMax( aabb.m_min, aabb.m_max);
	hkSimdReal radius = m.length<3>();
	radius.store<1>( &m_radiusOfComCenteredBoundingSphere );
}

void hknpBody::syncStaticMotionToBodyTransform()
{
	hkQTransform tfm; tfm.set( m_transform );
	hkQTransform m2b; m2b._setInverse( tfm );

	m_motionToBodyRotation.pack( m2b.m_rotation.m_vec );
	setBodyToMotionTranslation( m2b.m_translation );
}

void hknpBody::getBodyToMotionTransform( hkQTransform& transform ) const
{
	hkQTransform m2b;
	m_motionToBodyRotation.unpack( &m2b.m_rotation.m_vec );
	const hkReal* p = &m_transform.getColumn(0)(0);
	m2b.m_translation.set( p[0+3], p[4+3], p[8+3] );
	transform._setInverse(m2b);
}

void hknpBody::initialize( hknpBodyId id, const hknpBodyCinfo& cInfo )
{
	hkString::memClear16( this, sizeof(hknpBody)/16 );

	m_aabb.setEmpty();
	m_motionId = hknpMotionId::invalid();
	m_nextAttachedBodyId = hknpBodyId::invalid();
	m_broadPhaseId = hknpBroadPhaseId( HKNP_INVALID_BROAD_PHASE_ID );
	m_indexIntoActiveListOrDeactivatedIslandId = hknpBodyId::InvalidValue;
	m_radiusOfComCenteredBoundingSphere.setOne();	

	// ID
	m_id = id;

	//
	// Cinfo
	//

	hkTransform	transform;
	transform.set( cInfo.m_orientation, cInfo.m_position );
	setTransform( transform );
	setCollisionLookAheadDistance( cInfo.m_collisionLookAheadDistance );

	// Set SPU flags before calling setShape() as it may alter them.
	m_spuFlags = cInfo.m_spuFlags;

	setShape( cInfo.m_shape );
	m_materialId			= cInfo.m_materialId;
	m_qualityId				= cInfo.m_qualityId;
	m_collisionFilterInfo	= cInfo.m_collisionFilterInfo;

	const hkUint32 allowedFlags = hknpBody::FLAGS_MASK & ~hknpBody::INTERNAL_FLAGS_MASK;
	HK_WARN_ON_DEBUG_IF( cInfo.m_flags.anyIsSet( ~allowedFlags ),
		0xf0df3dde, "You enabled some disallowed body flags. These will be ignored.");
	m_flags = cInfo.m_flags.get( allowedFlags );

	// This body might have been in use before, so we need to make sure we clean the collision caches.
	m_flags.orWith( hknpBody::TEMP_REBUILD_COLLISION_CACHES );
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
