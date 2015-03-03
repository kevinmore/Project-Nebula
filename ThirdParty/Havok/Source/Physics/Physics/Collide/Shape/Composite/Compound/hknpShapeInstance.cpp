/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpShapeInstance.h>

#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>


void hknpShapeInstance::getFullTransform( hkTransform& transformOut ) const
{
	transformOut.getColumn(0).setMul( m_transform.getColumn<0>(), m_scale.getComponent<0>() );
	transformOut.getColumn(1).setMul( m_transform.getColumn<1>(), m_scale.getComponent<1>() );
	transformOut.getColumn(2).setMul( m_transform.getColumn<2>(), m_scale.getComponent<2>() );
	transformOut.setTranslation( m_transform.getTranslation() );
}

void hknpShapeInstance::getFullTransformInverse( hkTransform& transformOut ) const
{
	hkVector4 invScale; invScale.setReciprocal( m_scale );
	hkMatrix3 invRotation; invRotation._setTranspose( m_transform.getRotation() );

	transformOut.getColumn(0).setMul( invRotation.getColumn<0>(), invScale );
	transformOut.getColumn(1).setMul( invRotation.getColumn<1>(), invScale );
	transformOut.getColumn(2).setMul( invRotation.getColumn<2>(), invScale );

	hkVector4 invTranslation; invTranslation._setRotatedDir( transformOut.getRotation(), m_transform.getTranslation() );
	transformOut.getTranslation().setNeg<3>( invTranslation );
}


#ifndef HK_PLATFORM_SPU

void hknpShapeInstance::setShape( const hknpShape* shape )
{
	m_shape = shape;
	if( shape )
	{
		setShapeMemorySize( shape->calcSize() );
	}
	else
	{
		setShapeMemorySize(0);
	}
}

void hknpShapeInstance::setTransform( const hkTransform& transform )
{
	int data[] = { getFlags(), getShapeMemorySize(), getLeafIndex() };
	m_transform = transform;

	data[0] &= ~(HAS_TRANSLATION | HAS_ROTATION);
	if( !transform.getTranslation().equalZero().allAreSet( hkVector4ComparisonMask::MASK_XYZ ) )
	{
		data[0] |= HAS_TRANSLATION;
	}

	if( !transform.getRotation().isApproximatelyEqual( hkMatrix3::getIdentity() ) )
	{
		data[0] |= HAS_ROTATION;
	}

	setFlags( data[0] );
	setShapeMemorySize( data[1] );
	setLeafIndex( data[2] );
}

void hknpShapeInstance::setScale( hkVector4Parameter scale, hknpShape::ScaleMode mode )
{
	int flags = (getFlags() & ~(HAS_SCALE | FLIP_ORIENTATION | SCALE_SURFACE));

	if( scale.notEqual( hkVector4::getConstant<HK_QUADREAL_1>() ).anyIsSet( hkVector4ComparisonMask::MASK_XYZ ) )
	{
		flags |= HAS_SCALE;
	}

	switch( scale.lessZero().getMask( hkVector4ComparisonMask::MASK_XYZ ) )
	{
	case hkVector4ComparisonMask::MASK_X:
	case hkVector4ComparisonMask::MASK_Y:
	case hkVector4ComparisonMask::MASK_Z:
	case hkVector4ComparisonMask::MASK_XYZ:	flags |= FLIP_ORIENTATION; break;
	default:							break;
	}

	if( mode == hknpShape::SCALE_SURFACE )
	{
		flags |= SCALE_SURFACE;
	}

	m_scale.setXYZ_W( scale, hkSimdReal::getConstant<HK_QUADREAL_0>() );
	setFlags( flags );
}

void hknpShapeInstance::calculateAabb( hkAabb& aabbOut ) const
{
	const int flags = getFlags();
	hkReal radius = getShape()->m_convexRadius;
	hkVector4 scale = getScale();

	hkTransform	fullTransform;
	fullTransform.setTranslation( m_transform.getTranslation() );

	// If the instance has scale and is convex, calculate scaling parameters
	if ((flags & hknpShapeInstance::HAS_SCALE) && getShape()->m_dispatchType == hknpCollisionDispatchType::CONVEX )
	{
		const hknpConvexShape* convex = getShape()->asConvexShape();
		HK_ASSERT(0x636c1b68, convex != HK_NULL);
		hkVector4 offset;
		hknpShapeUtil::calcScalingParameters( *convex, getScaleMode(), &scale, &radius, &offset );
		offset.setRotatedDir( m_transform.getRotation(), offset );
		fullTransform.getTranslation().add( offset );
	}

	// Compose scale into the rotation matrix
	fullTransform.getColumn(0).setMul( m_transform.getColumn<0>(), scale.getComponent<0>() );
	fullTransform.getColumn(1).setMul( m_transform.getColumn<1>(), scale.getComponent<1>() );
	fullTransform.getColumn(2).setMul( m_transform.getColumn<2>(), scale.getComponent<2>() );

	// Calculate AABB and expand by any radius correction calculated for scaling
	getShape()->calcAabb( fullTransform, aabbOut );
	aabbOut.expandBy( hkSimdReal::fromFloat(radius - getShape()->m_convexRadius) );
}

#endif // !defined(HK_PLATFORM_SPU)

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
