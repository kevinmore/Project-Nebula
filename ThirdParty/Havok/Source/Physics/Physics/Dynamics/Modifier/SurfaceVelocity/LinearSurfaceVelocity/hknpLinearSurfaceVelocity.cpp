/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Modifier/SurfaceVelocity/LinearSurfaceVelocity/hknpLinearSurfaceVelocity.h>

hknpLinearSurfaceVelocity::hknpLinearSurfaceVelocity( hkVector4Parameter velocity, Space space, ProjectMethod projectMethod )
{
	//hkSimdReal velLength = velocity.length<3>();
	m_velocity.setXYZ_0( velocity );
	m_space = space;
	m_projectMethod = projectMethod;
	m_velocityMeasurePlane.setZero();
	m_maxVelocityScale = 4.0f;
}


void hknpLinearSurfaceVelocity::calcSurfaceVelocity(
								 hkVector4Parameter positionWs,	 hkVector4Parameter normalWs, const hkTransform& shapeTransform,
								 hkVector4* HK_RESTRICT linearSurfaceVelocityWsOut,	 hkVector4* HK_RESTRICT angularSurfaceVelocityWsOut
								 ) const
{
	angularSurfaceVelocityWsOut->setZero();
	hkVector4 linVel;
	if ( m_space == USE_LOCAL_SPACE )
	{
		linVel._setRotatedDir( shapeTransform.getRotation(), m_velocity );
	}
	else
	{
		linVel = m_velocity;
	}

	if  ( VELOCITY_PROJECT != m_projectMethod )
	{
		hkSimdReal origVel = linVel.length<3>();

		hkSimdReal dot = normalWs.dot<3>( linVel );
		linVel.subMul( dot, normalWs );	 // project into surface

		// to measure the new velocity length, we project it again onto our measure velocity plane (if available)
		hkVector4 vMeasure; vMeasure.setSubMul( linVel, m_velocityMeasurePlane, m_velocityMeasurePlane.dot<3>(linVel));

		hkSimdReal currentVelInv = vMeasure.lengthInverse<3, HK_ACC_23_BIT, HK_SQRT_SET_ZERO>( );

		hkSimdReal maxScale; maxScale.setFromFloat( m_maxVelocityScale );

		// clamp to avoid extreme cases
		currentVelInv.setMax( currentVelInv, -maxScale );
		currentVelInv.setMin( currentVelInv, maxScale );

		linVel.mul( origVel * currentVelInv );
	}

	*linearSurfaceVelocityWsOut = linVel;
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
