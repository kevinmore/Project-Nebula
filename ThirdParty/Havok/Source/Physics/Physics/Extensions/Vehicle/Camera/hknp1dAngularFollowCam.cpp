/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/Camera/hknp1dAngularFollowCam.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

hknp1dAngularFollowCam::hknp1dAngularFollowCam() :
	m_cameraYawAngle(0),
	m_yawCorrection(0),
	m_yawSignCorrection(0)
{
	m_upDirWS.setZero();
	m_rigidBodyForwardDir.setZero();

	m_flat0DirWS.setZero();
	m_flat1DirWS.setZero();
}


hknp1dAngularFollowCamCinfo::hknp1dAngularFollowCamCinfo()
{
	m_yawCorrection = 0.0f;
	m_yawSignCorrection = 1.0f;
	m_rigidBodyForwardDir.set( 0.0f, 1.0f, 0.0f);
	m_upDirWS.set( 0.0f, 0.0f, 1.0f );
}


hknp1dAngularFollowCamCinfo::CameraSet::CameraSet()
{
	m_positionUS.set( 0.0f, 12.0f, 5.5f);
	m_lookAtUS.setZero();

	m_fov = 1.0f;

	m_velocity = 0.0f;
	m_speedInfluenceOnCameraDirection = 0.01f;
	m_angularRelaxation = 4.0f;
}


void hknp1dAngularFollowCam::reinitialize( const hknp1dAngularFollowCamCinfo &bp )
{
	m_yawCorrection = bp.m_yawCorrection;
	m_yawSignCorrection = bp.m_yawSignCorrection;
	m_upDirWS = bp.m_upDirWS;
	m_rigidBodyForwardDir = bp.m_rigidBodyForwardDir;

	m_flat0DirWS.setCross( m_upDirWS, m_rigidBodyForwardDir );
	m_flat0DirWS.normalize<3>();

	m_flat1DirWS.setCross( m_upDirWS, m_flat0DirWS );

	m_cameraYawAngle = 0.0f;

	m_set[0] = bp.m_set[0];
	m_set[1] = bp.m_set[1];
}


hknp1dAngularFollowCam::hknp1dAngularFollowCam(const hknp1dAngularFollowCamCinfo &bp)
{
	reinitialize(bp);
}

hknp1dAngularFollowCam::~hknp1dAngularFollowCam()
{
}


hkReal hknp1dAngularFollowCam::calcYawAngle(const hkReal factor1, const hkTransform& trans, const hkVector4& linearVelocity)
{
	const hkReal factor0 = 1.0f - factor1;

	const hkTransform &t_wsFcs    = trans;

/*
// uncomment following if you want the camera to use the local 'up'
// is handy if you are driving in loopings etc.

	//#define USE_LOCAL_UP
#ifdef USE_LOCAL_UP
	m_upDirWS=hkVector4(t_wsFcs.getRotation()(0,2),t_wsFcs.getRotation()(1,2),t_wsFcs.getRotation()(2,2));
	m_flat0DirWS.setCross(m_upDirWS,m_flat1DirWS);
	m_flat0DirWS.normalize3();
	m_flat1DirWS.setCross(m_upDirWS,m_flat0DirWS );
#endif
*/

	hkVector4 forwardDirWS;
	forwardDirWS.setRotatedDir(t_wsFcs.getRotation(), m_rigidBodyForwardDir );

	const hkReal speedInfluenceOnCameraDirection = m_set[0].m_speedInfluenceOnCameraDirection * factor0 + m_set[1].m_speedInfluenceOnCameraDirection * factor1;

	const hkVector4 &velocityWS = linearVelocity;

	hkVector4 tv; tv.setAddMul( forwardDirWS, velocityWS, hkSimdReal::fromFloat(0.01f * speedInfluenceOnCameraDirection));

		// calculate new yaw angle
	const hkReal u = tv.dot<3>( m_flat0DirWS ).getReal();
	const hkReal v = tv.dot<3>( m_flat1DirWS ).getReal();

	hkReal yaw_angle = hkMath::atan2fApproximation( v, u ) * m_yawSignCorrection - m_yawCorrection + 0.5f * HK_REAL_PI;

	return yaw_angle;
}


hkReal hknp1dAngularFollowCam::calcVelocityFactor(const hkVector4& chassisVelocity)
{
		// Work out factors based on velocity
	const hkReal  absVelocity = chassisVelocity.length<3>().getReal();

	hkReal factor1 = (absVelocity-m_set[0].m_velocity) / (m_set[1].m_velocity-m_set[0].m_velocity);
	{   // clip factor1
		factor1 = hkMath::clamp( factor1, hkReal(0.0f), hkReal(1.0f) );  // clip it
	}

	return factor1;
}


void hknp1dAngularFollowCam::resetCamera( const hkTransform& trans, const hkVector4& linearVelocity, const hkVector4& angularVelocity)
{
	const hkReal factor1 = calcVelocityFactor(linearVelocity);
	hkReal yawAngle = calcYawAngle(factor1, trans,linearVelocity);

	m_cameraYawAngle = yawAngle;
}

void hknp1dAngularFollowCam::calculateCamera( const CameraInput &in, CameraOutput &out )
{
	const hkReal factor1 = calcVelocityFactor(in.m_linearVelocity);
	const hkReal factor0 = 1.0f - factor1;

		// Work out yaw change based on factors and velocity.
	{

		hkReal yawAngle = calcYawAngle(factor1, in.m_fromTrans,in.m_linearVelocity);

		if (hkMath::isFinite(yawAngle)) // To avoid hanging if the object flies to infinity
		{
			while ( ( yawAngle + HK_REAL_PI ) < m_cameraYawAngle )
			{
				m_cameraYawAngle -= ( 2.0f * HK_REAL_PI );
			}

			while ( ( yawAngle - HK_REAL_PI ) > m_cameraYawAngle )
			{
				m_cameraYawAngle += ( 2.0f * HK_REAL_PI );
			}
		}


		// now lets see how fast we turn the camera to achieve this target angle.
		const hkReal angularRelaxation = factor0 * m_set[0].m_angularRelaxation + factor1 * m_set[1].m_angularRelaxation;
		const hkReal angularFactor = hkMath::min2( hkReal(1.0f), angularRelaxation * in.m_deltaTime );
		const hkReal deltaAngle = angularFactor * (yawAngle - m_cameraYawAngle);

		m_cameraYawAngle += deltaAngle;

	}

	const hkTransform& chassisTransform = in.m_fromTrans;

	hkQuaternion q; q.setAxisAngle(m_upDirWS, m_cameraYawAngle);
	hkTransform	r_ws_us; r_ws_us.set(q, chassisTransform.getTranslation());

	{	// calculate camera position
		hkVector4 camPosUS;
		camPosUS.setInterpolate( m_set[0].m_positionUS, m_set[1].m_positionUS, hkSimdReal::fromFloat(factor1));
		out.m_positionWS.setTransformedPos(r_ws_us,camPosUS);
	}


	{	// calculate lookat
		hkVector4 lookAtUS;
		lookAtUS.setInterpolate( m_set[0].m_lookAtUS, m_set[1].m_lookAtUS, hkSimdReal::fromFloat(factor1));
		out.m_lookAtWS.setTransformedPos(chassisTransform,lookAtUS);
	}


	{	// calculate updir
		out.m_upDirWS = m_upDirWS;
	}

	{	// calculate fov
		out.m_fov = m_set[0].m_fov * factor0 + m_set[1].m_fov * factor1;
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
