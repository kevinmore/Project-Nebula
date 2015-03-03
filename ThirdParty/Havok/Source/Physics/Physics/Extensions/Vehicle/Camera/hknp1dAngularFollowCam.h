/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_1D_ANGULAR_FOLLOW_CAM_H
#define HKNP_1D_ANGULAR_FOLLOW_CAM_H

#include <Physics/Physics/Extensions/Vehicle/Camera/hknp1dAngularFollowCamCinfo.h>


/// Attaches a camera to a vehicle to aid rendering.
/// The position of the camera rotates around the object using a single axle (normally the up axis).
/// The camera tries to slowly move to a certain point (m_positionUS) defined in object space and always looks at a
/// given point in object space.
class hknp1dAngularFollowCam : public  hkReferencedObject
{
	public:

		struct CameraInput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_CAMERA, CameraInput);
			hkVector4		m_linearVelocity;
			hkVector4		m_angularVelocity;
			hkTransform		m_fromTrans;
			hkReal			m_deltaTime;
		};

		struct CameraOutput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_CAMERA, CameraInput);
			hkVector4  m_positionWS;
			hkVector4  m_lookAtWS;
			hkVector4  m_upDirWS;
			hkReal m_fov;
			hkReal m_pad[3];
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);

			/// Default constructor
		hknp1dAngularFollowCam();

			/// Constructor.
		hknp1dAngularFollowCam( const hknp1dAngularFollowCamCinfo &cinfo );

			/// Destructor.
		virtual ~hknp1dAngularFollowCam();

			/// Reset all values except pointer values.
		void reinitialize( const hknp1dAngularFollowCamCinfo &bp );

			/// Immediately jump to the ideal yaw angle.
		virtual void resetCamera( const hkTransform& trans, const hkVector4& linearVelocity, const hkVector4& angularVelocity);

		virtual void calculateCamera ( const CameraInput &in, CameraOutput &out );

	protected:

			/// Internal methods for calculating camera position
		HK_FORCE_INLINE hkReal calcYawAngle(const hkReal factor1, const hkTransform& trans, const hkVector4& linearVelocity);
		HK_FORCE_INLINE hkReal calcVelocityFactor(const hkVector4& bodyVelocity);

	protected:

		hkReal m_cameraYawAngle;

		hkReal m_yawCorrection;
		hkReal m_yawSignCorrection;

		hkVector4 m_upDirWS;
		hkVector4 m_rigidBodyForwardDir;

		hkVector4 m_flat0DirWS;  // an orthogonal to m_upDirWS
		hkVector4 m_flat1DirWS;	 // an orthogonal to m_upDirWS and m_flat0DirWS

		hknp1dAngularFollowCamCinfo::CameraSet m_set[2];
};

#endif // HKNP_1D_ANGULAR_FOLLOW_CAM_H

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
