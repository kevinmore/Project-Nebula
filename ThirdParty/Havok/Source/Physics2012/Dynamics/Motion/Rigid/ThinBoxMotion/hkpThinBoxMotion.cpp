/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Motion/Rigid/ThinBoxMotion/hkpThinBoxMotion.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformUtil.h>

HK_COMPILE_TIME_ASSERT( sizeof(hkpThinBoxMotion) == sizeof(hkpMotion) );

hkpThinBoxMotion::hkpThinBoxMotion(const hkVector4& position, const hkQuaternion& rotation)
	:hkpBoxMotion( position, rotation )
{
	m_type = MOTION_THIN_BOX_INERTIA;
}

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
