/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Physics/MotionState/hkMotionState.h>
#include <Common/Base/DebugUtil/DeterminismUtil/hkCheckDeterminismUtil.h>

void hkMotionState::initMotionState( const hkVector4& position, const hkQuaternion& rotation )
{
	m_sweptTransform.initSweptTransform( position, rotation );

	getTransform().set( rotation, position );
	
	m_deltaAngle.setZero();
	m_objectRadius = 1.0f;
	hkCheckDeterminismUtil::checkMt(0xf0000002, m_objectRadius);

	// Initialize the rest to "invalid" but deterministic values (HVK-6297).
	m_linearDamping.setZero();
	m_angularDamping.setZero();
	m_timeFactor.setZero();
	m_maxLinearVelocity = 0.0f;
	m_maxAngularVelocity = 0.0f;
	m_deactivationClass = 0;	// = hkpSolverInfo::DEACTIVATION_CLASS_INVALID
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
