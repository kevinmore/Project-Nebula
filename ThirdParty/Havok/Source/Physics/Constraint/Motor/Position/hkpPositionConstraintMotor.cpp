/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Motor/Position/hkpPositionConstraintMotor.h>


hkpPositionConstraintMotor::hkpPositionConstraintMotor( hkReal currentPosition )
: hkpLimitedForceConstraintMotor()
{
	m_type = TYPE_POSITION;
	setMaxForce(hkReal(1e6f));
	m_tau = hkReal(0.8f);
	m_damping = hkReal(1);
	m_constantRecoveryVelocity = hkReal(1);
	m_proportionalRecoveryVelocity = hkReal(2);
}

hkpConstraintMotor* hkpPositionConstraintMotor::clone() const
{
	hkpPositionConstraintMotor* pcm = new hkpPositionConstraintMotor( *this );
	return pcm;
}

HK_COMPILE_TIME_ASSERT( sizeof(hkpPositionConstraintMotor) <= sizeof(hkpMaxSizeConstraintMotor) );
#if ( HK_POINTER_SIZE == 4 ) && !defined(HK_REAL_IS_DOUBLE)
HK_COMPILE_TIME_ASSERT( sizeof(hkpMaxSizeConstraintMotor) == 48 );
#endif

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
