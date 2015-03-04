/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Physics/Constraint/Motor/SpringDamper/hkpSpringDamperConstraintMotor.h>


hkpSpringDamperConstraintMotor::hkpSpringDamperConstraintMotor()
: hkpLimitedForceConstraintMotor()
{
	m_type = TYPE_SPRING_DAMPER;
	setMaxForce(hkReal(1e6f));
	m_springConstant = hkReal(0);
	m_springDamping = hkReal(0);
}

// Construct a motor with the given properties.
hkpSpringDamperConstraintMotor::hkpSpringDamperConstraintMotor( hkReal springConstant, hkReal springDamping, hkReal maxForce )
: hkpLimitedForceConstraintMotor()
{
	m_type = TYPE_SPRING_DAMPER;
	setMaxForce(maxForce);
	m_springConstant = springConstant;
	m_springDamping = springDamping;
}

hkpConstraintMotor* hkpSpringDamperConstraintMotor::clone() const
{
	hkpSpringDamperConstraintMotor* sdcm = new hkpSpringDamperConstraintMotor( *this );
	return sdcm;
}


HK_COMPILE_TIME_ASSERT( sizeof(hkpSpringDamperConstraintMotor) <= sizeof(hkpMaxSizeConstraintMotor) );

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
