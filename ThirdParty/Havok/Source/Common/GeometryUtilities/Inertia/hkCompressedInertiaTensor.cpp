/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h> // Precompiled Header

#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Common/GeometryUtilities/Inertia/hkCompressedInertiaTensor.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

void hkCompressedMassProperties::pack( const hkMassProperties& mP )
{
	m_centerOfMass.pack( mP.m_centerOfMass );
	m_mass = mP.m_mass;
	m_volume = mP.m_volume;

	hkRotation principalAxis;
	hkMatrix3 inertia = mP.m_inertiaTensor;
	hkInertiaTensorComputer::convertInertiaTensorToPrincipleAxis( inertia, principalAxis );
	hkVector4 inertiaDiagonal;
	hkMatrix3Util::_getDiagonal(inertia, inertiaDiagonal);
	m_inertia.pack( inertiaDiagonal );
	hkQuaternion prinipleAxisQ; prinipleAxisQ.set( principalAxis );
	m_majorAxisSpace.pack( prinipleAxisQ.m_vec );
}

void hkCompressedMassProperties::pack( const hkDiagonalizedMassProperties& massPropertiesIn )
{
	m_centerOfMass.pack(massPropertiesIn.m_centerOfMass);
	m_inertia.pack(massPropertiesIn.m_inertiaTensor);
	m_majorAxisSpace.pack(massPropertiesIn.m_majorAxisSpace.m_vec);
	
	m_mass		= massPropertiesIn.m_mass;
	m_volume	= massPropertiesIn.m_volume;
}

void hkCompressedMassProperties::unpack( hkMassProperties& props ) const 
{
	hkVector4 inertia; m_inertia.unpack( inertia );
	hkQuaternion q; m_majorAxisSpace.unpack( &q.m_vec );
	q.normalize();
	hkRotation r; r.set(q);
	hkRotation r2;
	r2.getColumn(0).setMul( inertia.getComponent<0>(), r.getColumn<0>());
	r2.getColumn(1).setMul( inertia.getComponent<1>(), r.getColumn<1>());
	r2.getColumn(2).setMul( inertia.getComponent<2>(), r.getColumn<2>());

	props.m_inertiaTensor.setMulInverse( r2, r ); 

	props.m_volume = m_volume;
	m_centerOfMass.unpack( props.m_centerOfMass );
	props.m_mass   = m_mass;
}

void hkDiagonalizedMassProperties::pack( const hkMassProperties& mp )
{
	m_centerOfMass = mp.m_centerOfMass;
	m_mass   = mp.m_mass;
	m_volume = mp.m_volume;

	hkRotation principalAxis;
	hkMatrix3 inertia = mp.m_inertiaTensor;
	hkInertiaTensorComputer::convertInertiaTensorToPrincipleAxis( inertia, principalAxis );
	hkMatrix3Util::_getDiagonal(inertia, m_inertiaTensor);
	hkSimdReal vol; vol.load<1>(&mp.m_volume);
	m_inertiaTensor.setW(vol);
	m_majorAxisSpace.set( principalAxis );
}

void hkCompressedMassProperties::unpack( hkDiagonalizedMassProperties* HK_RESTRICT props ) const 
{
	hkQuaternion q; m_majorAxisSpace.unpack( &q.m_vec );
	q.normalize();
	m_inertia.unpack( props->m_inertiaTensor );
	m_centerOfMass.unpack( props->m_centerOfMass );
	props->m_volume = m_volume;
	props->m_mass   = m_mass;
	props->m_majorAxisSpace = q;
}

void hkDiagonalizedMassProperties::unpack( hkMassProperties* HK_RESTRICT props ) const
{
	hkRotation r; r.set( m_majorAxisSpace );
	hkRotation r2;
	r2.getColumn(0).setMul( m_inertiaTensor.getComponent<0>(), r.getColumn<0>());
	r2.getColumn(1).setMul( m_inertiaTensor.getComponent<1>(), r.getColumn<1>());
	r2.getColumn(2).setMul( m_inertiaTensor.getComponent<2>(), r.getColumn<2>());

	props->m_inertiaTensor.setMulInverse( r2, r ); 
	props->m_volume = m_volume;
	props->m_centerOfMass = m_centerOfMass;
	props->m_mass = m_mass;

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
