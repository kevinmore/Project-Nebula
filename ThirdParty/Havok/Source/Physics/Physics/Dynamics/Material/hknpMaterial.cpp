/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Material/hknpMaterial.h>


hknpMaterial::hknpMaterial()
{
	hkString::memSet( this, 0, sizeof(*this) );

	m_dynamicFriction.setReal<true>(0.5f);
	m_staticFriction.setReal<true>(0.5f);
	m_frictionCombinePolicy = hknpMaterial::COMBINE_MIN;

	m_restitutionCombinePolicy = hknpMaterial::COMBINE_MAX;

	m_triggerVolumeType = TRIGGER_VOLUME_NONE;
	m_triggerVolumeTolerance = HK_REAL_MAX;

	m_maxContactImpulse = HK_REAL_MAX;
	m_fractionOfClippedImpulseToApply = 1.0f;
	m_weldingTolerance.setReal<false>( 0.05f );

	m_massChangerCategory = hknpMaterial::MASS_CHANGER_IGNORE;
	m_massChangerHeavyObjectFactor.setOne();

	m_disablingCollisionsBetweenCvxCvxDynamicObjectsDistance.setReal<false>( 5.0f );
}

hknpMaterial::hknpMaterial( hkFinishLoadedObjectFlag flag ) : m_name(flag)
{

}

hknpMaterial::hknpMaterial( const hknpMaterial& other )
{
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF( hknpMaterial, m_name ) == 0 );
	m_name = other.m_name;
	int sizeofName = hkSizeOf( m_name );
	hkString::memCpy( hkAddByteOffset(this, sizeofName), hkAddByteOffsetConst(&other,sizeofName), hkSizeOf(hknpMaterial)-sizeofName );
}

void hknpMaterial::synchronizeFlags()
{
	hkUint32 flags = 0;

	if( !m_restitution.isZero() )
	{
		flags |= hknpMaterial::ENABLE_RESTITUTION;
	}
	if( m_triggerVolumeType != TRIGGER_VOLUME_NONE )
	{
		flags |= hknpMaterial::ENABLE_TRIGGER_VOLUME;
	}
	if( m_maxContactImpulse != HK_REAL_MAX )
	{
		flags |= hknpMaterial::ENABLE_IMPULSE_CLIPPING;
	}
	if( m_massChangerCategory == hknpMaterial::MASS_CHANGER_HEAVY )
	{
		flags |= hknpMaterial::ENABLE_MASS_CHANGER;
	}
	if( m_softContactForceFactor.getReal() != hkReal(0) )
	{
		flags |= hknpMaterial::ENABLE_SOFT_CONTACTS;
	}
	if( m_surfaceVelocity )
	{
		flags |= hknpMaterial::ENABLE_SURFACE_VELOCITY;
	}

	HK_ASSERT( 0x5e19ebfa, (flags & AUTO_FLAGS_MASK) == flags );
	m_flags.setWithMask( flags, AUTO_FLAGS_MASK );
}


hknpMaterialDescriptor::hknpMaterialDescriptor( hkFinishLoadedObjectFlag flag )
	:	m_name(flag)
	,	m_material(flag)
{}


hknpRefMaterial::hknpRefMaterial( hkFinishLoadedObjectFlag flag )
	:	hkReferencedObject(flag)
	,	m_material(flag)
{
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
