/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/hknpViewerColorScheme.h>

// Force explicit template instantiation
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase< hknpDefaultViewerColorScheme::Uid, hkColor::Argb, hknpDefaultViewerColorScheme::UIdMapOperations >;

const hkColor::Argb hknpDefaultViewerColorScheme::STATIC_BODY_COLOR = 0xFF999999;
const hkColor::Argb hknpDefaultViewerColorScheme::DYNAMIC_BODY_COLOR = 0xFFFFB300;
const hkColor::Argb hknpDefaultViewerColorScheme::KEYFRAMED_BODY_COLOR = 0xFF800080;
const hkColor::Argb hknpDefaultViewerColorScheme::TRIGGER_VOLUME_COLOR = 0x4400FF00;
const hkColor::Argb hknpDefaultViewerColorScheme::DONT_COLLIDE_COLOR = 0x66FFFF00;


hknpDefaultViewerColorScheme::hknpDefaultViewerColorScheme()
:	m_staticBodyColor(STATIC_BODY_COLOR),
	m_dynamicBodyColor(DYNAMIC_BODY_COLOR),
	m_keyframedBodyColor(KEYFRAMED_BODY_COLOR),
	m_triggerVolumeColor(TRIGGER_VOLUME_COLOR),
	m_dontCollideColor(DONT_COLLIDE_COLOR)
{

}

void hknpDefaultViewerColorScheme::setBodyColor( const hknpWorld* world, hknpBodyId bodyId, hkColor::Argb color, hknpViewer* viewer )
{
	Uid uId(world, bodyId);
	m_overriddenColors.insert(uId, color);
}

void hknpDefaultViewerColorScheme::clearBodyColor( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer )
{
	Uid uId(world, bodyId);
	m_overriddenColors.remove(uId);
}

hkColor::Argb hknpDefaultViewerColorScheme::getBodyColor( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer ) const
{
	// Check for body color override
	{
		hkColor::Argb color = hkColor::BLACK;
		Uid uId(world, bodyId);
		if( m_overriddenColors.get( uId, &color ) == HK_SUCCESS )
		{
			return color;
		}
	}

	// Check for body color property
	{
		hkColor::Argb* color = world->getBodyProperty<hkColor::Argb>( bodyId, hknpBodyPropertyKeys::DEBUG_DISPLAY_COLOR );
		if( color != HK_NULL )
		{
			return *color;
		}
	}

	const hknpBody& body = world->getBody( bodyId );
	const hknpMaterial& material = world->getMaterialLibrary()->getEntry( body.m_materialId );

	if( body.m_flags.get( hknpBody::DONT_COLLIDE ) )
	{
		return m_dontCollideColor;
	}
	if ( material.m_flags.get(hknpMaterial::ENABLE_TRIGGER_VOLUME) )	
	{
		return m_triggerVolumeColor;
	}
	if ( body.isStatic() )
	{
		return m_staticBodyColor;
	}
	if ( body.isKeyframed() )
	{
		return m_keyframedBodyColor;
	}
	return m_dynamicBodyColor;
}

bool hknpDefaultViewerColorScheme::isBodyVisible( const hknpWorld* world, hknpBodyId bodyId, hknpViewer* viewer ) const
{
	return true;
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
